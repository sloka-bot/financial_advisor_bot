"""Forecast horizon returns using a two-layer LSTM over rolling feature sequences."""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from backend.config.settings import PREDICTION_HORIZON
from backend.prediction.temporal_split import date_cutoff

logger = logging.getLogger(__name__)

SEQ_LEN = 30
MODELS_DIR = Path("models/lstm")

# Column names match feature_engineer.py output.
TRAIN_FEATURES = [
    "sma20",
    "sma50",
    "ema12",
    "ema26",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi",
    "stoch_k",
    "williams_r",
    "atr_pct",
    "bb_pct",
    "volatility",
    "momentum_10d",
    "momentum_21d",
    "daily_return",
    "weekly_return",
    "monthly_return",
    "volume_ratio",
    "close_to_sma20",
    "adx",
    "cci",
    "sent_score",
    "sent_news_count",
]


# Use two recurrent layers, 128 hidden units, output dropout and validation early stopping.
def _build_net(input_size, hidden=128, layers=2, dropout=0.2):
    import torch.nn as nn

    class Net(nn.Module):
        """Map a sequence of feature vectors to a single return estimate."""

        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden,
                num_layers=layers,
                dropout=0.0,  # dropout applied to the final hidden state
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            """Apply the recurrent layers and project the final timestep to a return."""
            out, _ = self.lstm(x)
            # Make the final hidden-state slice contiguous before the output layer.
            last = out[:, -1, :].contiguous()
            return self.fc(self.dropout(last)).squeeze(-1)

    return Net()


class _WindowStore:
    """Index overlapping windows without allocating a full sequence tensor."""

    def __init__(self, arrays, anchors, seq_len):
        self.arrays = arrays
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.seq_len = seq_len
        self.shape = (len(self.anchors), seq_len, arrays[0].shape[1])

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            group, end = self.anchors[key]
            return self.arrays[group][end - self.seq_len + 1 : end + 1]
        return _WindowStore(self.arrays, self.anchors[key], self.seq_len)


class _WindowTargets:
    """Expose lazily sliced windows and their observed labels to a loader."""

    def __init__(self, windows, targets):
        self.windows, self.targets = windows, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.windows[index], self.targets[index]


class LSTMForecaster:
    """Fit a sequence model for the forward return target."""

    def __init__(self, seq_len=SEQ_LEN, hidden=128, layers=2, dropout=0.2, models_dir=None, horizon=None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.horizon = int(horizon) if horizon is not None else PREDICTION_HORIZON
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def train(
        self, df: pd.DataFrame, epochs=60, lr=0.001, batch_size=32, patience=10, progress_cb=None, persist: bool = True
    ) -> dict:
        """Fit chronologically split sequences and persist the selected return model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        # Limit CPU thread contention and initialise repeatable random seeds.
        torch.set_num_threads(1)
        torch.manual_seed(42)
        np.random.seed(42)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        X_seq, y_seq, val_mask = self._build_sequences(df, fit_scaler=True, lazy=True)
        if X_seq is None or len(X_seq) < 50:
            return {"error": "Not enough sequence data"}

        # Split sequences chronologically using a scaler fitted only on training rows.
        if val_mask is not None and val_mask.any() and (~val_mask).any():
            X_tr, X_val = X_seq[~val_mask], X_seq[val_mask]
            y_tr, y_val = y_seq[~val_mask], y_seq[val_mask]
        else:
            # Reject missing date boundaries rather than splitting concatenated ticker rows.
            raise RuntimeError(
                "LSTM training requires a valid chronological validation split "
                "(a row-position fallback would leak across tickers)."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = DataLoader(_WindowTargets(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(_WindowTargets(X_val, y_val), batch_size=batch_size)
        self.model = _build_net(X_tr.shape[2], self.hidden, self.layers, self.dropout).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        best_val = float("inf")
        best_w = None
        no_improve = 0
        history = []

        logger.info(
            f"LSTM training: {len(X_seq)} sequences on {device}  ({epochs} epochs max, early-stop patience {patience})"
        )
        _t_start = time.time()

        for epoch in range(1, epochs + 1):
            _t_epoch = time.time()
            self.model.train()
            train_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_tr)

            self.model.eval()
            val_total, correct, count = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_preds = self.model(xb)
                    val_total += criterion(val_preds, yb).item() * len(yb)
                    correct += int((torch.sign(val_preds) == torch.sign(yb)).sum().item())
                    count += len(yb)
            val_loss = val_total / count
            scheduler.step(val_loss)
            dir_acc = correct / count
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "dir_acc": round(dir_acc, 4),
                }
            )

            # Log epoch duration and estimated remaining training time.
            _dt = time.time() - _t_epoch
            _eta = _dt * (epochs - epoch)
            logger.info(
                f"  Epoch {epoch:3d}/{epochs}  val_loss={val_loss:.6f}  "
                f"dir_acc={dir_acc:.3f}  ({_dt:.1f}s/epoch, ~{_eta / 60:.1f} min left)"
            )
            if progress_cb is not None:
                try:
                    progress_cb(epoch, epochs, val_loss, dir_acc)
                except Exception:
                    pass  # ignore progress callback errors

            if val_loss < best_val:
                best_val = val_loss
                best_w = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

        if best_w:
            self.model.load_state_dict(best_w)

        if persist:
            self.save()

        best_epoch = min(history, key=lambda x: x["val_loss"])
        _dt_index = isinstance(df.index, pd.DatetimeIndex)
        training_results = {
            "best_val_loss": best_epoch["val_loss"],
            "best_dir_acc": best_epoch["dir_acc"],
            "best_epoch": best_epoch["epoch"],
            "total_epochs": len(history),
            "history": history,
            "trained_at": datetime.now().isoformat(),
            "horizon": self.horizon,
            "config": {
                "epochs_max": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "patience": patience,
                "seq_len": self.seq_len,
                "hidden": self.hidden,
                "layers": self.layers,
                "dropout": self.dropout,
            },
            "n_sequences": int(len(X_seq)),
            "date_range": [str(df.index.min())[:10], str(df.index.max())[:10]] if _dt_index else None,
        }

        # Save training history for the validation dashboard.
        if persist:
            with open(self.models_dir / "training_history.json", "w") as f:
                json.dump(training_results, f, indent=2)

        logger.info(
            f"LSTM saved - best val_loss={best_epoch['val_loss']:.6f} "
            f"dir_acc={best_epoch['dir_acc']:.3f} (epoch {best_epoch['epoch']})"
        )
        return training_results

    def predict_ticker(self, df: pd.DataFrame) -> float | None:
        """Predict the horizon return from the latest complete feature sequence."""
        if not self._loaded():
            return None

        import torch

        # Return unavailable when a required feature is missing.
        missing = [c for c in TRAIN_FEATURES if c not in df.columns]
        if missing:
            logger.warning(f"LSTM predict: required features absent {missing}; prediction unavailable")
            return None
        X_df = df[TRAIN_FEATURES].ffill()  # forward-fill warm-up gaps only

        if len(X_df) < self.seq_len:
            return None

        window_df = X_df.tail(self.seq_len)
        if window_df.isna().to_numpy().any():
            logger.warning("LSTM predict: unobserved (NaN) features in the latest window; prediction unavailable")
            return None
        window = window_df.values
        if window.shape != (self.seq_len, len(TRAIN_FEATURES)):
            logger.warning(
                f"LSTM predict {getattr(df, 'shape', '?')}: unexpected window shape "
                f"{window.shape}; prediction unavailable"
            )
            return None
        try:
            window = self.scaler.transform(window)
        except Exception as e:
            # Reject inference when the fitted scaler cannot transform the inputs.
            logger.warning(f"LSTM predict: scaler.transform failed ({e}); prediction unavailable")
            return None
        if not np.isfinite(window).all():
            logger.warning("LSTM predict: non-finite scaled features; prediction unavailable")
            return None

        seq = torch.tensor(window, dtype=torch.float32, device=next(self.model.parameters()).device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(seq).item())

    def predict_panel(self, history, observations, batch_size=256):
        """Predict dated ticker rows in batches while preserving observation order."""
        if not self._loaded():
            raise ValueError("LSTM model is unavailable")
        import torch

        result = np.full(len(observations), np.nan)
        positions = observations.assign(_position=np.arange(len(observations)))
        self.model.eval()
        device = next(self.model.parameters()).device
        grouped = history.groupby("ticker", sort=False)
        for ticker, requested in positions.groupby("ticker", sort=False):
            if ticker not in grouped.indices:
                continue
            sub = grouped.get_group(ticker).sort_index()
            sub = sub[~sub.index.duplicated(keep="last")]
            missing = [c for c in TRAIN_FEATURES if c not in sub.columns]
            if missing:
                raise ValueError(f"LSTM predict_panel: {ticker} missing required features {missing}")
            features = sub[TRAIN_FEATURES].ffill()  # non-finite windows raise below
            values = self.scaler.transform(features.to_numpy()).astype(np.float32)
            ends = sub.index.get_indexer(requested.index)
            eligible = ends >= self.seq_len - 1
            ends = ends[eligible]
            slots = requested["_position"].to_numpy()[eligible]
            for begin in range(0, len(ends), batch_size):
                batch_ends = ends[begin : begin + batch_size]
                windows = np.stack([values[end - self.seq_len + 1 : end + 1] for end in batch_ends])
                if not np.isfinite(windows).all():
                    raise ValueError(f"Non-finite LSTM inputs for {ticker}")
                with torch.no_grad():
                    pred = self.model(torch.as_tensor(windows, device=device)).cpu().numpy()
                result[slots[begin : begin + batch_size]] = pred
        return result

    def is_trained(self) -> bool:
        """Check whether the saved sequence-model artifact exists."""
        return (self.models_dir / "model.pt").exists()

    def save(self):
        """Persist network weights, scaler and the selected feature schema."""
        import torch

        from backend.infra.model_registry import artifact_fingerprint as _artifact_fingerprint

        self.models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.models_dir / "model.pt")
        with open(self.models_dir / "meta.pkl", "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "seq_len": self.seq_len,
                    "hidden": self.hidden,
                    "layers": self.layers,
                    "dropout": self.dropout,
                    "feature_names": list(TRAIN_FEATURES),
                    "fingerprint": _artifact_fingerprint(list(TRAIN_FEATURES)),
                },
                f,
            )
        logger.info(f"LSTM saved to {self.models_dir}")

    def _loaded(self) -> bool:
        if self.model is not None:
            return True
        mp = self.models_dir / "model.pt"
        pp = self.models_dir / "meta.pkl"
        if not mp.exists() or not pp.exists():
            return False
        import torch

        with open(pp, "rb") as f:
            meta = pickle.load(f)
        self.scaler = meta["scaler"]
        self.seq_len = meta["seq_len"]
        self.hidden = meta["hidden"]
        self.layers = meta["layers"]
        self.dropout = meta["dropout"]
        saved_features = meta.get("feature_names")
        if saved_features is not None and list(saved_features) != list(TRAIN_FEATURES):
            raise RuntimeError("Saved LSTM feature schema does not match current TRAIN_FEATURES; retrain the model.")
        self.feature_names = list(saved_features) if saved_features is not None else list(TRAIN_FEATURES)
        fingerprint = meta.get("fingerprint")
        if fingerprint is not None:
            from backend.infra.model_registry import check_artifact_compatibility

            ok, issues = check_artifact_compatibility(fingerprint, self.feature_names)
            if not ok:
                raise RuntimeError(
                    "Saved LSTM artifact is incompatible with the current environment: "
                    + "; ".join(issues)
                    + "; retrain the model."
                )
            if issues:
                logger.warning("LSTM artifact compatibility warnings: %s", "; ".join(issues))
        self.model = _build_net(len(self.feature_names), self.hidden, self.layers, self.dropout)
        self.model.load_state_dict(torch.load(mp, map_location="cpu"))
        self.model.eval()
        return True

    def _build_sequences(self, df: pd.DataFrame, fit_scaler: bool = True, val_frac: float = 0.2, lazy: bool = False):
        df = df.copy()
        # Check required features before building training sequences.
        missing = [c for c in TRAIN_FEATURES if c not in df.columns]
        if missing:
            raise RuntimeError(f"LSTM training is missing required features: {missing}")

        X_all, y_all, val_all = [], [], []
        scaled_arrays, anchors = [], []
        tickers = df["ticker"].unique() if "ticker" in df.columns else ["all"]

        # Assemble labelled ticker arrays and fit the scaler before the validation boundary.
        prepared = []
        for t in tickers:
            sub = df[df["ticker"] == t] if "ticker" in df.columns else df
            has_target = "target_return" in sub.columns
            need = TRAIN_FEATURES + (["target_return"] if has_target else [])
            sub = sub[need + []].copy()
            # Fill features only; rows with a missing label are dropped.
            sub[TRAIN_FEATURES] = sub[TRAIN_FEATURES].replace([float("inf"), float("-inf")], np.nan)
            sub[TRAIN_FEATURES] = sub[TRAIN_FEATURES].ffill()
            # Exclude rows with incomplete indicators before scaling.
            sub = sub.dropna(subset=TRAIN_FEATURES)
            if has_target:
                sub = sub[sub["target_return"].notna()]
            if len(sub) < self.seq_len + 10:
                continue
            feats = sub[TRAIN_FEATURES].values
            targets = sub.get("target_return", pd.Series(np.zeros(len(sub)))).values
            dates = pd.to_datetime(sub.index).values
            prepared.append((feats, targets, dates))

        if not prepared:
            return None, None, None

        # Purge training anchors whose forward labels reach the validation window.
        is_dt = isinstance(df.index, pd.DatetimeIndex)
        cutoff = None
        if is_dt:
            # Use the latest fraction of unique trading dates for validation.
            all_dates = np.concatenate([d for _, _, d in prepared])
            cutoff = date_cutoff(all_dates, val_frac)
        purge = self.horizon

        def _first_val_pos(feats, dates):
            if is_dt:
                vp = np.where(pd.to_datetime(dates) >= cutoff)[0]
                return int(vp[0]) if len(vp) else len(feats)
            return int(len(feats) * (1.0 - val_frac))

        if fit_scaler:
            for feats, _, dates in prepared:
                fv = _first_val_pos(feats, dates)
                if fv > 0:
                    self.scaler.partial_fit(feats[:fv])  # training rows only

        # End each feature window at the anchor of its forward-return target.
        for feats, targets, dates in prepared:
            feats_sc = self.scaler.transform(feats).astype(np.float32)
            scaled_arrays.append(feats_sc)
            first_val = _first_val_pos(feats, dates)
            for i in range(self.seq_len - 1, len(feats_sc)):
                is_val = i >= first_val
                if (not is_val) and (first_val - i) <= purge:
                    continue  # label window overlaps the validation block
                anchors.append((len(scaled_arrays) - 1, i))
                if not lazy:
                    X_all.append(feats_sc[i - self.seq_len + 1 : i + 1])
                y_all.append(float(targets[i]))
                val_all.append(bool(is_val))

        if not anchors:
            return None, None, None
        windows = _WindowStore(scaled_arrays, anchors, self.seq_len) if lazy else np.array(X_all, dtype=np.float32)
        return windows, np.array(y_all, dtype=np.float32), np.array(val_all, dtype=bool)

    def update(self, df: pd.DataFrame, epochs: int = 10) -> dict:
        """Fine-tune existing weights at a lower learning rate and roll back worse validation loss."""
        import copy

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        torch.set_num_threads(1)  # limit CPU threads

        if not self._loaded():
            raise RuntimeError("LSTM.update() called before any model has been trained. Run train() first.")

        X_seq, y_seq, val_mask = self._build_sequences(df, fit_scaler=False, lazy=True)
        if X_seq is None or len(X_seq) < 20:
            logger.warning(f"LSTM update: only {len(X_seq) if X_seq is not None else 0} sequences, skipping")
            return {"mode": "skipped", "reason": "too_few_sequences"}

        have_val = val_mask is not None and val_mask.any() and (~val_mask).any()
        if have_val:
            X_tr, y_tr = X_seq[~val_mask], y_seq[~val_mask]
            X_val, y_val = X_seq[val_mask], y_seq[val_mask]
        else:
            # Skip fine-tuning when a chronological validation split is unavailable.
            logger.warning("LSTM update: no valid validation split; skipping update")
            return {"mode": "skipped", "reason": "no_validation_split"}

        criterion = nn.MSELoss()

        def _val_loss():
            if X_val is None or len(X_val) == 0:
                return None
            self.model.eval()
            with torch.no_grad():
                total = 0.0
                for xb, yb in DataLoader(_WindowTargets(X_val, y_val), batch_size=256):
                    device = next(self.model.parameters()).device
                    pred = self.model(xb.to(device))
                    total += criterion(pred, yb.to(device)).item() * len(yb)
                return total / len(y_val)

        before = _val_loss()
        snapshot = copy.deepcopy(self.model.state_dict())  # kept for rollback

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        loader = DataLoader(
            _WindowTargets(X_tr, y_tr),
            batch_size=32,
            shuffle=True,
        )
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                device = next(self.model.parameters()).device
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.info(f"  LSTM fine-tune epoch {epoch + 1}/{epochs}  loss={epoch_loss / len(loader):.6f}")

        self.model.eval()
        after = _val_loss()

        # Keep the update only if validation loss did not worsen.
        if before is not None and after is not None and after > before * 1.01:
            self.model.load_state_dict(snapshot)  # restore pre-update weights
            self.model.eval()
            logger.warning(
                f"LSTM update rolled back: val loss worsened {before:.6f} -> {after:.6f}; kept the previous model"
            )
            return {
                "mode": "rolled_back",
                "epochs": epochs,
                "val_loss_before": round(before, 6),
                "val_loss_after": round(after, 6),
                "updated_at": datetime.now().isoformat(),
            }

        self.save()
        logger.info(f"LSTM fine-tuned for {epochs} epochs (val {before} -> {after})")
        return {
            "mode": "fine_tune",
            "epochs": epochs,
            "val_loss_before": round(before, 6) if before is not None else None,
            "val_loss_after": round(after, 6) if after is not None else None,
            "updated_at": datetime.now().isoformat(),
        }
