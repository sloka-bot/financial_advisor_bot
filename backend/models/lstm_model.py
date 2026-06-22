import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

SEQ_LEN    = 30
MODELS_DIR = Path('models/lstm')

# column names must match feature_engineer.py output exactly
TRAIN_FEATURES = [
    'sma20', 'sma50', 'ema12', 'ema26',
    'macd', 'macd_signal', 'macd_hist',
    'rsi', 'stoch_k', 'williams_r',
    'atr_pct', 'bb_pct', 'volatility',
    'momentum_10d', 'momentum_21d',
    'daily_return', 'weekly_return', 'monthly_return',
    'volume_ratio', 'close_to_sma20',
    'adx', 'cci',
    'sent_score', 'sent_news_count',
]


def _build_net(input_size, hidden=128, layers=2, dropout=0.2):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # stacked LSTM: lower layers extract short-range patterns,
            # upper layers integrate them into longer-horizon structure
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            # use only the last hidden state — the LSTM's summary of the sequence
            return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)

    return Net()


class LSTMForecaster:

    def __init__(self, seq_len=SEQ_LEN, hidden=128, layers=2, dropout=0.2):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.hidden  = hidden
        self.layers  = layers
        self.dropout = dropout
        self.model   = None
        self.scaler  = MinMaxScaler(feature_range=(-1, 1))

    def train(self, df: pd.DataFrame, epochs=60, lr=0.001, batch_size=32, patience=10) -> dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        X_seq, y_seq = self._build_sequences(df)
        if X_seq is None or len(X_seq) < 50:
            return {'error': 'Not enough sequence data'}

        split   = int(len(X_seq) * 0.8)
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]

        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tr_t  = torch.FloatTensor(X_tr).to(device)
        y_tr_t  = torch.FloatTensor(y_tr).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)

        loader    = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
        self.model = _build_net(X_tr.shape[2], self.hidden, self.layers, self.dropout).to(device)
        opt        = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion  = nn.MSELoss()
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        best_val   = float('inf')
        best_w     = None
        no_improve = 0
        history    = []

        logger.info(f'LSTM training: {len(X_seq)} sequences on {device}')

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_tr)

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t)
                val_loss  = criterion(val_preds, y_val_t).item()

            scheduler.step(val_loss)

            preds_np   = val_preds.cpu().numpy()
            actuals_np = y_val_t.cpu().numpy()
            dir_acc    = float(np.mean(np.sign(preds_np) == np.sign(actuals_np)))
            history.append({'epoch': epoch, 'train_loss': round(train_loss, 6),
                            'val_loss': round(val_loss, 6), 'dir_acc': round(dir_acc, 4)})

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f'  Epoch {epoch:3d}  val_loss={val_loss:.6f}  dir_acc={dir_acc:.3f}')

            if val_loss < best_val:
                best_val   = val_loss
                best_w     = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f'  Early stop at epoch {epoch}')
                    break

        if best_w:
            self.model.load_state_dict(best_w)

        self.save()

        best_epoch = min(history, key=lambda x: x['val_loss'])
        training_results = {
            'best_val_loss':  best_epoch['val_loss'],
            'best_dir_acc':   best_epoch['dir_acc'],
            'best_epoch':     best_epoch['epoch'],
            'total_epochs':   len(history),
            'history':        history,
            'trained_at':     datetime.now().isoformat(),
        }

        # persist training history so the validation dashboard can plot it
        with open(MODELS_DIR / 'training_history.json', 'w') as f:
            json.dump(training_results, f, indent=2)

        logger.info(f'LSTM saved — best val_loss={best_epoch["val_loss"]:.6f} '
                    f'dir_acc={best_epoch["dir_acc"]:.3f} (epoch {best_epoch["epoch"]})')
        return training_results

    def predict_ticker(self, df: pd.DataFrame) -> float | None:
        if not self._loaded():
            return None

        import torch

        avail = [c for c in TRAIN_FEATURES if c in df.columns]
        X_df  = df[avail].copy()

        # pad any missing columns with zeros so the sequence shape is consistent
        for col in TRAIN_FEATURES:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[TRAIN_FEATURES].ffill().fillna(0.0)

        if len(X_df) < self.seq_len:
            return None

        window = X_df.tail(self.seq_len).values
        try:
            window = self.scaler.transform(window)
        except Exception:
            pass

        seq = torch.FloatTensor(window).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(seq).item())

    def is_trained(self) -> bool:
        return (MODELS_DIR / 'model.pt').exists()

    def save(self):
        import torch
        torch.save(self.model.state_dict(), MODELS_DIR / 'model.pt')
        with open(MODELS_DIR / 'meta.pkl', 'wb') as f:
            pickle.dump({
                'scaler':  self.scaler,
                'seq_len': self.seq_len,
                'hidden':  self.hidden,
                'layers':  self.layers,
                'dropout': self.dropout,
            }, f)
        logger.info(f'LSTM saved to {MODELS_DIR}')

    def _loaded(self) -> bool:
        if self.model is not None:
            return True
        mp = MODELS_DIR / 'model.pt'
        pp = MODELS_DIR / 'meta.pkl'
        if not mp.exists() or not pp.exists():
            return False
        import torch
        with open(pp, 'rb') as f:
            meta = pickle.load(f)
        self.scaler  = meta['scaler']
        self.seq_len = meta['seq_len']
        self.hidden  = meta['hidden']
        self.layers  = meta['layers']
        self.dropout = meta['dropout']
        self.model   = _build_net(len(TRAIN_FEATURES), self.hidden, self.layers, self.dropout)
        self.model.load_state_dict(torch.load(mp, map_location='cpu'))
        self.model.eval()
        return True

    def _build_sequences(self, df: pd.DataFrame):
        # ensure all expected columns exist before building windows
        for col in TRAIN_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        X_all, y_all = [], []
        tickers = df['ticker'].unique() if 'ticker' in df.columns else ['all']

        for t in tickers:
            sub = df[df['ticker'] == t] if 'ticker' in df.columns else df
            need = TRAIN_FEATURES + ['target_return'] if 'target_return' in sub.columns else TRAIN_FEATURES
            sub  = sub[need].ffill().fillna(0.0).dropna()

            if len(sub) < self.seq_len + 10:
                continue

            feats   = sub[TRAIN_FEATURES].values
            targets = sub.get('target_return', pd.Series(np.zeros(len(sub)))).values

            self.scaler.partial_fit(feats)
            feats_sc = self.scaler.transform(feats)

            for i in range(self.seq_len, len(sub)):
                X_all.append(feats_sc[i - self.seq_len: i])
                y_all.append(float(targets[i]))

        if not X_all:
            return None, None
        return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)
