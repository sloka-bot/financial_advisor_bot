"""Persist user portfolios and recommendations with atomic, process-locked updates."""

import json
import logging
import os
import shutil
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)

# User data is stored in one JSON file in the data directory.
STORE = Path("data/users.json")

# One reentrant lock serialises all load -> modify -> save cycles in this process.
_LOCK = threading.RLock()


def _locked(fn):
    """Hold the reentrant store lock for the entire read-modify-write operation."""

    @wraps(fn)
    def wrap(self, *args, **kwargs):
        """Serialize access to the user store with the shared lock."""
        with _LOCK:
            return fn(self, *args, **kwargs)

    return wrap


class UserStore:
    """Persist profiles, holdings and recommendation decisions in a local JSON store."""

    def _load(self) -> dict:
        """Read saved users; preserve and reject corrupt files rather than treating them as empty."""
        if not STORE.exists():
            return {}
        text = STORE.read_text()
        try:
            return json.loads(text).get("users", {})
        except Exception as e:
            backup = STORE.with_suffix(".corrupt")
            try:
                shutil.copy2(STORE, backup)
            except Exception:
                pass
            logger.error(f"user store corrupt; backed up to {backup.name}; file left unchanged: {e}")
            raise RuntimeError(f"user store corrupt (backed up to {backup.name})") from e

    # Write the users dict atomically.
    def _save(self, users: dict):
        STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"users": users}, indent=2))
        os.replace(tmp, STORE)  # atomic on POSIX and Windows

    @_locked
    def get(self, user_id: str) -> dict | None:
        """Return the saved profile for a user, or None if it does not exist."""
        return self._load().get(user_id)

    @_locked
    def is_new(self, user_id: str) -> bool:
        """Check whether a user has no saved profile."""
        return user_id not in self._load()

    @_locked
    def create_profile(self, user_id: str, profile: dict) -> dict:
        """Write a new user record from the onboarding questionnaire answers."""
        users = self._load()
        if user_id in users:
            return users[user_id]
        now = datetime.now().isoformat()
        users[user_id] = {
            "user_id": user_id,
            "name": profile.get("name", ""),
            "risk_profile": profile.get("risk_profile", "moderate"),
            "investment_horizon": profile.get("investment_horizon", "5-10 years"),
            "goal": profile.get("goal", "growth"),
            "budget": float(profile.get("budget", 10000)),
            "monthly_contribution": float(profile.get("monthly_contribution", 0)),
            "market": profile.get("market", "United States"),
            "index": profile.get("index", "S&P 500"),
            "created_at": now,
            "updated_at": now,
            "portfolio": {
                "holdings": [],
                "cash": float(profile.get("budget", 10000)),
                "created_at": now,
                "updated_at": now,
            },
            "pending_recommendations": [],
            "recommendation_history": [],
            "next_rec_id": 1,  # per-user monotonic recommendation id
        }
        self._save(users)
        logger.info(f"Profile created for {user_id}")
        return users[user_id]

    @_locked
    def update_profile(self, user_id: str, updates: dict) -> dict:
        """Update profile fields without changing saved holdings or cash balances."""
        users = self._load()
        if user_id not in users:
            return self.create_profile(user_id, updates)
        new_budget = updates.get("budget", users[user_id]["budget"])
        holdings = users[user_id].get("portfolio", {}).get("holdings", [])
        if holdings and new_budget is not None and float(new_budget) != float(users[user_id]["budget"]):
            logger.warning(
                "User %s budget changed %s -> %s while holding %d position(s); budget is the "
                "recommendation target, portfolio holdings/cash remain the source of truth for current value.",
                user_id,
                users[user_id]["budget"],
                new_budget,
                len(holdings),
            )
        users[user_id].update(
            {
                "name": updates.get("name", users[user_id].get("name", "")),
                "risk_profile": updates.get("risk_profile", users[user_id]["risk_profile"]),
                "investment_horizon": updates.get("investment_horizon", users[user_id].get("investment_horizon")),
                "goal": updates.get("goal", users[user_id].get("goal")),
                "budget": float(updates.get("budget", users[user_id]["budget"])),
                "monthly_contribution": float(
                    updates.get("monthly_contribution", users[user_id].get("monthly_contribution", 0))
                ),
                "market": updates.get("market", users[user_id].get("market", "United States")),
                "index": updates.get("index", users[user_id].get("index", "S&P 500")),
                "updated_at": datetime.now().isoformat(),
            }
        )
        self._save(users)
        return users[user_id]

    @_locked
    def update_portfolio(self, user_id: str, holdings: list, cash: float):
        """Replace the portfolio holdings and update the cash balance."""
        users = self._load()
        if user_id not in users:
            return
        port = users[user_id].setdefault("portfolio", {})
        port["holdings"] = holdings
        port["cash"] = cash
        port["updated_at"] = datetime.now().isoformat()
        users[user_id]["updated_at"] = datetime.now().isoformat()
        self._save(users)

    @_locked
    def execute_portfolio_action(self, user_id, executor, *, action):
        """Apply a validated local transaction and its audit entry under one lock."""
        users = self._load()
        current = users.get(user_id)
        if current is None:
            return None
        result = executor(current)
        if result.get("error") or result.get("respects_profile") is False:
            return result
        now = datetime.now().isoformat()
        current["portfolio"].update(holdings=result["holdings"], cash=result["cash"], updated_at=now)
        current["updated_at"] = now
        current.setdefault("transaction_history", []).append(
            {
                "action": action,
                "ticker": result.get("ticker"),
                "fees": result.get("fees", 0),
                "cash_after": result["cash"],
                "created_at": now,
            }
        )
        current["pending_recommendations"] = []
        self._save(users)
        return result

    @_locked
    def add_recommendations(self, user_id: str, recs: list):
        """Replace pending recommendations with unique, monotonically increasing identifiers."""
        users = self._load()
        if user_id not in users:
            return
        seq = int(users[user_id].get("next_rec_id", 1))
        for r in recs:
            r["id"] = seq
            r["created_at"] = datetime.now().isoformat()
            r["status"] = "pending"
            seq += 1
        users[user_id]["pending_recommendations"] = recs
        users[user_id]["next_rec_id"] = seq
        self._save(users)

    @_locked
    def get_pending_recommendation(self, user_id: str, rec_id: int) -> dict | None:
        """Read a pending recommendation without changing its status."""
        pending = self._load().get(user_id, {}).get("pending_recommendations", [])
        return next((r for r in pending if r["id"] == rec_id), None)

    @_locked
    def approve_recommendation(self, user_id: str, rec_id: int) -> dict | None:
        """Record approval after successful execution of the recommendation."""
        users = self._load()
        pending = users.get(user_id, {}).get("pending_recommendations", [])
        target = next((r for r in pending if r["id"] == rec_id), None)
        if not target:
            return None
        target["status"] = "approved"
        target["approved_at"] = datetime.now().isoformat()
        users[user_id]["pending_recommendations"] = [r for r in pending if r["id"] != rec_id]
        users[user_id].setdefault("recommendation_history", []).append(target)
        self._save(users)
        return target

    @_locked
    def reject_recommendation(self, user_id: str, rec_id: int) -> dict | None:
        """Remove a recommendation from the pending list without adding to history."""
        users = self._load()
        pending = users.get(user_id, {}).get("pending_recommendations", [])
        target = next((r for r in pending if r["id"] == rec_id), None)
        if not target:
            return None
        target["status"] = "rejected"
        target["rejected_at"] = datetime.now().isoformat()
        users[user_id]["pending_recommendations"] = [r for r in pending if r["id"] != rec_id]
        self._save(users)
        return target

    @_locked
    def execute_recommendation(self, user_id, rec_id, executor):
        """Apply balances and approval status in one atomic transaction."""
        users = self._load()
        current = users.get(user_id)
        if current is None:
            return None
        pending = current.get("pending_recommendations", [])
        rec = next((item for item in pending if item["id"] == rec_id), None)
        if rec is None:
            return None
        result = executor(rec, current)
        if result.get("error"):
            return result
        # Reject resulting portfolios that breach position or cash limits before saving.
        if result.get("respects_profile") is False:
            return {
                "error": "profile_violation",
                "message": (
                    "Executing this recommendation would breach the risk-profile "
                    "limits (position cap or cash floor); it was not approved."
                ),
                "respects_profile": False,
            }
        current["portfolio"].update(
            holdings=result["holdings"], cash=result["cash"], updated_at=datetime.now().isoformat()
        )
        current["pending_recommendations"] = [item for item in pending if item["id"] != rec_id]
        current.setdefault("recommendation_history", []).append(
            {**rec, "status": "approved", "approved_at": datetime.now().isoformat()}
        )
        self._save(users)
        return result
