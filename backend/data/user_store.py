import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STORE = Path('data/users.json')


class UserStore:
    """
    Flat-file user store keyed by Clerk user ID.
    Stores risk profile, portfolio state, and recommendation history.
    Using JSON files rather than a database keeps the prototype dependency-free
    while still persisting state across server restarts.
    """

    def _load(self) -> dict:
        if not STORE.exists():
            return {}
        try:
            return json.loads(STORE.read_text()).get('users', {})
        except Exception:
            return {}

    def _save(self, users: dict):
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps({'users': users}, indent=2))

    def get(self, user_id: str) -> dict | None:
        return self._load().get(user_id)

    def is_new(self, user_id: str) -> bool:
        return user_id not in self._load()

    def create_profile(self, user_id: str, profile: dict) -> dict:
        """Save a new user profile from the onboarding questionnaire."""
        users = self._load()
        now   = datetime.now().isoformat()
        users[user_id] = {
            'user_id':            user_id,
            'name':               profile.get('name', ''),
            'risk_profile':       profile.get('risk_profile', 'moderate'),
            'investment_horizon': profile.get('investment_horizon', '5-10 years'),
            'goal':               profile.get('goal', 'growth'),
            'budget':             float(profile.get('budget', 10000)),
            'monthly_contribution': float(profile.get('monthly_contribution', 0)),
            'created_at':         now,
            'updated_at':         now,
            'portfolio': {
                'holdings':   [],
                'cash':       float(profile.get('budget', 10000)),
                'created_at': now,
                'updated_at': now,
            },
            'pending_recommendations': [],
            'recommendation_history':  [],
        }
        self._save(users)
        logger.info(f'Profile created for {user_id}')
        return users[user_id]

    def update_profile(self, user_id: str, updates: dict) -> dict:
        users = self._load()
        if user_id not in users:
            return self.create_profile(user_id, updates)
        users[user_id].update({**updates, 'updated_at': datetime.now().isoformat()})
        self._save(users)
        return users[user_id]

    def update_portfolio(self, user_id: str, holdings: list, cash: float):
        users = self._load()
        if user_id not in users:
            return
        users[user_id]['portfolio'] = {
            'holdings':   holdings,
            'cash':       cash,
            'updated_at': datetime.now().isoformat(),
        }
        self._save(users)

    def add_recommendations(self, user_id: str, recs: list):
        """Queue new recommendations as pending approval."""
        users = self._load()
        if user_id not in users:
            return
        existing = users[user_id].get('pending_recommendations', [])
        # assign sequential IDs
        next_id  = max((r.get('id', 0) for r in existing), default=0) + 1
        for r in recs:
            r['id']         = next_id
            r['created_at'] = datetime.now().isoformat()
            r['status']     = 'pending'
            next_id        += 1
        users[user_id]['pending_recommendations'] = existing + recs
        self._save(users)

    def approve_recommendation(self, user_id: str, rec_id: int) -> dict | None:
        users   = self._load()
        pending = users.get(user_id, {}).get('pending_recommendations', [])
        target  = next((r for r in pending if r['id'] == rec_id), None)
        if not target:
            return None
        target['status']      = 'approved'
        target['decided_at']  = datetime.now().isoformat()
        # move from pending to history
        users[user_id]['pending_recommendations'] = [r for r in pending if r['id'] != rec_id]
        history = users[user_id].get('recommendation_history', [])
        history.append(target)
        users[user_id]['recommendation_history'] = history[-100:]
        self._save(users)
        return target

    def reject_recommendation(self, user_id: str, rec_id: int) -> dict | None:
        users   = self._load()
        pending = users.get(user_id, {}).get('pending_recommendations', [])
        target  = next((r for r in pending if r['id'] == rec_id), None)
        if not target:
            return None
        target['status']     = 'rejected'
        target['decided_at'] = datetime.now().isoformat()
        users[user_id]['pending_recommendations'] = [r for r in pending if r['id'] != rec_id]
        history = users[user_id].get('recommendation_history', [])
        history.append(target)
        users[user_id]['recommendation_history'] = history[-100:]
        self._save(users)
        return target
