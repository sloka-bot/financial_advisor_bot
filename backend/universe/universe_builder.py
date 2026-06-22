import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from backend.config.markets import MARKETS

logger = logging.getLogger(__name__)

TICKER_COLS = ['Symbol', 'Ticker', 'Code', 'Epic', 'NSE Symbol', 'Company']

SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/json,*/*',
})


class UniverseBuilder:

    def __init__(self, data_dir='data/universe'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def list_markets(self):
        return {m: list(cfg['indices'].keys()) for m, cfg in MARKETS.items()}

    def get_universe(self, market, index, force_refresh=False):
        self._validate(market, index)
        cache = self._cache_path(market, index)

        if cache.exists() and not force_refresh:
            logger.info(f'Loading cached universe: {cache.name}')
            return json.loads(cache.read_text())['tickers']

        logger.info(f'Fetching {market} / {index}')
        tickers = self._fetch(market, index)

        if not tickers:
            raise RuntimeError(f'No tickers retrieved for {market} / {index}')

        self._save(cache, market, index, tickers)
        return tickers

    def get_info(self, market, index):
        cache = self._cache_path(market, index)
        if not cache.exists():
            return {'cached': False}
        data = json.loads(cache.read_text())
        return {'cached': True, 'count': data['count'], 'fetched_at': data['fetched_at']}

    # ── routing ────────────────────────────────────────────────────────────

    def _fetch(self, market, index):
        cfg = MARKETS[market]['indices'][index]
        src = cfg['source_type']

        if src == 'static':
            return cfg['static_tickers']

        # dedicated API sources first — faster and more reliable than scraping
        if market == 'United States' and index == 'NASDAQ 100':
            t = self._nasdaq_api()
            if t: return t

        if market == 'United States' and index == 'S&P 500':
            t = self._sp500_api()
            if t: return t

        if market == 'India':
            t = self._nse_api(index)
            if t: return t

        # Wikipedia via requests as fallback for everything
        return self._from_wikipedia(cfg)

    # ── NASDAQ 100 official API ─────────────────────────────────────────────

    def _nasdaq_api(self):
        try:
            r = SESSION.get('https://api.nasdaq.com/api/quote/list-type/nasdaq100', timeout=15)
            r.raise_for_status()
            rows = r.json()['data']['data']['rows']
            return [row['symbol'].strip() for row in rows
                    if re.match(r'^[A-Z]{1,5}$', row.get('symbol', '').strip())]
        except Exception as e:
            logger.warning(f'NASDAQ API failed: {e}')
            return []

    # ── S&P 500 via datahub.io ─────────────────────────────────────────────

    def _sp500_api(self):
        try:
            url = ('https://pkgstore.datahub.io/core/s-and-p-500-companies/'
                   'constituents_json/data/87616065395e0e7a96ef65c31e7f3c16/constituents_json.json')
            r = SESSION.get(url, timeout=15)
            r.raise_for_status()
            return [row['Symbol'].strip().replace('.', '-') for row in r.json()
                    if re.match(r'^[A-Z0-9.\-]{1,6}$', row.get('Symbol', '').strip())]
        except Exception as e:
            logger.warning(f'S&P 500 API failed: {e}')
            return []

    # ── NSE India API ──────────────────────────────────────────────────────

    def _nse_api(self, index):
        # NSE requires a session cookie — get one first then hit the data endpoint
        index_map = {
            'NIFTY 50':  'NIFTY%2050',
            'NIFTY 100': 'NIFTY%20100',
            'SENSEX':    'SENSEX',
        }
        nse_index = index_map.get(index)
        if not nse_index:
            return []

        try:
            # establish a session cookie first
            SESSION.get('https://www.nseindia.com', timeout=10)
            r = SESSION.get(
                f'https://www.nseindia.com/api/equity-stockIndices?index={nse_index}',
                timeout=15,
            )
            r.raise_for_status()
            data  = r.json()
            rows  = data.get('data', [])
            suffix = '.NS' if index != 'SENSEX' else '.BO'
            return [row['symbol'].strip() + suffix for row in rows
                    if row.get('symbol') and row['symbol'] != 'NIFTY 50']
        except Exception as e:
            logger.warning(f'NSE API failed for {index}: {e} — trying Wikipedia')
            return []

    # ── Wikipedia via requests ─────────────────────────────────────────────

    def _from_wikipedia(self, cfg):
        url       = cfg['wiki_url']
        col       = cfg.get('ticker_column', 'Symbol')
        suffix    = cfg.get('ticker_suffix', '')
        table_idx = cfg.get('wiki_table_index')
        min_rows  = cfg.get('min_rows', 10)
        fix_dots  = cfg.get('fix_dots', False)

        logger.info(f'  Scraping {url}')
        try:
            resp = SESSION.get(url, timeout=25)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.warning(f'  HTTP error {e} for {url}')
            return []
        except Exception as e:
            logger.warning(f'  Could not fetch {url}: {e}')
            return []

        try:
            tables = pd.read_html(io.StringIO(resp.text), flavor='lxml')
        except Exception as e:
            logger.warning(f'  Could not parse tables from {url}: {e}')
            return []

        # try the known table index first, then scan all tables
        candidates = []
        if table_idx is not None:
            candidates = [table_idx]
        candidates += list(range(len(tables)))

        for i in candidates:
            if i >= len(tables):
                continue
            tickers = self._pull_col(tables[i], col, suffix, fix_dots)
            if len(tickers) >= min_rows:
                logger.info(f"  {len(tickers)} tickers from table[{i}] column '{col}'")
                return tickers

        # fallback — scan for any common ticker column
        for i, table in enumerate(tables):
            for candidate_col in TICKER_COLS:
                tickers = self._pull_col(table, candidate_col, suffix, fix_dots)
                if len(tickers) >= min_rows:
                    logger.info(f"  {len(tickers)} tickers — fallback '{candidate_col}' in table[{i}]")
                    return tickers

        logger.warning(f'  No ticker table found on {url}')
        return []

    # ── helpers ────────────────────────────────────────────────────────────

    def _pull_col(self, table, col, suffix, fix_dots):
        if isinstance(table.columns, pd.MultiIndex):
            table = table.copy()
            table.columns = [' '.join(str(c) for c in t).strip() for t in table.columns]

        match = next((c for c in table.columns if c.strip().lower() == col.lower()), None)
        if match is None:
            return []

        raw = (table[match].dropna().astype(str).str.strip()
               .pipe(lambda s: s[s != ''])
               .pipe(lambda s: s[s != 'nan'])
               .pipe(lambda s: s[~s.str.startswith('http')])
               .pipe(lambda s: s[s.str.len() <= 15])
               .pipe(lambda s: s[~s.str.contains(r'\s', regex=True)])
               .pipe(lambda s: s[s.str.match(r'^[A-Z0-9&.\-]+$')]))

        if fix_dots:
            raw = raw.str.replace(r'\.(?=[A-Z])', '-', regex=True)
        if suffix:
            raw = raw.apply(lambda t: t if t.endswith(suffix) else t + suffix)

        return raw.tolist()

    def _cache_path(self, market, index):
        safe = lambda s: re.sub(r'[^A-Za-z0-9_]', '_', s)
        return self.data_dir / f'{safe(market)}__{safe(index)}.json'

    def _save(self, path, market, index, tickers):
        path.write_text(json.dumps({
            'market': market, 'index': index,
            'count': len(tickers),
            'fetched_at': datetime.utcnow().isoformat() + 'Z',
            'tickers': tickers,
        }, indent=2))
        logger.info(f'  Saved {len(tickers)} tickers → {path.name}')

    def _validate(self, market, index):
        if market not in MARKETS:
            raise ValueError(f'Unknown market: {market!r}')
        if index not in MARKETS[market]['indices']:
            raise ValueError(f'Unknown index: {index!r} for {market}')
