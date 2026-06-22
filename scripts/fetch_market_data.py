import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.universe.universe_builder import UniverseBuilder
from backend.data.downloader import MarketDataDownloader
from backend.config.markets import MARKETS

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')


def pick(prompt, options):
    print(f'\n{prompt}')
    for i, o in enumerate(options, 1):
        print(f'  {i}. {o}')
    while True:
        r = input('  Select: ').strip()
        if r.isdigit() and 1 <= int(r) <= len(options):
            return options[int(r) - 1]
        print('  Enter a number from the list')


def main():
    builder = UniverseBuilder()
    dl      = MarketDataDownloader()

    mode = pick('Download mode?', [
        'Single market and index',
        'All markets — all indices',
    ])

    if 'All markets' in mode:
        all_tickers = []
        for market, cfg in MARKETS.items():
            for index in cfg['indices']:
                print(f'\nFetching {market} / {index}...')
                try:
                    tickers = builder.get_universe(market, index)
                    print(f'  {len(tickers)} tickers')
                    all_tickers.extend(tickers)
                except Exception as e:
                    print(f'  Skipped ({e})')

        # deduplicate — same ticker can appear in multiple indices
        all_tickers = list(dict.fromkeys(all_tickers))
        print(f'\nTotal unique tickers: {len(all_tickers)}')
        print(f'Estimated time: ~{len(all_tickers)//5} minutes')
        if input('Continue? [y/n]: ').strip().lower() != 'y':
            print('Cancelled.')
            return
        target = all_tickers

    else:
        markets = builder.list_markets()
        market  = pick('Which market?', list(markets.keys()))
        index   = pick(f'Which index ({market})?', markets[market])

        print(f'\nFetching {index} constituents...')
        tickers = builder.get_universe(market, index)
        print(f'  {len(tickers)} stocks — first 5: {tickers[:5]}')

        scope  = pick('Download scope?', [
            f'All {len(tickers)} stocks',
            'First 10 only (quick test)',
        ])
        target = tickers if 'All' in scope else tickers[:10]

    print(f'\nDownloading price history (2018 to today) for {len(target)} stocks...')
    results = dl.download_universe(target)

    print(f'\nDownload complete:')
    print(f'  Downloaded : {len(results["success"])}')
    print(f'  Skipped    : {len(results["skipped"])} (already cached)')
    print(f'  Failed     : {len(results["failed"])}')
    if results['failed']:
        print(f'  Failed     : {results["failed"][:10]}')
    print(f'\nSaved to data/raw/')
    print('Next: python scripts/build_features.py')


if __name__ == '__main__':
    main()
