"""Stream the FNSPID CSV into per-ticker S&P 500 news files with resumable checkpoints."""

import argparse
import json
import os
import time
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "fnspid")
OUT = os.path.join(D, "raw")
STATE = os.path.join(D, "state.json")
PROG = os.path.join(D, "extract_progress.json")
os.makedirs(OUT, exist_ok=True)

uni = json.load(open(os.path.join(ROOT, "data", "universe", "United_States__S_P_500.json")))
canon = {}
for t in (str(x).strip() for x in uni["tickers"]):
    canon[t] = t
    canon[t.replace(".", "-")] = t
    canon[t.replace("-", ".")] = t
match_set = set(canon)

COLS = ["Date", "Stock_symbol", "Article_title", "Textrank_summary"]
CHUNK = 250_000

parser = argparse.ArgumentParser()
parser.add_argument(
    "csv", nargs="?", default=os.path.join(D, "nasdaq_exteral_data.csv"), help="path to the FNSPID news CSV"
)
parser.add_argument("--time-budget", type=int, default=0, help="seconds before a checkpointed stop; 0 runs to the end")
args = parser.parse_args()
F = os.path.expanduser(args.csv)
TIME_BUDGET = args.time_budget

# Load or initialise the checkpoint.
if os.path.exists(STATE):
    st = json.load(open(STATE))
    fresh = False
else:
    st = {"rows_processed": 0, "matched": 0, "done": False, "started_at": time.time()}
    fresh = True
skip = st["rows_processed"]
_cpath = os.path.join(D, "counts.json")
counts = Counter(json.load(open(_cpath))) if os.path.exists(_cpath) else Counter()

header_seen = set()  # tickers written this run
t0 = time.time()
rows_this = 0
matched_this = 0
stopped_early = False

reader = pd.read_csv(
    F,
    usecols=COLS,
    dtype=str,
    engine="c",
    on_bad_lines="skip",
    chunksize=CHUNK,
    skiprows=range(1, skip + 1) if skip else None,
)
for chunk in reader:
    n = len(chunk)
    m = chunk[chunk["Stock_symbol"].isin(match_set)]
    if len(m):
        m = m.copy()
        m["sym"] = m["Stock_symbol"].map(canon)
        for sym, g in m.groupby("sym"):
            out = g[["Date", "Article_title", "Textrank_summary"]].rename(
                columns={"Date": "date", "Article_title": "title", "Textrank_summary": "summary"}
            )
            path = os.path.join(OUT, f"{sym.replace('/', '_')}.csv")
            if fresh and sym not in header_seen:
                out.to_csv(path, mode="w", header=True, index=False)  # truncate partial file
            else:
                write_header = not os.path.exists(path)
                out.to_csv(path, mode="a", header=write_header, index=False)
            header_seen.add(sym)
            counts[sym] += len(g)
        matched_this += len(m)
    rows_this += n
    st["rows_processed"] = skip + rows_this
    st["matched"] = st.get("matched", 0) + len(m)
    json.dump(st, open(STATE, "w"))
    json.dump(
        {
            "done": False,
            "rows_processed": st["rows_processed"],
            "matched_total": st["matched"],
            "tickers": len(counts),
            "elapsed_this_call": round(time.time() - t0, 1),
        },
        open(PROG, "w"),
    )
    if TIME_BUDGET and time.time() - t0 > TIME_BUDGET:
        stopped_early = True
        break

if not stopped_early:
    st["done"] = True
json.dump(st, open(STATE, "w"))
json.dump(dict(counts.most_common()), open(os.path.join(D, "counts.json"), "w"), indent=2)
json.dump(
    {
        "done": st["done"],
        "rows_processed": st["rows_processed"],
        "matched_total": st["matched"],
        "tickers": len(counts),
        "elapsed_this_call": round(time.time() - t0, 1),
    },
    open(PROG, "w"),
)
print(
    f"{'DONE' if st['done'] else 'PAUSED'} rows_processed={st['rows_processed']} "
    f"matched_total={st['matched']} tickers={len(counts)} this_call={time.time() - t0:.0f}s"
)
