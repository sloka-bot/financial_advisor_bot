"""
extract_fnspid_news.py  (Stage 1, resumable)

Streams the 22GB FNSPID CSV, keeps only S&P-500 rows, writes compact
per-ticker raw news (date,title,summary). Resumable: checkpoints rows
processed to data/fnspid/state.json and self-stops before the shell's
time cap, so it survives VM reboots and can be re-run to continue.

Fresh run (no state.json)  -> truncates each ticker file on first write.
Resume  (state.json exists)-> skips already-processed rows, appends.
Duplicates possible only for a chunk interrupted mid-write; de-duped in Stage 2.
"""

import json
import os
import time
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.expanduser("~/mnt/Desktop/fnspid_news.csv")
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
TIME_BUDGET = 150  # seconds of real work before graceful stop

# --- load / init checkpoint ---
if os.path.exists(STATE):
    st = json.load(open(STATE))
    fresh = False
else:
    st = {"rows_processed": 0, "matched": 0, "done": False, "started_at": time.time()}
    fresh = True
skip = st["rows_processed"]
_cpath = os.path.join(D, "counts.json")
counts = Counter(json.load(open(_cpath))) if os.path.exists(_cpath) else Counter()

header_seen = set()  # tickers written this run (controls w vs a on fresh run)
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
                out.to_csv(path, mode="w", header=True, index=False)  # truncate stale partial
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
    if time.time() - t0 > TIME_BUDGET:
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
