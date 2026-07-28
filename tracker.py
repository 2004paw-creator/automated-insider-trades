"""
tracker.py
A live paper book of qualifying insider signals, and its performance.

Rules (chosen to avoid look-ahead):
- Entries: filter-passing signals, one per ticker per calendar year, added as
  they arrive (highest score first within a run) until the year's `cap` slots
  are full. Nothing is ever selected with hindsight.
- Entry price: the first daily close STRICTLY AFTER the filing date. Form 4s
  are often filed after hours, so the next close is the first realistic fill.
  Until that close exists, an entry is "pending".
- Performance: equal-weight mean of per-entry returns (entry -> latest close).
  Benchmark: SPY over each entry's matching window, averaged the same way.
- Year end: the book freezes; its final numbers move to the history table.

State lives in three JSON files under signals/ and is committed by the daily
GitHub Action, so the repo itself is the database: versioned, free, no infra.

  signals/latest.json       this week's filter-passing signals
  signals/portfolio.json    entries per year (the book)
  signals/performance.json  current + daily history + frozen final per year
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from config import get_logger

log = get_logger("insider.tracker")

BENCH = "SPY"


# --------------------------------------------------------------------------
# State I/O
# --------------------------------------------------------------------------
def _load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        log.warning("could not read %s (%s); starting fresh", path, e)
        return default


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def fetch_closes(tickers: list[str], start: str) -> pd.DataFrame:
    """Adjusted daily closes for `tickers` from `start`. Empty df on failure."""
    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return pd.DataFrame()
    try:
        px = yf.download(tickers, start=start, progress=False,
                         auto_adjust=True, threads=True)
    except Exception as e:
        log.warning("price download failed: %s: %s", type(e).__name__, e)
        return pd.DataFrame()
    if px is None or px.empty:
        return pd.DataFrame()
    if isinstance(px.columns, pd.MultiIndex):
        close = px["Close"].copy()
        close.columns = [c if isinstance(c, str) else c[0] for c in close.columns]
    else:  # single ticker -> flat columns
        close = px[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.loc[:, ~close.columns.duplicated()].astype(float)
    return close.dropna(axis=1, how="all")


def _series(close: pd.DataFrame, sym: str) -> pd.Series | None:
    if close.empty or sym not in close.columns:
        return None
    s = close[sym].dropna()
    return s if not s.empty else None


def _first_close_after(close: pd.DataFrame, sym: str, d: str):
    """(price, iso_date) of the first close strictly after date `d`, or None."""
    s = _series(close, sym)
    if s is None:
        return None, None
    i = s.index.searchsorted(pd.Timestamp(d) + pd.Timedelta(days=1))
    if i >= len(s):
        return None, None
    return float(s.iloc[i]), s.index[i].date().isoformat()


def _close_on(close: pd.DataFrame, sym: str, d: str):
    """Close on date `d` (or the latest close at/before it), or None."""
    s = _series(close, sym)
    if s is None:
        return None
    i = s.index.searchsorted(pd.Timestamp(d), side="right") - 1
    if i < 0:
        return None
    return float(s.iloc[i])


def _last_close(close: pd.DataFrame, sym: str):
    s = _series(close, sym)
    return float(s.iloc[-1]) if s is not None else None


# --------------------------------------------------------------------------
# Book updates
# --------------------------------------------------------------------------
def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _iso(x):
    ts = pd.to_datetime(x, errors="coerce")
    return None if pd.isna(ts) else ts.date().isoformat()


def _add_entries(portfolio: dict, matches: pd.DataFrame, year: str, cap: int) -> int:
    """Add new qualifying tickers to this year's book, best score first."""
    book = portfolio.setdefault(year, {"cap": cap, "closed": False, "entries": []})
    book["cap"] = cap                      # cap follows config, every run
    if book["closed"]:
        return 0
    have = {e["ticker"] for e in book["entries"]}
    added = 0
    if matches is None or matches.empty:
        return 0
    rows = matches.sort_values("Score", ascending=False)
    for _, r in rows.iterrows():
        if len(book["entries"]) >= cap:
            break
        t = str(r.get("Yahoo", "")).strip()
        filed = _iso(r.get("Filed"))
        if not t or t in have or not filed or filed[:4] != year:
            continue
        book["entries"].append({
            "ticker": t,
            "filed": filed,
            "score": _num(r.get("Score")),
            "insider": r.get("Insider"),
            "title": r.get("Title"),
            "signal_value": _num(r.get("Value")),
            "status": "pending",
            "entry_price": None,
            "entry_date": None,
            "spy_entry": None,
        })
        have.add(t)
        added += 1
    if added:
        log.info("book %s: added %d entries (%d/%d slots)",
                 year, added, len(book["entries"]), book["cap"])
    return added


def _fill_pending(portfolio: dict, close: pd.DataFrame) -> None:
    for year, book in portfolio.items():
        if book.get("closed"):
            continue
        for e in book["entries"]:
            if e["status"] != "pending":
                continue
            px, d = _first_close_after(close, e["ticker"], e["filed"])
            if px is None:
                continue
            spy = _close_on(close, BENCH, d)
            e.update(status="active", entry_price=round(px, 4),
                     entry_date=d, spy_entry=round(spy, 4) if spy else None)
            log.info("filled %s @ %.2f on %s", e["ticker"], px, d)


def _year_performance(book: dict, close: pd.DataFrame, as_of: str) -> dict:
    rets, srets = [], []
    n_pending = 0
    for e in book["entries"]:
        if e["status"] != "active" or not e.get("entry_price"):
            n_pending += 1
            continue
        last = _last_close(close, e["ticker"])
        if last is None:            # no data (halted/delisted): hold last known
            continue
        rets.append(last / e["entry_price"] - 1.0)
        spy_last = _last_close(close, BENCH)
        if e.get("spy_entry") and spy_last:
            srets.append(spy_last / e["spy_entry"] - 1.0)
    return {
        "as_of": as_of,
        "n_active": len(rets),
        "n_pending": n_pending,
        "book_return": round(float(np.mean(rets)), 6) if rets else None,
        "spy_return": round(float(np.mean(srets)), 6) if srets else None,
    }


def _upsert_history(perf_year: dict, point: dict) -> None:
    hist = [h for h in perf_year.get("history", []) if h["date"] != point["date"]]
    hist.append(point)
    perf_year["history"] = sorted(hist, key=lambda h: h["date"])


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def run_update(matches: pd.DataFrame, out_dir: str = "signals",
               cap: int = 50, today: date | None = None) -> None:
    """Update latest.json, the book, and performance. Never raises."""
    today = today or date.today()
    year = str(today.year)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    portfolio = _load_json(f"{out_dir}/portfolio.json", {})
    performance = _load_json(f"{out_dir}/performance.json", {})

    # 1. latest.json — this week's filter-passing signals, for the site.
    signals = []
    if matches is not None and not matches.empty:
        for _, r in matches.sort_values("Score", ascending=False).iterrows():
            signals.append({
                "ticker": str(r.get("Yahoo", "")),
                "company": r.get("Company"),
                "insider": r.get("Insider"),
                "title": r.get("Title"),
                "filed": _iso(r.get("Filed")),
                "value": _num(r.get("Value")),
                "price": _num(r.get("px_now")),
                "score": _num(r.get("Score")),
            })
    _save_json(f"{out_dir}/latest.json",
               {"generated_at": now_iso, "count": len(signals), "signals": signals})

    # 2. Add this run's qualifying signals to the current year's book.
    _add_entries(portfolio, matches, year, cap)

    # 3. Prices for every entry in every not-yet-closed year, plus SPY.
    open_entries = [e for y, b in portfolio.items() if not b.get("closed")
                    for e in b["entries"]]
    if open_entries:
        tickers = sorted({e["ticker"] for e in open_entries}) + [BENCH]
        start = (pd.Timestamp(min(e["filed"] for e in open_entries))
                 - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        close = fetch_closes(tickers, start)
    else:
        close = pd.DataFrame()

    # 4. Fill pending entries, then mark today's performance per open year.
    if not close.empty:
        _fill_pending(portfolio, close)
        for y, book in portfolio.items():
            if book.get("closed"):
                continue
            perf = performance.setdefault(
                y, {"closed": False, "current": None, "history": [], "final": None})
            cur = _year_performance(book, close, today.isoformat())
            perf["current"] = cur
            if cur["book_return"] is not None:
                _upsert_history(perf, {
                    "date": today.isoformat(), "book": cur["book_return"],
                    "spy": cur["spy_return"], "n": cur["n_active"],
                })
    else:
        log.warning("no price data this run; book/performance unchanged")

    # 5. Freeze any year that has ended.
    for y in list(portfolio.keys()):
        if int(y) < today.year and not portfolio[y].get("closed"):
            portfolio[y]["closed"] = True
            py = performance.setdefault(
                y, {"closed": False, "current": None, "history": [], "final": None})
            py["closed"] = True
            py["final"] = py.get("current")
            log.info("year %s closed; final book return: %s",
                     y, (py["final"] or {}).get("book_return"))

    _save_json(f"{out_dir}/portfolio.json", portfolio)
    _save_json(f"{out_dir}/performance.json", performance)
    log.info("signals/ updated (%d live signals, year %s book %d entries)",
             len(signals), year,
             len(portfolio.get(year, {}).get("entries", [])))


def safe_run_update(matches: pd.DataFrame, **kw) -> None:
    """Wrapper so a tracker problem can never block the email digest."""
    try:
        run_update(matches, **kw)
    except Exception as e:
        log.error("tracker failed: %s: %s", type(e).__name__, e, exc_info=True)
