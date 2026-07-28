"""
scoring.py
Turn raw insider purchases into a ranked table.

The model is a small cross-sectional factor score. For each purchase we
compute four factors, rank-normalise each within its filing date (so scores
are relative to same-day peers, not absolute), and take a weighted sum:

  insider_size  log dollar value of the purchase   (conviction / skin-in-game)
  cluster       purchases in a trailing 10 days     (corroboration)
  mom_trend     12-1 month momentum                 (buying into strength)
  mom_contra    negative 3-month return             (buying a dip)

The trend/contra pair deliberately pulls in opposite directions: the score
rewards names that are either in a longer up-trend or recently sold off,
which is where insider purchases have historically been most informative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from config import CFG, get_logger

log = get_logger("insider.scoring")

_FEATURE_COLS = [
    "Yahoo", "Filed", "Date", "Company", "Insider", "Title", "Price", "Shares",
    "Value", "cluster_10d", "mom_12_1", "mom_3m", "px_now", "dvol20", "Score",
]


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------
def build_prices(ins_df: pd.DataFrame):
    """Download adjusted close and volume for the tickers in `ins_df`.

    Uses auto_adjust=True so 'Close' is already split/dividend-adjusted --
    correct for return-based factors and robust to yfinance dropping the
    'Adj Close' column, which it has done in past releases.
    """
    if ins_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    tickers = sorted(ins_df["Yahoo"].dropna().unique().tolist())
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    start = (pd.to_datetime(ins_df["Date"]).min() - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(ins_df["Date"]).max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    try:
        px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                         progress=False, threads=True)
    except Exception as e:
        log.warning("yfinance download failed: %s: %s", type(e).__name__, e)
        return pd.DataFrame(), pd.DataFrame()

    if px is None or px.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(px.columns, pd.MultiIndex):
        close_all = px["Close"].copy()
        vol_all = px["Volume"].copy()
        close_all.columns = [c if isinstance(c, str) else c[0] for c in close_all.columns]
        vol_all.columns = [c if isinstance(c, str) else c[0] for c in vol_all.columns]
    else:  # single ticker -> flat columns
        sym = tickers[0]
        close_all = px[["Close"]].rename(columns=lambda _: sym)
        vol_all = px[["Volume"]].rename(columns=lambda _: sym)

    close_all = close_all.loc[:, ~close_all.columns.duplicated()].astype(float)
    vol_all = vol_all.loc[:, ~vol_all.columns.duplicated()].astype(float)
    bad = [c for c in close_all.columns if close_all[c].isna().all()]
    if bad:
        log.debug("no price data for: %s", ", ".join(bad))
        close_all = close_all.drop(columns=bad, errors="ignore")
        vol_all = vol_all.drop(columns=bad, errors="ignore")
    return close_all, vol_all


def price_at(close_all, y, dt, off=0):
    try:
        idx = close_all.index
        j = idx.searchsorted(pd.Timestamp(dt)) + off
        if j < 0 or j >= len(idx):
            return np.nan
        return float(close_all[y].iloc[j])
    except Exception:
        return np.nan


def mom_from(close_all, y, dt, lb):
    p0, pL = price_at(close_all, y, dt, 0), price_at(close_all, y, dt, -lb)
    if not np.isfinite(p0) or not np.isfinite(pL) or pL == 0:
        return np.nan
    return p0 / pL - 1.0


def mom_12m_minus_1m(close_all, y, dt):
    p_1m, p_12m = price_at(close_all, y, dt, -21), price_at(close_all, y, dt, -252)
    if not np.isfinite(p_1m) or not np.isfinite(p_12m) or p_12m == 0:
        return np.nan
    return p_1m / p_12m - 1.0


def dvol20(close_all, vol_all, y, dt):
    try:
        idx = close_all.index
        i = idx.searchsorted(pd.Timestamp(dt))
        lo, hi = max(0, i - 19), i + 1
        px = close_all[y].iloc[lo:hi].astype(float)
        vo = vol_all[y].iloc[lo:hi].astype(float)
        if px.empty or vo.empty:
            return np.nan
        return float((px * vo).mean())
    except Exception:
        return np.nan


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def add_features_and_score(raw_df: pd.DataFrame, cfg=CFG):
    """Return (matches, all_ranked): filtered top rows, and the full ranking."""
    if raw_df is None or raw_df.empty:
        empty = pd.DataFrame(columns=_FEATURE_COLS)
        return empty, empty.copy()

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Filed"] = (
        pd.to_datetime(df["Filed"], errors="coerce").dt.normalize()
        if "Filed" in df.columns
        else pd.to_datetime(df["Date"]).dt.normalize()
    )
    df = df.dropna(subset=["Date", "Yahoo"]).reset_index(drop=True)

    # Clustering: purchases within a trailing 10 calendar days, per ticker.
    df = df.sort_values(["Yahoo", "Date"])
    cl = (
        df.assign(_one=1).set_index("Date").groupby("Yahoo")["_one"]
        .rolling("10D").sum().reset_index(level=0, drop=True).astype(int)
    )
    df["cluster_10d"] = cl.values

    close_all, vol_all = build_prices(df)
    if close_all.empty or vol_all.empty:
        for c in ("mom_3m", "mom_12_1", "dvol20", "px_now"):
            df[c] = np.nan
    else:
        df["mom_3m"] = [mom_from(close_all, y, d, 63) for y, d in zip(df["Yahoo"], df["Date"])]
        df["mom_12_1"] = [mom_12m_minus_1m(close_all, y, d) for y, d in zip(df["Yahoo"], df["Date"])]
        df["dvol20"] = [dvol20(close_all, vol_all, y, d) for y, d in zip(df["Yahoo"], df["Date"])]
        df["px_now"] = [price_at(close_all, y, d, 0) for y, d in zip(df["Yahoo"], df["Date"])]

    # Role flags. Guard the Title column explicitly so a missing column
    # degrades to "no CEO/CFO" instead of raising.
    title_u = (
        df["Title"].astype(str).str.upper()
        if "Title" in df.columns
        else pd.Series("", index=df.index)
    )
    df["is_CEO"] = title_u.str.contains("CEO") | title_u.str.contains("CHIEF EXECUTIVE")
    df["is_CFO"] = title_u.str.contains("CFO") | title_u.str.contains("CHIEF FINANCIAL")
    df["Value"] = pd.to_numeric(df.get("Value"), errors="coerce").fillna(0.0)

    # Raw factors.
    df["f_insider_size"] = np.log1p(df["Value"]).replace([np.inf, -np.inf], np.nan)
    df["f_cluster"] = pd.to_numeric(df["cluster_10d"], errors="coerce")
    df["f_mom_trend"] = pd.to_numeric(df["mom_12_1"], errors="coerce")
    df["f_mom_contra"] = -pd.to_numeric(df["mom_3m"], errors="coerce")

    def _rank01(s: pd.Series) -> pd.Series:
        r = s.rank(pct=True, method="average")
        return (r if not r.isna().all() else pd.Series(0.5, index=s.index)).fillna(0.5)

    for col in ("f_insider_size", "f_cluster", "f_mom_trend", "f_mom_contra"):
        df[col + "_n"] = df.groupby("Filed")[col].transform(_rank01)

    df["Score"] = (
        cfg.w_insider_size * df["f_insider_size_n"]
        + cfg.w_cluster * df["f_cluster_n"]
        + cfg.w_mom_trend * df["f_mom_trend_n"]
        + cfg.w_mom_contra * df["f_mom_contra_n"]
    )

    all_ranked = (
        df[[c for c in _FEATURE_COLS if c in df.columns]]
        .sort_values(["Score", "Value"], ascending=[False, False])
    )

    price_ok = df["px_now"] >= cfg.min_price if "px_now" in df else pd.Series(False, index=df.index)
    vol_ok = df["dvol20"] >= cfg.min_dvol if "dvol20" in df else pd.Series(False, index=df.index)
    if cfg.require_ceo_cfo:
        role_ok = df["is_CEO"] if cfg.ceo_only else (df["is_CEO"] | df["is_CFO"])
    else:
        role_ok = pd.Series(True, index=df.index)
    value_ok = df["Value"] >= cfg.min_value if "Value" in df else pd.Series(True, index=df.index)

    mask = (
        price_ok.fillna(False) & vol_ok.fillna(False)
        & role_ok.fillna(False) & value_ok.fillna(False)
    )
    matches = (
        df.loc[mask, [c for c in _FEATURE_COLS if c in df.columns]]
        if mask.any() else all_ranked.iloc[0:0]
    )

    log.info("scored %d rows -> %d pass filters", len(df), int(mask.sum()))
    return matches.head(cfg.max_email_rows), all_ranked.head(cfg.max_email_rows)
