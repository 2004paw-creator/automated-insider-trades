"""
config.py
Central configuration, logging, and secrets for the insider-trades notifier.

Everything tunable lives here so the fetch / scoring / email modules stay
free of magic numbers, and the scoring weights can be changed (or swept in a
test) without touching any logic.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = "insider") -> logging.Logger:
    """A stdout logger. Set LOG_LEVEL=DEBUG for verbose runs."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    return logger


# --------------------------------------------------------------------------
# Secrets (from environment / GitHub Actions secrets)
# --------------------------------------------------------------------------
def _clean_env(x: str | None) -> str:
    return (x or "").strip().replace("\r", "").replace("\n", "")


GMAIL_USER = _clean_env(os.getenv("GMAIL_USER"))
GMAIL_APP_PASS = _clean_env(os.getenv("GMAIL_APP_PASSWORD"))
TO_EMAIL = _clean_env(os.getenv("TO_EMAIL"))
UA_EMAIL = GMAIL_USER or "you@example.com"


# --------------------------------------------------------------------------
# SEC / HTTP
# --------------------------------------------------------------------------
# SEC fair-access policy caps clients at 10 requests/second and requires a
# descriptive User-Agent. We stay comfortably under the ceiling.
# https://www.sec.gov/os/accessing-edgar-data
SEC_RATE_PER_SEC = float(os.getenv("SEC_RATE_PER_SEC", "7"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))
REQ_TIMEOUT_S = float(os.getenv("REQ_TIMEOUT_S", "20"))
RETRY_TOTAL = int(os.getenv("RETRY_TOTAL", "3"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "0.4"))

SEC_HEADERS = {
    "User-Agent": f"Mozilla/5.0 (compatible; InsiderBot/1.0; +mailto:{UA_EMAIL})",
    "From": UA_EMAIL,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
    "Referer": "https://www.sec.gov/edgar/searchedgar/companysearch",
}
CURRENT_FORM4_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&CIK=&type=4&owner=only&count=200"
)

# The canonical column set for an empty result, kept in one place.
EMPTY_COLS = [
    "Yahoo", "Date", "Filed", "Company", "Insider", "Title",
    "Price", "Shares", "Value", "Transaction",
]


# --------------------------------------------------------------------------
# Scoring configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringConfig:
    """Filters and factor weights for ranking insider purchases.

    Weights are applied to rank-normalised factors (each in [0, 1], ranked
    within a filing date) and should sum to roughly 1.
    """

    # --- Filters -----------------------------------------------------------
    require_ceo_cfo: bool = True     # keep only CEO/CFO purchases
    ceo_only: bool = False           # if True, CEO only (ignored unless above)
    min_value: float = 100_000       # minimum USD value of the purchase
    min_price: float = 3.0           # drop sub-$3 names (penny-stock noise)
    min_dvol: float = 200_000        # minimum 20-day average dollar volume
    max_email_rows: int = 50
    lookback_week: int = 7           # rolling window for the "this week" view
    book_cap: int = 50               # max entries per calendar-year paper book

    # --- Factor weights ----------------------------------------------------
    w_insider_size: float = 0.35     # log dollar-value of the purchase
    w_cluster: float = 0.15          # number of purchases in a 10-day window
    w_mom_trend: float = 0.30        # 12-1 momentum (trend)
    w_mom_contra: float = 0.20       # short-term (3m) reversal


CFG = ScoringConfig()

SUBJECT_PREFIX = "Insider buys (SEC)"
YF_LINK_FMT = "https://finance.yahoo.com/quote/{sym}"
