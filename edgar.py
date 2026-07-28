"""
edgar.py
Fetch and parse SEC Form 4 insider-purchase filings.

Two sources, in priority order:
  1. EDGAR daily master index -> per-filing ownership XML (structured, reliable)
  2. EDGAR "current filings" HTML table (fallback when the master is unavailable)

Only open-market purchases (transaction code "P") are kept. All network
access is rate-limited to respect the SEC's 10 req/s fair-access policy, and
failures are logged rather than silently swallowed.
"""
from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    CURRENT_FORM4_URL, EMPTY_COLS, MAX_WORKERS, REQ_TIMEOUT_S,
    RETRY_BACKOFF, RETRY_TOTAL, SEC_HEADERS, SEC_RATE_PER_SEC, get_logger,
)

log = get_logger("insider.edgar")


# --------------------------------------------------------------------------
# Rate limiting + HTTP
# --------------------------------------------------------------------------
class RateLimiter:
    """Thread-safe token spacer: at most `rate_per_sec` calls per second."""

    def __init__(self, rate_per_sec: float):
        self._interval = 1.0 / max(rate_per_sec, 0.1)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if sleep_for:
            time.sleep(sleep_for)


_limiter = RateLimiter(SEC_RATE_PER_SEC)


def session_with_retries(total=RETRY_TOTAL, backoff=RETRY_BACKOFF) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total, connect=total, read=total, backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def _get(sess: requests.Session, url: str) -> requests.Response:
    """Rate-limited GET with SEC headers."""
    _limiter.wait()
    return sess.get(url, headers=SEC_HEADERS, timeout=REQ_TIMEOUT_S)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_COLS)


def to_num(x) -> float:
    s = (
        str(x).replace(",", "").replace("$", "").replace("%", "")
        .replace("\u2014", "").replace("\u2212", "-")
        .replace("(", "-").replace(")", "")
    )
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------------
# Daily master index -> accession XML
# --------------------------------------------------------------------------
def _qtr_of(dt: date) -> int:
    return (dt.month - 1) // 3 + 1


def _daily_master_url(dt: date) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/daily-index/"
        f"{dt.year}/QTR{_qtr_of(dt)}/master.{dt:%Y%m%d}.idx"
    )


def _fetch_master_form4(dt: date, sess: requests.Session) -> pd.DataFrame:
    cols = ["cik", "company", "form", "filed", "path"]
    try:
        r = _get(sess, _daily_master_url(dt))
        if r.status_code in (403, 404):
            return pd.DataFrame(columns=cols)
        r.raise_for_status()
    except Exception as e:
        log.warning("master index %s failed: %s: %s", dt, type(e).__name__, e)
        return pd.DataFrame(columns=cols)

    rows, started = [], False
    for line in r.text.splitlines():
        if not started:
            if line.strip().startswith("-----"):
                started = True
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        cik, company, form, filed, path = parts[:5]
        if form.strip().upper() in ("4", "4/A"):
            rows.append((cik.strip(), company.strip(), form.strip().upper(),
                         filed.strip(), path.strip()))
    return pd.DataFrame(rows, columns=cols)


def _accession_base_from_path(path: str) -> str | None:
    m = re.search(r"edgar/data/(\d+)/(\d{10})-(\d{2})-(\d{6})\.txt$", path)
    if not m:
        return None
    cik, a, b, c = m.groups()
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{a}{b}{c}"


def _choose_ownership_xml(files: list[dict]) -> str | None:
    named = [
        f["name"] for f in files
        if f.get("name", "").lower().endswith(".xml")
        and ("ownership" in f["name"].lower() or "primary" in f["name"].lower())
    ]
    if named:
        return named[0]
    any_xml = [f["name"] for f in files if f.get("name", "").lower().endswith(".xml")]
    return any_xml[0] if any_xml else None


def _parse_form4_ownership_xml(xml_text: str, filed_dt: pd.Timestamp) -> list[dict]:
    """Extract open-market purchase (code 'P') rows from a Form 4 XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.debug("XML parse error: %s", e)
        return []

    ns = {"n": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}

    def find(elem, path):
        return elem.find(path, ns) if ns else elem.find(path)

    def findall(elem, path):
        # ElementTree needs the namespace prefix baked into the path.
        if ns:
            path = "/".join(
                seg if seg in (".", "..", "") or seg.startswith("n:") else f"n:{seg}"
                for seg in path.split("/")
            )
            return elem.findall(path, ns)
        return elem.findall(path)

    def text_of(elem, path):
        node = find(elem, path)
        return node.text.strip() if node is not None and node.text else None

    issuer = find(root, ".//issuer")
    ticker = text_of(issuer, "issuerTradingSymbol") if issuer is not None else None
    company = text_of(issuer, "issuerName") if issuer is not None else None

    owner = find(root, ".//reportingOwner")
    insider = title = None
    if owner is not None:
        rid = find(owner, "reportingOwnerId")
        if rid is not None:
            insider = text_of(rid, "rptOwnerName")
        rel = find(owner, "reportingOwnerRelationship")
        if rel is not None:
            title = text_of(rel, "officerTitle")

    nd_table = find(root, ".//nonDerivativeTable")
    if nd_table is None:
        return []

    out = []
    for tx in findall(nd_table, ".//nonDerivativeTransaction"):
        code = text_of(tx, ".//transactionCoding/transactionCode")
        if (code or "").upper() != "P":
            continue

        tx_date_raw = text_of(tx, ".//transactionDate/value")
        tx_date = pd.to_datetime(tx_date_raw, errors="coerce") if tx_date_raw else pd.NaT

        price = to_num(text_of(tx, ".//transactionAmounts/transactionPricePerShare/value"))
        shares = to_num(text_of(tx, ".//transactionAmounts/transactionShares/value"))
        value = price * shares if np.isfinite(price) and np.isfinite(shares) else np.nan

        out.append({
            "Yahoo": (ticker or "").upper().replace(".", "-"),
            "Date": tx_date,
            "Filed": filed_dt,
            "Company": company,
            "Insider": insider,
            "Title": title,
            "Price": price,
            "Shares": shares,
            "Value": value,
            "Transaction": "P - Purchase",
        })
    return out


def _fetch_accession_rows(base_url: str, filed_dt: pd.Timestamp,
                          sess: requests.Session) -> list[dict]:
    try:
        j = _get(sess, base_url + "/index.json")
        if j.status_code in (403, 404):
            return []
        j.raise_for_status()
        name = _choose_ownership_xml(j.json().get("directory", {}).get("item", []))
        if not name:
            return []
        xmlr = _get(sess, f"{base_url}/{name}")
        if xmlr.status_code in (403, 404):
            return []
        xmlr.raise_for_status()
        return _parse_form4_ownership_xml(xmlr.text, filed_dt)
    except Exception as e:
        log.debug("accession %s: %s: %s", base_url, type(e).__name__, e)
        return []


def _fetch_day_via_master(dt: date) -> pd.DataFrame:
    """Daily master index -> per-filing XML. Filed date == dt for every row."""
    sess = session_with_retries()
    master = _fetch_master_form4(dt, sess)
    if master.empty:
        return _empty()

    jobs = []
    for _, r in master.iterrows():
        base = _accession_base_from_path(r["path"])
        if base:
            jobs.append((base, pd.to_datetime(r["filed"], errors="coerce").normalize()))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_fetch_accession_rows, b, f, sess) for b, f in jobs]
        for fut in as_completed(futures):
            rows.extend(fut.result())

    if not rows:
        return _empty()
    df = pd.DataFrame(rows).dropna(subset=["Yahoo", "Date"])
    log.info("master %s: %d purchase rows from %d filings", dt, len(df), len(jobs))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Fallback: SEC current-filings HTML
# --------------------------------------------------------------------------
def _fetch_today_via_current_html() -> pd.DataFrame:
    sess = session_with_retries()
    try:
        r = _get(sess, CURRENT_FORM4_URL)
        if r.status_code in (403, 404):
            return _empty()
        r.raise_for_status()
        tables = pd.read_html(r.text, displayed_only=False)
    except Exception as e:
        log.warning("current-filings HTML failed: %s: %s", type(e).__name__, e)
        return _empty()

    if not tables:
        return _empty()

    t = max(tables, key=lambda d: (d.shape[1], d.shape[0])).copy()
    t.columns = [str(c).strip().replace("\xa0", " ") for c in t.columns]

    def pick(names):
        low = {c.lower(): c for c in t.columns}
        for n in names:
            if n in t.columns:
                return n
            if n.lower() in low:
                return low[n.lower()]
        return None

    c_sym = pick(["Ticker", "Symbol", "Issuer Trading Symbol", "Trading Symbol", "Ticker Symbol"])
    c_comp = pick(["Issuer", "Company", "Company Name", "Issuer Name"])
    c_owner = pick(["Reporting Owner", "Owner", "Reporting Owner Name"])
    c_rel = pick(["Relationship", "Reporting Owner Relationship", "Officer Title"])
    c_desc = pick(["Transaction", "Transaction Description"])
    c_date = pick(["Transaction Date", "Date", "Reporting Date"])
    c_price = pick(["Price", "Transaction Price Per Share", "Price Per Share"])
    c_amt = pick(["Amount", "Shares", "Quantity", "Number of Securities Transacted"])
    c_code = pick(["Transaction Code", "Trans Code", "Code"])
    c_filed = pick(["Filing Date", "Filed", "FilingDate"])

    if not (c_sym and c_date and c_filed):
        log.warning("current-filings HTML: expected columns not found")
        return _empty()

    df = pd.DataFrame()
    df["Yahoo"] = t[c_sym].astype(str).str.upper().str.replace(".", "-", regex=False).str.strip()
    df["Date"] = pd.to_datetime(t[c_date], errors="coerce")
    df["Filed"] = pd.to_datetime(t[c_filed], errors="coerce").dt.normalize()
    if c_comp:
        df["Company"] = t[c_comp].astype(str).str.strip()
    if c_owner:
        df["Insider"] = t[c_owner].astype(str).str.strip()
    if c_rel:
        df["Title"] = t[c_rel].astype(str).str.strip()
    if c_desc:
        df["Transaction"] = t[c_desc].astype(str).str.strip()
    if c_price:
        df["Price"] = to_num(t[c_price])
    if c_amt:
        df["Shares"] = to_num(t[c_amt])
    if {"Price", "Shares"}.issubset(df.columns):
        df["Value"] = df["Price"] * df["Shares"]

    if c_code and c_code in t.columns:
        df = df[t[c_code].astype(str).str.upper().str.strip().eq("P")].copy()
    else:
        mask = pd.Series(True, index=df.index)
        if "Transaction" in df.columns:
            mask &= df["Transaction"].astype(str).str.contains("purchase", case=False, na=False)
        if {"Price", "Shares"}.issubset(df.columns):
            mask &= (df["Price"] > 0) & (df["Shares"] > 0)
        df = df[mask].copy()

    today_utc = pd.Timestamp.utcnow().normalize()
    df = df[df["Filed"] == today_utc]
    df = df.dropna(subset=["Yahoo", "Date"]).sort_values("Filed").groupby("Yahoo", as_index=False).tail(1)
    log.info("current-filings HTML fallback: %d rows", len(df))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def fetch_sec_day(dt: date) -> pd.DataFrame:
    """All Form 4 purchases filed on `dt`. Uses the master index, with the
    current-filings HTML as a fallback for today only."""
    df = _fetch_day_via_master(dt)
    if not df.empty:
        return df.assign(Filed=pd.to_datetime(dt))
    if dt == date.today():
        return _fetch_today_via_current_html()
    return _empty()


def _dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent purchase per ticker."""
    if df.empty:
        return df
    return (
        df.sort_values(["Filed", "Date"])
        .groupby("Yahoo", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def fetch_sec_week_raw(days: int = 7) -> pd.DataFrame:
    """Concatenated purchases for the last `days` days, WITHOUT dedup.

    Fetching once here (rather than separately for 'today' and 'week') means
    today's filings are pulled a single time per run.
    """
    frames, today = [], date.today()
    for k in range(days):
        d = today - timedelta(days=k)
        df = fetch_sec_day(d)
        if not df.empty:
            frames.append(df.assign(Filed=pd.to_datetime(d).normalize()))
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def fetch_sec_week(days: int = 7) -> pd.DataFrame:
    """Deduped weekly view (kept for direct/standalone use)."""
    return _dedup_latest(fetch_sec_week_raw(days))


def health_check() -> dict:
    try:
        sess = session_with_retries(total=1, backoff=0.1)
        r = sess.head("https://www.sec.gov", headers=SEC_HEADERS, timeout=5)
        ok = r.status_code < 500
        return {"sec_ok": bool(ok), "sec_reason": "" if ok else f"HTTP {r.status_code}"}
    except Exception as e:
        return {"sec_ok": False, "sec_reason": str(e)}
