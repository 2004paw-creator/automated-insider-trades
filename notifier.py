# notifier.py
# Daily SEC insider purchase scanner + scorer + Gmail emailer.
# Uses official EDGAR daily master index + index.json + ownership XML (no scraping).
# Requires: pandas, requests, lxml (or html5lib), yfinance

import os, re, time, math, traceback, json
from datetime import datetime, timedelta, date
from email.mime.text import MIMEText
from email.utils import formatdate
import smtplib
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import yfinance as yf

# =========================
# ------- CONFIG ----------
# =========================
LOOKBACK_TODAY  = 1          # today (via daily master index)
LOOKBACK_WEEK   = 7          # rolling 7 calendar days

REQUIRE_CEO_CFO = True       # role filter
MIN_VALUE       = 100_000    # min $ value
MAX_EMAIL_ROWS  = 50         # cap in email

# Scoring knobs (same spirit as your notebook)
RECENT_DAYS     = 30         # normalize by event date; most rows are same day
MIN_PRICE       = 3.0
MIN_DVOL        = 2_000_000  # 20D avg $ volume
CEO_ONLY        = False      # if True, only CEO; else CEO or CFO
# feature weights (sum ~1)
W_MOM_TREND     = 0.30       # (12–1m) momentum
W_MOM_CONTRA    = 0.20       # contrarian last 3m (more negative = better)
W_INSIDER_SZ    = 0.35       # log($ value)
W_CLUSTER       = 0.15       # insider cluster size (10D)

# Email + SEC headers
def _clean_env(x): return (x or "").strip().replace("\r","").replace("\n","")
GMAIL_USER      = _clean_env(os.getenv("GMAIL_USER"))
GMAIL_APP_PASS  = _clean_env(os.getenv("GMAIL_APP_PASSWORD"))
TO_EMAIL        = _clean_env(os.getenv("TO_EMAIL"))
UA_EMAIL        = GMAIL_USER if GMAIL_USER else "you@example.com"

SUBJECT_PREFIX  = "Insider buys (SEC)"
YF_LINK_FMT     = "https://finance.yahoo.com/quote/{sym}"

# Polite headers for SEC
SEC_HEADERS = {
    "User-Agent": f"Mozilla/5.0 (compatible; InsiderBot/1.0; +mailto:{UA_EMAIL})",
    "From": UA_EMAIL,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

# =========================
# ---- HTTP / UTILITIES ---
# =========================
def session_with_retries(total=5, backoff=0.6):
    s = requests.Session()
    r = Retry(total=total, connect=total, read=total,
              backoff_factor=backoff,
              status_forcelist=(429,500,502,503,504),
              allowed_methods=frozenset(["GET","HEAD"]))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://",  HTTPAdapter(max_retries=r))
    return s

def to_num(x):
    s = str(x).replace(",", "").replace("$","").replace("%","").replace("—","").replace("\u2212","-")
    s = s.replace("(", "-").replace(")","")
    return pd.to_numeric(s, errors="coerce")

# =========================
# ---- EDGAR DAILY INDEX --
# =========================
def _qtr_of(dt: date) -> int:
    return (dt.month - 1)//3 + 1

def _daily_master_url(dt: date) -> str:
    # Example: https://www.sec.gov/Archives/edgar/daily-index/2025/QTR4/master.20251010.idx
    y, q = dt.year, _qtr_of(dt)
    return f"https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/master.{dt:%Y%m%d}.idx"

def _fetch_master_form4(dt: date) -> pd.DataFrame:
    """
    Return rows from the daily master index for Form 4 / 4/A.
    Columns: cik, company, form, filed, path
    """
    url = _daily_master_url(dt)
    sess = session_with_retries(total=4, backoff=0.8)
    r = sess.get(url, headers=SEC_HEADERS, timeout=30)
    if r.status_code == 404:
        return pd.DataFrame(columns=["cik","company","form","filed","path"])
    r.raise_for_status()
    lines = r.text.splitlines()

    rows, started = [], False
    for line in lines:
        if not started:
            if line.strip().startswith("-----"):
                started = True
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        cik, company, form, filed, path = parts[:5]
        form = form.strip().upper()
        if form in ("4", "4/A"):
            rows.append((cik.strip(), company.strip(), form, filed.strip(), path.strip()))
    return pd.DataFrame(rows, columns=["cik","company","form","filed","path"])

def _accession_base_from_path(path: str) -> str | None:
    # master path: edgar/data/320193/0000320193-25-000012.txt
    # index.json : https://www.sec.gov/Archives/edgar/data/320193/000032019325000012/index.json
    m = re.search(r"edgar/data/(\d+)/(\d{10})-(\d{2})-(\d{6})\.txt$", path)
    if not m:
        return None
    cik, a, b, c = m.groups()
    acc_nodash = f"{a}{b}{c}"
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"

def _choose_ownership_xml(files: list[dict]) -> str | None:
    # prefer ownership or primary xml
    candidates = []
    for f in files:
        name = f.get("name","").lower()
        if name.endswith(".xml") and ("ownership" in name or "primary" in name):
            candidates.append(f["name"])
    if not candidates:
        for f in files:
            name = f.get("name","").lower()
            if name.endswith(".xml"):
                candidates.append(f["name"])
    return candidates[0] if candidates else None

def _parse_form4_ownership_xml(xml_text: str) -> list[dict]:
    """
    Extract non-derivative purchases (transactionCode=='P').
    Returns dicts: Yahoo, Date, Company, Insider, Title, Price, Shares, Value, Transaction
    """
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    ns = {"n": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    def xp(elem, path):
        return elem.find(path, ns) if ns else elem.find(path)

    issuer = xp(root, ".//issuer")
    ticker = xp(issuer, "issuerTradingSymbol").text.strip() if issuer is not None and xp(issuer, "issuerTradingSymbol") is not None else None
    company = xp(issuer, "issuerName").text.strip() if issuer is not None and xp(issuer, "issuerName") is not None else None

    owner  = xp(root, ".//reportingOwner")
    insider = None
    title   = None
    if owner is not None:
        rid = xp(owner, "reportingOwnerId")
        if rid is not None and xp(rid, "rptOwnerName") is not None:
            insider = xp(rid, "rptOwnerName").text.strip()
        rel = xp(owner, "reportingOwnerRelationship")
        if rel is not None and xp(rel, "officerTitle") is not None and xp(rel, "isOfficer") is not None:
            if (xp(rel, "isOfficer").text or "").strip().lower() == "1":
                title = xp(rel, "officerTitle").text.strip()

    nd_table = xp(root, ".//nonDerivativeTable")
    if nd_table is None:
        return out

    tx_nodes = nd_table.findall(".//nonDerivativeTransaction") if ns=={} else nd_table.findall(".//n:nonDerivativeTransaction", ns)
    for tx in tx_nodes:
        code_el = xp(tx, ".//transactionCoding/transactionCode")
        code = code_el.text.strip().upper() if code_el is not None and code_el.text else ""
        if code != "P":
            continue
        d_el = xp(tx, ".//transactionDate/value")
        tx_date = pd.to_datetime(d_el.text.strip(), errors="coerce") if d_el is not None and d_el.text else pd.NaT
        price_el  = xp(tx, ".//transactionAmounts/transactionPricePerShare/value")
        shares_el = xp(tx, ".//transactionAmounts/transactionShares/value")
        try:
            price  = float(price_el.text.replace(",","")) if price_el is not None and price_el.text else np.nan
        except Exception:
            price = np.nan
        try:
            shares = float(shares_el.text.replace(",","")) if shares_el is not None and shares_el.text else np.nan
        except Exception:
            shares = np.nan
        value = price * shares if np.isfinite(price) and np.isfinite(shares) else np.nan

        out.append({
            "Yahoo": (ticker or "").upper().replace(".","-"),
            "Date":  tx_date,
            "Company": company,
            "Insider": insider,
            "Title": title,
            "Price": price,
            "Shares": shares,
            "Value": value,
            "Transaction": "P - Purchase",
        })
    return out

def fetch_sec_day(dt: date) -> pd.DataFrame:
    """
    Use daily master index -> index.json -> ownership XML to return all purchase rows for a day.
    """
    master = _fetch_master_form4(dt)
    if master.empty:
        return pd.DataFrame(columns=["Yahoo","Date","Company","Insider","Title","Price","Shares","Value","Transaction"])
    sess = session_with_retries(total=4, backoff=0.8)
    rows = []
    for _, r in master.iterrows():
        base = _accession_base_from_path(r["path"])
        if not base:
            continue
        try:
            j = sess.get(base + "/index.json", headers=SEC_HEADERS, timeout=30)
            j.raise_for_status()
            meta = j.json()
            name = _choose_ownership_xml(meta.get("directory", {}).get("item", []))
            if not name:
                continue
            xmlr = sess.get(f"{base}/{name}", headers=SEC_HEADERS, timeout=30)
            xmlr.raise_for_status()
            rows.extend(_parse_form4_ownership_xml(xmlr.text))
            time.sleep(0.15)  # polite pause
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["Yahoo","Date","Company","Insider","Title","Price","Shares","Value","Transaction"])
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Yahoo","Date"]).reset_index(drop=True)
    return df

def fetch_sec_week(days=7) -> pd.DataFrame:
    """
    Aggregate the last `days` calendar days using the public APIs above.
    Keep 1-most-recent row per ticker.
    """
    out = []
    today = date.today()
    for k in range(days):
        dt = today - timedelta(days=k)
        try:
            d = fetch_sec_day(dt)
            if not d.empty:
                out.append(d)
        except Exception:
            continue
    if not out:
        return pd.DataFrame(columns=["Yahoo","Date","Company","Insider","Title","Price","Shares","Value","Transaction"])
    df = pd.concat(out, ignore_index=True)
    df = df.sort_values("Date").groupby("Yahoo", as_index=False).tail(1)
    return df.reset_index(drop=True)

def health_check():
    """Quick diagnostics using the official flow."""
    try:
        today_rows = len(fetch_sec_day(date.today()))
        week_rows  = len(fetch_sec_week(7))
        return {"sec_ok": True, "sec_rows_today": int(today_rows), "sec_rows_week": int(week_rows), "sec_reason": ""}
    except Exception as e:
        return {"sec_ok": False, "sec_rows_today": 0, "sec_rows_week": 0, "sec_reason": str(e)}

# =========================
# ------- SCORING ---------
# =========================
def build_prices(ins_df: pd.DataFrame):
    """Download yfinance Close & Volume for tickers around the event dates."""
    if ins_df.empty: 
        return pd.DataFrame(), pd.DataFrame()
    tickers = sorted(ins_df["Yahoo"].dropna().unique().tolist())
    start = (pd.to_datetime(ins_df["Date"]).min() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end   = (pd.to_datetime(ins_df["Date"]).max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if px.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(px.columns, pd.MultiIndex):
        close_all = px["Adj Close"].copy() if "Adj Close" in px.columns.levels[0] else px["Close"].copy()
        vol_all   = px["Volume"].copy()
    else:
        sym = tickers[0]
        close_all = px[["Adj Close" if "Adj Close" in px.columns else "Close"]].rename(columns=lambda _: sym)
        vol_all   = px[["Volume"]].rename(columns=lambda _: sym)
    close_all = close_all.loc[:, ~close_all.columns.duplicated()].astype(float)
    vol_all   = vol_all.loc[:, ~vol_all.columns.duplicated()].astype(float)
    bad = [c for c in close_all.columns if close_all[c].isna().all()]
    if bad:
        close_all = close_all.drop(columns=bad, errors="ignore")
        vol_all   = vol_all.drop(columns=bad,   errors="ignore")
    return close_all, vol_all

def price_at(close_all, y, dt, off=0):
    try:
        idx = close_all.index
        i = idx.searchsorted(pd.Timestamp(dt))
        j = i + off
        if j < 0 or j >= len(idx): return np.nan
        return float(close_all[y].iloc[j])
    except Exception:
        return np.nan

def mom_from(close_all, y, dt, lb):
    p0 = price_at(close_all, y, dt, 0); pL = price_at(close_all, y, dt, -lb)
    if not np.isfinite(p0) or not np.isfinite(pL) or pL == 0: return np.nan
    return p0/pL - 1.0

def mom_12m_minus_1m(close_all, y, dt):
    p_1m = price_at(close_all, y, dt, -21); p_12m = price_at(close_all, y, dt, -252)
    if not np.isfinite(p_1m) or not np.isfinite(p_12m) or p_12m == 0: return np.nan
    return p_1m/p_12m - 1.0

def dvol20(close_all, vol_all, y, dt):
    try:
        idx = close_all.index
        i = idx.searchsorted(pd.Timestamp(dt))
        lo = max(0, i-19); hi = i+1
        px = close_all[y].iloc[lo:hi].astype(float)
        vo = vol_all[y].iloc[lo:hi].astype(float)
        if px.empty or vo.empty: return np.nan
        return float((px * vo).mean())
    except Exception:
        return np.nan

def add_features_and_score(raw_df: pd.DataFrame):
    """
    Build features + composite Score.
    - 'matches': passed hygiene filters (price, dollar volume, role) + Score
    - 'forced':  ALL rows with Score, even if price data missing (neutral where needed)
    """
    if raw_df is None or raw_df.empty:
        empty_cols = ["Yahoo","Date","Company","Insider","Title","Price","Shares","Value",
                      "cluster_10d","mom_12_1","mom_3m","px_now","dvol20","Score"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date","Yahoo"]).reset_index(drop=True)

    # cluster (10 calendar days within ticker)
    df = df.sort_values(["Yahoo","Date"])
    cl = (df.assign(_one=1).set_index("Date").groupby("Yahoo")["_one"].rolling("10D").sum()
          .reset_index(level=0, drop=True).astype(int))
    df["cluster_10d"] = cl.values

    # fetch prices
    close_all, vol_all = build_prices(df)
    if close_all.empty or vol_all.empty:
        df["mom_3m"]   = np.nan
        df["mom_12_1"] = np.nan
        df["dvol20"]   = np.nan
        df["px_now"]   = np.nan
    else:
        df["mom_3m"]   = [mom_from(close_all, y, d, 63)          for y, d in zip(df["Yahoo"], df["Date"])]
        df["mom_12_1"] = [mom_12m_minus_1m(close_all, y, d)      for y, d in zip(df["Yahoo"], df["Date"])]
        df["dvol20"]   = [dvol20(close_all, vol_all, y, d)       for y, d in zip(df["Yahoo"], df["Date"])]
        df["px_now"]   = [price_at(close_all, y, d, 0)           for y, d in zip(df["Yahoo"], df["Date"])]

    # role flags & value
    df["is_CEO"] = df.get("Title","").astype(str).str.contains(r"\bCEO\b", case=False, na=False)
    df["is_CFO"] = df.get("Title","").astype(str).str.contains(r"\bCFO\b", case=False, na=False)
    df["Value"]  = pd.to_numeric(df.get("Value"), errors="coerce").fillna(0.0)

    # normalized features (date-wise)
    df["f_insider_size"] = np.log1p(df["Value"]).replace([np.inf,-np.inf], np.nan)
    df["f_cluster"]      = pd.to_numeric(df["cluster_10d"], errors="coerce")
    df["f_mom_trend"]    = pd.to_numeric(df["mom_12_1"], errors="coerce")
    df["f_mom_contra"]   = -pd.to_numeric(df["mom_3m"], errors="coerce")

    def _rank01(s):
        r = s.rank(pct=True, method="average")
        return (r if not r.isna().all() else pd.Series(0.5, index=s.index)).fillna(0.5)

    for col in ["f_insider_size","f_cluster","f_mom_trend","f_mom_contra"]:
        df[col+"_n"] = df.groupby("Date")[col].transform(_rank01)

    df["Score"] = (
        W_INSIDER_SZ*df["f_insider_size_n"] +
        W_CLUSTER   *df["f_cluster_n"] +
        W_MOM_TREND *df["f_mom_trend_n"] +
        W_MOM_CONTRA*df["f_mom_contra_n"]
    )

    order_cols = ["Yahoo","Date","Company","Insider","Title","Price","Shares","Value",
                  "cluster_10d","mom_12_1","mom_3m","px_now","dvol20","Score"]
    forced = df[[c for c in order_cols if c in df.columns]].sort_values(["Score","Value"], ascending=[False,False])

    price_ok = (df["px_now"] >= MIN_PRICE) if "px_now" in df else pd.Series(False, index=df.index)
    vol_ok   = (df["dvol20"] >= MIN_DVOL)  if "dvol20" in df else pd.Series(False, index=df.index)
    role_ok  = (df["is_CEO"] if CEO_ONLY else (df["is_CEO"] | df["is_CFO"])) if REQUIRE_CEO_CFO else pd.Series(True, index=df.index)
    value_ok = (df["Value"] >= MIN_VALUE) if "Value" in df else pd.Series(True, index=df.index)

    mask = price_ok.fillna(False) & vol_ok.fillna(False) & role_ok.fillna(False) & value_ok.fillna(False)
    matches = df.loc[mask, order_cols].dropna(how="all", axis=1) if mask.any() else forced.iloc[0:0]

    return matches.head(MAX_EMAIL_ROWS), forced.head(MAX_EMAIL_ROWS)

# =========================
# -------- EMAIL ----------
# =========================
def send_email(subject: str, html: str):
    assert GMAIL_USER and GMAIL_APP_PASS and TO_EMAIL, \
        "Missing GMAIL_USER, GMAIL_APP_PASSWORD or TO_EMAIL env vars."

    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = TO_EMAIL
    msg["Date"]    = formatdate(localtime=True)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_USER, GMAIL_APP_PASS)
        smtp.sendmail(GMAIL_USER, [TO_EMAIL], msg.as_string())

def df_to_html(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h4>{title}</h4><p>No rows.</p>"
    d = df.copy()
    if "Yahoo" in d.columns:
        d["Yahoo"] = d["Yahoo"].apply(lambda s: f'<a href="{YF_LINK_FMT.format(sym=s)}">{s}</a>')
    for col in ("Price","px_now"):
        if col in d.columns: d[col] = pd.to_numeric(d[col], errors="coerce").map(lambda v: f"${v:,.2f}" if pd.notna(v) else "")
    for col in ("Shares",):
        if col in d.columns: d[col] = pd.to_numeric(d[col], errors="coerce").map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    for col in ("Value","dvol20"):
        if col in d.columns: d[col] = pd.to_numeric(d[col], errors="coerce").map(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
    if "Score" in d.columns:
        d["Score"] = pd.to_numeric(d["Score"], errors="coerce").map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    preferred = ["Yahoo","Date","Company","Insider","Title","Price","Shares","Value",
                 "cluster_10d","mom_12_1","mom_3m","px_now","dvol20","Score"]
    cols = [c for c in preferred if c in d.columns]
    return f"<h4>{title}</h4>" + d[cols].to_html(index=False, escape=False)

def send_heartbeat(note: str):
    try:
        send_email("[bot heartbeat] notifier ran", f"<p>{note}</p>")
    except Exception as e:
        print("Heartbeat email failed:", e)

# =========================
# ---------- MAIN ---------
# =========================
def main():
    try:
        # Health
        diag = health_check()
        diag_text = (f"SEC ok: {diag.get('sec_ok')} | "
                     f"today rows: {diag.get('sec_rows_today')} | "
                     f"week rows: {diag.get('sec_rows_week')} | "
                     f"reason: {diag.get('sec_reason') or '—'}")

        # Today via EDGAR
        raw_today = fetch_sec_day(date.today())
        matches_today, forced_today = add_features_and_score(raw_today)

        # This week via EDGAR
        raw_week = fetch_sec_week(days=LOOKBACK_WEEK)
        matches_week, forced_week = add_features_and_score(raw_week)

        n_match_t = 0 if matches_today is None else len(matches_today)
        n_forced_t = 0 if forced_today  is None else len(forced_today)
        n_match_w = 0 if matches_week  is None else len(matches_week)
        n_forced_w = 0 if forced_week  is None else len(forced_week)

        subject = (f"{SUBJECT_PREFIX}: today {n_match_t}/{n_forced_t} | "
                   f"week {n_match_w}/{n_forced_w}")

        html = f"""
        <h3>{SUBJECT_PREFIX}</h3>
        <p><b>Health:</b> {diag_text}</p>
        <p>Filters: CEO/CFO={REQUIRE_CEO_CFO} | Min Value=${MIN_VALUE:,} | Min Price=${MIN_PRICE} | Min $Vol(20D)=${MIN_DVOL:,}</p>

        {df_to_html(matches_today, "Today — Matches (pass filters + score)")}
        {df_to_html(forced_today,  "Today — All-ranked (forced scores)")}

        {df_to_html(matches_week,  "This Week (rolling 7 days) — Matches (pass filters + score)")}
        {df_to_html(forced_week,   "This Week (rolling 7 days) — All-ranked (forced scores)")}

        <p style="color:#999">Source: SEC EDGAR (Form 4 purchases parsed from ownership XML). Not investment advice.</p>
        """
        send_email(subject, html)

        try:
            send_heartbeat("Run OK — report sent.")
        except Exception:
            pass

    except Exception:
        tb = traceback.format_exc()
        print("Error:\n", tb)
        try:
            send_email("Insider bot error", f"<pre>{tb}</pre>")
        except Exception:
            pass
        try:
            send_heartbeat("Run failed — error mailed.")
        except Exception:
            pass

if __name__ == "__main__":
    main()
