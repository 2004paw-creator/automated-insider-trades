# notifier.py
# Daily SEC insider purchase scanner + scorer + Gmail emailer.
# No API keys. Requires: pandas, requests, lxml (or html5lib), yfinance, python-dateutil.

import os, re, math, time, traceback
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.utils import formatdate
import smtplib

import numpy as np
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import yfinance as yf
from dateutil.parser import parse as dtparse

# =========================
# ------- CONFIG ----------
# =========================
# SEC fetch
DAYS_LOOKBACK   = 1          # SEC “current” is essentially today; keep 1
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

SEC_URL         = "https://www.sec.gov/cgi-bin/own-disp?action=getcurrent"
SEC_HEADERS     = {"User-Agent": f"InsiderBot/1.0 (+mailto:{UA_EMAIL})"}

# =========================
# ---- HTTP / SEC FETCH ---
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

def best_table_by_aliases(tables):
    aliases = {
        "ticker": ["Ticker","Symbol","Issuer Trading Symbol","Trading Symbol","Ticker Symbol"],
        "date":   ["Transaction Date","Date","Reporting Date"],
        "price":  ["Price","Transaction Price Per Share","Price Per Share"],
        "shares": ["Amount","Shares","Quantity","Number of Securities Transacted"],
    }
    def score(df):
        cols = {str(c).strip().replace("\xa0"," ") for c in df.columns}
        sc = 0
        for names in aliases.values():
            sc += int(any(n in cols for n in names))
        return (sc, df.shape[1], df.shape[0])
    best, key = None, (-1,-1,-1)
    for t in tables:
        t.columns = [str(c).strip().replace("\xa0"," ") for c in t.columns]
        k = score(t)
        if k > key:
            best, key = t, k
    return best

def fetch_sec_current():
    """Pull the SEC 'current insider transactions' page and return a clean DataFrame of purchases."""
    sess = session_with_retries()
    r = sess.get(SEC_URL, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(r.text, displayed_only=False)
    t = best_table_by_aliases(tables)
    if t is None or t.empty:
        return pd.DataFrame()

    df = t.copy()

    # Flexible column picks
    def pick(names):
        for n in names:
            if n in df.columns: return n
        low = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in low: return low[n.lower()]
        return None

    c_ticker = pick(["Ticker","Symbol","Issuer Trading Symbol","Trading Symbol","Ticker Symbol"])
    c_date   = pick(["Transaction Date","Date","Reporting Date"])
    c_price  = pick(["Price","Transaction Price Per Share","Price Per Share"])
    c_shares = pick(["Amount","Shares","Quantity","Number of Securities Transacted"])
    c_title  = pick(["Relationship","Reporting Owner Relationship","Officer Title"])
    c_owner  = pick(["Reporting Owner","Owner","Reporting Owner Name"])
    c_comp   = pick(["Issuer","Company","Issuer Name","Company Name"])
    c_code   = pick(["Transaction Code","Trans Code","Code"])
    c_desc   = pick(["Transaction","Transaction Description"])

    if not c_ticker or not c_date:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["Ticker"] = df[c_ticker].astype(str).str.strip()
    out["Date"]   = pd.to_datetime(df[c_date], errors="coerce")

    if c_price:  out["Price"]  = df[c_price].map(to_num)
    if c_shares: out["Shares"] = df[c_shares].map(to_num)
    if {"Price","Shares"}.issubset(out.columns):
        out["Value"] = out["Price"] * out["Shares"]

    if c_title:  out["Title"]   = df[c_title].astype(str).str.strip()
    if c_owner:  out["Insider"] = df[c_owner].astype(str).str.strip()
    if c_comp:   out["Company"] = df[c_comp].astype(str).str.strip()
    if c_desc:   out["Transaction"] = df[c_desc].astype(str).str.strip()

    # Keep only Purchases: transaction code 'P' (if present)
    if c_code:
        code = df[c_code].astype(str).str.strip().str.upper()
        out = out[code.eq("P")].copy()
    else:
        if {"Price","Shares"}.issubset(out.columns):
            out = out[(out["Price"] > 0) & (out["Shares"] > 0)]

    # CEO/CFO filter
    if REQUIRE_CEO_CFO and "Title" in out.columns:
        mask = out["Title"].str.contains(r"\b(CEO|CFO)\b", case=False, regex=True)
        out = out[mask]

    # Min $ filter
    if "Value" in out.columns:
        out = out[out["Value"] >= MIN_VALUE]

    # Clean tickers to Yahoo format (BRK.B -> BRK-B)
    def to_yahoo(t):
        t = str(t).strip().upper()
        if not t or t == "NAN": return np.nan
        return re.sub(r"[^A-Z0-9\-.]", "", t).replace(".", "-")
    out["Yahoo"] = out["Ticker"].map(to_yahoo)

    # One-most-recent per ticker (if duplicates)
    out = (out.dropna(subset=["Yahoo","Date"])
              .sort_values("Date")
              .groupby("Yahoo", as_index=False)
              .tail(1))

    # Order
    cols = [c for c in ["Yahoo","Date","Company","Insider","Title","Price","Shares","Value","Transaction"] if c in out.columns]
    out = out.sort_values(["Date","Value"], ascending=[False, False])[cols].reset_index(drop=True)
    return out

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
    # Clean
    close_all = close_all.loc[:, ~close_all.columns.duplicated()].astype(float)
    vol_all   = vol_all.loc[:, ~vol_all.columns.duplicated()].astype(float)
    # Drop no-data columns
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

def rank01(s):
    r = s.rank(pct=True, method="average")
    if r.isna().all(): 
        return pd.Series(0.5, index=s.index)
    return r.fillna(0.5)

def add_features_and_score(raw_df: pd.DataFrame):
    """
    Returns:
      matches_df -> rows that pass hygiene filters + Score
      forced_df  -> all rows with Score (even if filters fail)
    """
    if raw_df.empty:
        return raw_df.assign(Score=np.nan), raw_df.assign(Score=np.nan)

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date","Yahoo"]).reset_index(drop=True)

    # cluster count over last 10 calendar days within same ticker
    df = df.sort_values(["Yahoo","Date"])
    cluster = (df.assign(_one=1).set_index("Date")
                 .groupby("Yahoo")["_one"].rolling("10D").sum()
                 .reset_index(level=0, drop=True).astype(int))
    df["cluster_10d"] = cluster.values

    # prices for features
    close_all, vol_all = build_prices(df)
    if close_all.empty or vol_all.empty:
        # cannot compute price-based features; force neutral
        df["mom_3m"] = np.nan; df["mom_12_1"] = np.nan; df["dvol20"] = np.nan
        df["px_now"] = np.nan
    else:
        df["mom_3m"]   = [mom_from(close_all, y, d, 63) for y, d in zip(df["Yahoo"], df["Date"])]
        df["mom_12_1"] = [mom_12m_minus_1m(close_all, y, d) for y, d in zip(df["Yahoo"], df["Date"])]
        df["dvol20"]   = [dvol20(close_all, vol_all, y, d) for y, d in zip(df["Yahoo"], df["Date"])]
        df["px_now"]   = [price_at(close_all, y, d, 0) for y, d in zip(df["Yahoo"], df["Date"])]

    # role flags & value
    df["is_CEO"] = df.get("Title","").astype(str).str.contains(r"\bCEO\b", case=False, na=False)
    df["is_CFO"] = df.get("Title","").astype(str).str.contains(r"\bCFO\b", case=False, na=False)
    df["Value"]  = pd.to_numeric(df.get("Value"), errors="coerce")

    # --- build normalized features ---
    df["f_insider_size"] = np.log1p(df["Value"]).replace([np.inf, -np.inf], np.nan)
    df["f_cluster"]      = pd.to_numeric(df["cluster_10d"], errors="coerce")
    df["f_mom_trend"]    = pd.to_numeric(df["mom_12_1"], errors="coerce")     # higher better
    df["f_mom_contra"]   = -pd.to_numeric(df["mom_3m"], errors="coerce")      # more negative 3m -> better

    # normalize cross-section by event date
    for col in ["f_insider_size","f_cluster","f_mom_trend","f_mom_contra"]:
        df[col+"_n"] = df.groupby("Date")[col].transform(rank01)

    # composite score for ALL rows (forced)
    df["Score"] = (
        W_INSIDER_SZ*df["f_insider_size_n"] +
        W_CLUSTER   *df["f_cluster_n"] +
        W_MOM_TREND *df["f_mom_trend_n"] +
        W_MOM_CONTRA*df["f_mom_contra_n"]
    )

    forced = df.copy()

    # hygiene filters for "matches"
    price_ok = df["px_now"] >= MIN_PRICE if "px_now" in df else pd.Series(False, index=df.index)
    vol_ok   = df["dvol20"] >= MIN_DVOL  if "dvol20" in df else pd.Series(False, index=df.index)
    role_ok  = df["is_CEO"] if CEO_ONLY else (df["is_CEO"] | df["is_CFO"]) if REQUIRE_CEO_CFO else pd.Series(True, index=df.index)

    mask = price_ok.fillna(False) & vol_ok.fillna(False) & role_ok.fillna(False)
    matches = df.loc[mask].copy()

    # pretty/ordering
    def fmt_buck(x):
        try:
            x = float(x)
            if not np.isfinite(x): return ""
            return f"${x:,.0f}"
        except: return ""

    order_cols = ["Yahoo","Date","Company","Insider","Title","Price","Shares","Value",
                  "cluster_10d","mom_12_1","mom_3m","px_now","dvol20","Score"]
    matches = matches[[c for c in order_cols if c in matches.columns]].sort_values(["Score","Value"], ascending=[False,False])
    forced  = forced [[c for c in order_cols if c in forced.columns ]].sort_values(["Score","Value"],  ascending=[False,False])

    # top caps for email
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
    # nice formatting
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
    preferred = ["Yahoo","Date","Company","Insider","Title","Price","Shares","Value","cluster_10d","mom_12_1","mom_3m","px_now","dvol20","Score"]
    cols = [c for c in preferred if c in d.columns]
    return f"<h4>{title}</h4>" + d[cols].to_html(index=False, escape=False)

# =========================
# ---------- MAIN ---------
# =========================
def main():
    try:
        raw = fetch_sec_current()                  # SEC pulls (today)
        matches, forced = add_features_and_score(raw)

        n_match = 0 if matches is None else len(matches)
        n_forced = 0 if forced  is None else len(forced)

        subject = f"{SUBJECT_PREFIX}: {n_match} match(es); forced list {n_forced}"
        html = f"""
        <h3>{SUBJECT_PREFIX}</h3>
        <p>Filters: CEO/CFO={REQUIRE_CEO_CFO} | Min Value=${MIN_VALUE:,} | Min Price=${MIN_PRICE} | Min $Vol(20D)=${MIN_DVOL:,}</p>
        {df_to_html(matches, "Matches (pass filters + score)")}
        {df_to_html(forced,  "All-ranked (forced scores, even if filters fail)")}
        <p style="color:#999">Source: SEC current insider transactions. Not investment advice.</p>
        """
        send_email(subject, html)
        print("Email sent.")
    except Exception:
        tb = traceback.format_exc()
        print("Error:\n", tb)
        # try to email the error so you see it
        try:
            send_email("Insider bot error", f"<pre>{tb}</pre>")
        except Exception:
            pass

if __name__ == "__main__":
    main()
