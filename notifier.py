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
DAYS_LOOKBACK   = 1          # kept for compatibility (today)
LOOKBACK_TODAY  = 1
LOOKBACK_WEEK   = 7          # rolling 7 days section

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

#heartbeat start
# add near top of notifier.py
def send_heartbeat(note: str):
    try:
        send_email("[bot heartbeat] notifier ran", f"<p>{note}</p>")
    except Exception as e:
        print("Heartbeat email failed:", e)


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

def fetch_sec_current(days=LOOKBACK_TODAY):
    """
    SEC 'current insider transactions' page -> return ALL purchase rows (code 'P'),
    with flexible header mapping. NO role or $ filters here; we do that later so
    the 'forced' list always has content when any purchases exist.

    days: rolling calendar-day window (e.g., 1 for 'today', 7 for 'this week').
    """
    sess = session_with_retries()
    r = sess.get(SEC_URL, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(r.text, displayed_only=False)
    t = best_table_by_aliases(tables)
    if t is None or t.empty:
        return pd.DataFrame(columns=["Ticker","Company","Insider","Title","Transaction",
                                     "Price","Shares","Value","Date","Filing Date","Yahoo"])

    # normalize headers
    t = t.copy()
    t.columns = [str(c).strip().replace("\xa0", " ") for c in t.columns]

    # helper to pick a column among aliases
    def pick(names):
        for n in names:
            if n in t.columns: return n
        low = {c.lower(): c for c in t.columns}
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
        return pd.DataFrame(columns=["Ticker","Company","Insider","Title","Transaction",
                                     "Price","Shares","Value","Date","Filing Date","Yahoo"])

    df = pd.DataFrame()
    df["Ticker"] = t[c_ticker].astype(str).str.strip()
    df["Date"]   = pd.to_datetime(t[c_date], errors="coerce")

    if c_price:  df["Price"]  = t[c_price].map(to_num)
    if c_shares: df["Shares"] = t[c_shares].map(to_num)
    if {"Price","Shares"}.issubset(df.columns):
        df["Value"] = df["Price"] * df["Shares"]

    if c_title:  df["Title"]   = t[c_title].astype(str).str.strip()
    if c_owner:  df["Insider"] = t[c_owner].astype(str).str.strip()
    if c_comp:   df["Company"] = t[c_comp].astype(str).str.strip()
    if c_desc:   df["Transaction"] = t[c_desc].astype(str).str.strip()

    # keep purchases only (code 'P') if code column exists; else best-effort keep positive buys
    if c_code and c_code in t.columns:
        code = t[c_code].astype(str).str.strip().str.upper()
        df = df[code.eq("P")].copy()
    else:
        if {"Price","Shares"}.issubset(df.columns):
            df = df[(df["Price"] > 0) & (df["Shares"] > 0)]

    # filter by rolling window (calendar days)
    if df["Date"].notna().any():
        # cutoff includes today as day 1
        cutoff = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days-1))
        df = df[df["Date"] >= cutoff]

    # Ticker -> Yahoo format
    def to_yahoo(s):
        s = str(s).strip().upper()
        if not s or s == "NAN": return np.nan
        return re.sub(r"[^A-Z0-9\-.]", "", s).replace(".", "-")
    df["Yahoo"] = df["Ticker"].map(to_yahoo)

    # sort, dedupe most recent per ticker
    df = (df.dropna(subset=["Yahoo","Date"])
            .sort_values("Date")
            .groupby("Yahoo", as_index=False)
            .tail(1))

    # Order columns for downstream
    cols = [c for c in ["Yahoo","Date","Company","Insider","Title","Price","Shares","Value","Transaction"] if c in df.columns]
    return df[cols].reset_index(drop=True)

def health_check():
    """Quick diagnostics to explain empty results."""
    info = {"sec_ok": False, "sec_rows": 0, "sec_reason": "", "oi_rows_7d": 0}

    try:
        s = session_with_retries()
        r = s.get(SEC_URL, headers=SEC_HEADERS, timeout=20)
        r.raise_for_status()
        html = r.text[:2000].lower()  # first 2k chars for quick checks
        if any(bad in html for bad in ["access denied", "temporarily unavailable", "blocked", "service unavailable"]):
            info["sec_reason"] = "SEC page returned an access/availability message"
        else:
            try:
                tables = pd.read_html(r.text, displayed_only=False)
                t = best_table_by_aliases(tables)
                if t is not None and not t.empty:
                    info["sec_ok"] = True
                    # Count likely purchase rows (code 'P' or positive Price/Shares)
                    cols = {str(c).strip().replace("\xa0"," ") for c in t.columns}
                    if "Transaction Code" in cols:
                        code_col = [c for c in t.columns if str(c).strip().replace("\xa0"," ")=="Transaction Code"][0]
                        info["sec_rows"] = int((t[code_col].astype(str).str.upper().str.strip()=="P").sum())
                    else:
                        # fallback: positive price & shares rows
                        def _num(x):
                            s = str(x).replace(",", "").replace("$","").replace("%","").replace("—","").replace("\u2212","-")
                            s = s.replace("(", "-").replace(")","")
                            return pd.to_numeric(s, errors="coerce")
                        price_col = [c for c in t.columns if "price" in str(c).lower()]
                        shares_col = [c for c in t.columns if "amount" in str(c).lower() or "share" in str(c).lower()]
                        if price_col and shares_col:
                            pc, sc = price_col[0], shares_col[0]
                            p = _num(t[pc]); q = _num(t[sc])
                            info["sec_rows"] = int(((p>0) & (q>0)).sum())
                else:
                    info["sec_reason"] = "SEC: no suitable table found"
            except Exception as pe:
                info["sec_reason"] = f"SEC parse error: {pe}"
    except Exception as he:
        info["sec_reason"] = f"SEC request error: {he}"

    # Probe OpenInsider 7d quickly
    try:
        oi = probe_openinsider_7d()
        info["oi_rows_7d"] = int(len(oi))
    except Exception:
        pass

    return info


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
    cl = (df.assign(_one=1).set_index("Date")
            .groupby("Yahoo")["_one"].rolling("10D").sum()
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
        df["mom_3m"]   = [mom_from(close_all, y, d, 63)   for y, d in zip(df["Yahoo"], df["Date"])]
        df["mom_12_1"] = [mom_12m_minus_1m(close_all, y, d) for y, d in zip(df["Yahoo"], df["Date"])]
        df["dvol20"]   = [dvol20(close_all, vol_all, y, d) for y, d in zip(df["Yahoo"], df["Date"])]
        df["px_now"]   = [price_at(close_all, y, d, 0)     for y, d in zip(df["Yahoo"], df["Date"])]

    # role flags & value
    df["is_CEO"] = df.get("Title","").astype(str).str.contains(r"\bCEO\b", case=False, na=False)
    df["is_CFO"] = df.get("Title","").astype(str).str.contains(r"\bCFO\b", case=False, na=False)
    df["Value"]  = pd.to_numeric(df.get("Value"), errors="coerce").fillna(0.0)   # allow 0 so log1p=0

    # normalized features (date-wise)
    df["f_insider_size"] = np.log1p(df["Value"]).replace([np.inf,-np.inf], np.nan)
    df["f_cluster"]      = pd.to_numeric(df["cluster_10d"], errors="coerce")
    df["f_mom_trend"]    = pd.to_numeric(df["mom_12_1"], errors="coerce")
    df["f_mom_contra"]   = -pd.to_numeric(df["mom_3m"], errors="coerce")

    def rank01_local(s):
        r = s.rank(pct=True, method="average")
        return (r if not r.isna().all() else pd.Series(0.5, index=s.index)).fillna(0.5)

    for col in ["f_insider_size","f_cluster","f_mom_trend","f_mom_contra"]:
        df[col+"_n"] = df.groupby("Date")[col].transform(rank01_local)

    df["Score"] = (
        W_INSIDER_SZ*df["f_insider_size_n"] +
        W_CLUSTER   *df["f_cluster_n"] +
        W_MOM_TREND *df["f_mom_trend_n"] +
        W_MOM_CONTRA*df["f_mom_contra_n"]
    )

    # --- forced: everyone gets a score ---
    order_cols = ["Yahoo","Date","Company","Insider","Title","Price","Shares","Value",
                  "cluster_10d","mom_12_1","mom_3m","px_now","dvol20","Score"]
    forced = df[[c for c in order_cols if c in df.columns]].sort_values(["Score","Value"], ascending=[False,False])

    # --- matches: apply hygiene filters now ---
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

def render_section(section_title, matches, forced):
    return (
        df_to_html(matches, f"{section_title} — Matches (pass filters + score)")
        + df_to_html(forced, f"{section_title} — All-ranked (forced scores)")
        + "<hr/>"
    )

# =========================
# ---------- MAIN ---------
# =========================
def main():
    try:
        # --- Health/diagnostics so emails explain empties ---
        diag = health_check()
        diag_text = (
            f"SEC ok: {diag.get('sec_ok')} | "
            f"SEC purchase rows: {diag.get('sec_rows')} | "
            f"SEC reason: {diag.get('sec_reason') or '—'} | "
            f"OpenInsider 7d rows: {diag.get('oi_rows_7d')}"
        )

        # --- Today (SEC current) ---
        raw_today = fetch_sec_current()
        matches_today, forced_today = add_features_and_score(raw_today)

        # --- This week (7d) via OpenInsider probe as fallback/context ---
        matches_week, forced_week = pd.DataFrame(), pd.DataFrame()
        try:
            oi7 = probe_openinsider_7d()  # may be empty
            if not oi7.empty:
                # Map minimal columns into our schema so scorer can run
                rename = {
                    "Ticker":"Ticker","Company Name":"Company","Insider Name":"Insider",
                    "Title":"Title","Trade Type":"Transaction","Price":"Price",
                    "Qty":"Shares","Value ($)":"Value","Trade Date":"Date","Filing Date":"Filing Date"
                }
                oi7 = oi7.rename(columns={k:v for k,v in rename.items() if k in oi7.columns})
                # numeric/date cleanup
                for c in ("Price","Shares","Value"):
                    if c in oi7.columns: oi7[c] = to_num(oi7[c])
                for c in ("Date","Filing Date"):
                    if c in oi7.columns: oi7[c] = pd.to_datetime(oi7[c], errors="coerce")

                matches_week, forced_week = add_features_and_score(oi7)
        except Exception:
            # don’t let weekly context break the run
            pass

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

        <p style="color:#999">Sources: SEC current insider transactions (today); OpenInsider (7d probe). Not investment advice.</p>
        """

        send_email(subject, html)

        # optional success heartbeat
        try:
            send_heartbeat("Run OK — report sent.")
        except Exception:
            pass

    except Exception:
        tb = traceback.format_exc()
        print("Error:\n", tb)
        # email the error so you see it
        try:
            send_email("Insider bot error", f"<pre>{tb}</pre>")
        except Exception:
            pass
        # optional failure heartbeat
        try:
            send_heartbeat("Run failed — error mailed.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
