import os, re, io, ssl, smtplib, time, math, json, traceback
import numpy as np
import pandas as pd
import requests
import concurrent.futures as cf
import yfinance as yf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ----------- CONFIG -----------
DAYS_LOOKBACK   = 1        # look back N days on OpenInsider
MAX_RESULTS     = 5000
MIN_PRICE       = 3.0
MIN_DVOL        = 2_000_000   # 20D avg dollar volume
CAP_MIN         = 2_000_000_000
CEO_ONLY        = False       # True => only CEO; False => CEO or CFO
TOP_K           = 10

# Composite weights (sum ~ 1)
W_MOM_TREND   = 0.30  # 12–1 month momentum (higher better)
W_MOM_CONTRA  = 0.20  # -3M momentum (more negative better)
W_INSIDER_SZ  = 0.35  # log(Value) (bigger buys better)
W_CLUSTER     = 0.15  # # filings last 10 calendar days (more better)

# Email
FROM_EMAIL = os.environ.get("GMAIL_USER")
FROM_PASS  = os.environ.get("GMAIL_APP_PASSWORD")
TO_EMAIL   = os.environ.get("TO_EMAIL", FROM_EMAIL)
SUBJECT    = "Daily Insider Buy Signals (screened & scored)"

# ----------- HELPERS -----------
def fetch_openinsider(days=DAYS_LOOKBACK, maxresults=MAX_RESULTS, timeout=25):
    """Fetch latest insider purchases table from OpenInsider."""
    url = (
        "https://openinsider.com/screener?"
        "s=&o=&pl=&ph=&ll=&lh=&fd=0&fdr=&td=&tdr=&fdlyl=&fdlyh="
        f"&days={days}"
        "&xp=1&vl=100&vh=&ocl=0"
        "&sicMin=&sicMax=&sortcol=0&sortdir=0"
        f"&maxresults={maxresults}&group=filing"
        "&isOfficer=1&isCEO=1&isCFO=1"
    )
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status()
            tables = pd.read_html(io.StringIO(r.text), displayed_only=False)
            # Pick table with typical columns
            wanted = {"Filing Date","Trade Date","Ticker","Company Name","Insider Name","Title","Trade Type","Price","Qty","Value ($)"}
            best_i, best_score = None, -1
            for i,t in enumerate(tables):
                cols = set(map(str,t.columns))
                score = len(wanted & cols) + (t.shape[1]>=8)*2 + (t.shape[0]>=20)*1
                if score > best_score:
                    best_i, best_score = i, score
            if best_i is None:
                return pd.DataFrame()
            df = tables[best_i].copy()
            return df
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 + attempt)

def to_num(x):
    s = str(x).replace(",", "").replace("$","").replace("%","").replace("—","").replace("\u2212","-")
    s = s.replace("(", "-").replace(")","")
    return pd.to_numeric(s, errors="coerce")

def norm_cols(df):
    df = df.rename(columns={c:str(c).replace("\xa0"," ").strip() for c in df.columns})
    # Keep purchases only
    if "Trade Type" in df.columns:
        df = df[df["Trade Type"].astype(str).str.contains("P - Purchase", na=False)]
    rename = {
        "Ticker":"Ticker","Company Name":"Company","Insider Name":"Insider",
        "Title":"Title","Trade Type":"Transaction","Price":"Price",
        "Qty":"Shares","Value ($)":"Value","Trade Date":"Date","Filing Date":"Filing Date"
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    if "Ticker" in df.columns:  # remove repeated header rows
        df = df[df["Ticker"].astype(str) != "Ticker"]
    for col in ("Price","Shares","Value"):
        if col in df.columns:
            df[col] = df[col].map(to_num)
    for col in ("Date","Filing Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Yahoo symbol
    def to_yahoo(t):
        t = str(t).strip().upper()
        if not t or t == "NAN": return np.nan
        return t.replace(".", "-")
    df["Yahoo"] = df["Ticker"].map(to_yahoo)
    # Pick event date = Filing Date preferred, else Trade Date, else Date
    for name in ["Filing Date","Trade Date","Date"]:
        if name in df.columns:
            df["EventDate"] = pd.to_datetime(df[name], errors="coerce")
            break
    keep = ["Yahoo","Ticker","Company","Insider","Title","EventDate","Value","Price"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["Yahoo","EventDate"]).reset_index(drop=True)

def download_history(symbols, start, end=None):
    if not symbols: 
        return pd.DataFrame()
    data = yf.download(sorted(symbols), start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"], data["Volume"]
    # single ticker case
    sym = list(symbols)[0]
    return data[["Close"]].rename(columns={"Close":sym}), data[["Volume"]].rename(columns={"Volume":sym})

def compute_features(ins):
    # momentum & cluster need price rows around each date
    tickers = sorted(ins["Yahoo"].unique())
    start = (ins["EventDate"].min() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    close_all, vol_all = download_history(tickers, start=start)
    if close_all.empty: 
        return ins.assign(Score=np.nan)
    idx = close_all.index

    # cluster (rolling 10 calendar days by ticker)
    tmp = ins.sort_values(["Yahoo","EventDate"]).copy()
    tmp["_one"] = 1
    cl = (tmp.set_index("EventDate")
             .groupby("Yahoo")["_one"]
             .rolling("10D").sum()
             .reset_index(level=0, drop=True)
             .reindex(tmp.index))
    ins["cluster_10d"] = cl.values

    def price_at(y, dt, off=0):
        try:
            i = idx.searchsorted(pd.Timestamp(dt))
            j = i + off
            if j < 0 or j >= len(idx): return np.nan
            return float(close_all[y].iloc[j])
        except Exception:
            return np.nan

    def mom_from(y, dt, lb):
        p0 = price_at(y, dt, 0); pL = price_at(y, dt, -lb)
        if not np.isfinite(p0) or not np.isfinite(pL) or pL == 0: return np.nan
        return p0/pL - 1.0

    def mom_12_1(y, dt):
        p_1m = price_at(y, dt, -21); p_12m = price_at(y, dt, -252)
        if not np.isfinite(p_1m) or not np.isfinite(p_12m) or p_12m == 0: return np.nan
        return p_1m/p_12m - 1.0

    ins["mom_3m"]   = [mom_from(y, d, 63) for y,d in zip(ins["Yahoo"], ins["EventDate"])]
    ins["mom_12_1"] = [mom_12_1(y, d) for y,d in zip(ins["Yahoo"], ins["EventDate"])]

    # price_now & 20D dollar volume
    def px_now(y, d): return price_at(y, d, 0)
    ins["px_now"] = [px_now(y, d) for y,d in zip(ins["Yahoo"], ins["EventDate"])]
    def dvol20(y, d):
        try:
            i = idx.searchsorted(pd.Timestamp(d))
            lo, hi = max(0, i-19), i+1
            px = close_all[y].iloc[lo:hi].astype(float)
            vo = vol_all[y].iloc[lo:hi].astype(float)
            if px.empty or vo.empty: return np.nan
            return float((px*vo).mean())
        except Exception:
            return np.nan
    ins["dvol20"] = [dvol20(y, d) for y,d in zip(ins["Yahoo"], ins["EventDate"])]

    # market cap (few tickers → parallel info calls)
    def cap_one(t):
        try:
            return yf.Ticker(t).fast_info.market_cap or yf.Ticker(t).info.get("marketCap")
        except Exception:
            return np.nan
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        caps = list(ex.map(cap_one, ins["Yahoo"]))
    ins["marketCap"] = pd.to_numeric(caps, errors="coerce")

    # CEO/CFO filter
    title = ins.get("Title", "").astype(str)
    ins["is_CEO"] = title.str.contains(r"\bCEO\b", case=False, regex=True)
    ins["is_CFO"] = title.str.contains(r"\bCFO\b", case=False, regex=True)

    # HARD FILTERS
    role_ok  = ins["is_CEO"] if CEO_ONLY else (ins["is_CEO"] | ins["is_CFO"])
    price_ok = pd.to_numeric(ins["px_now"], errors="coerce") >= MIN_PRICE
    vol_ok   = pd.to_numeric(ins["dvol20"], errors="coerce") >= MIN_DVOL
    cap_ok   = pd.to_numeric(ins["marketCap"], errors="coerce") >= CAP_MIN

    eligible = (role_ok.fillna(False) & price_ok.fillna(False) &
                vol_ok.fillna(False)  & cap_ok.fillna(False))

    if eligible.sum() == 0:
        return ins.assign(Score=np.nan, eligible=False)

    # Features for scoring
    ins["f_insider_size"] = np.log1p(pd.to_numeric(ins.get("Value"), errors="coerce")).replace([np.inf,-np.inf], np.nan)
    ins["f_cluster"]      = ins["cluster_10d"].astype(float)
    ins["f_mom_trend"]    = ins["mom_12_1"].astype(float)    # higher better
    ins["f_mom_contra"]   = (-ins["mom_3m"]).astype(float)   # more negative 3m is better

    # Rank to [0,1] within the SAME EVENT DATE (to avoid look-ahead)
    def rank01(s):
        r = s.rank(pct=True, method="average")
        return r.fillna(0.5) if not r.isna().all() else pd.Series(0.5, index=s.index)

    for col in ["f_insider_size","f_cluster","f_mom_trend","f_mom_contra"]:
        ins[col+"_n"] = ins.groupby("EventDate")[col].transform(rank01)

    ins["Score"] = (
        W_INSIDER_SZ*ins["f_insider_size_n"] +
        W_CLUSTER   *ins["f_cluster_n"] +
        W_MOM_TREND *ins["f_mom_trend_n"] +
        W_MOM_CONTRA*ins["f_mom_contra_n"]
    )

    ins["eligible"] = eligible
    return ins

def email_table(df):
    if df.empty:
        return "<p>No signals passed today.</p>"
    show = df.copy()
    # nice columns
    keep = ["Ticker","EventDate","Title","Value","px_now","dvol20","marketCap","Score"]
    keep = [c for c in keep if c in show.columns]
    show = show[keep].sort_values("Score", ascending=False)
    # prettify
    def fmt_money(x):
        try:
            x = float(x)
            if not np.isfinite(x): return ""
            if abs(x) >= 1e9: return f"{x/1e9:.1f}B"
            if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
            return f"{x:,.0f}"
        except: return ""
    if "Value" in show.columns:    show["Value"]    = show["Value"].map(fmt_money)
    if "dvol20" in show.columns:   show["dvol20"]   = show["dvol20"].map(fmt_money)
    if "marketCap" in show.columns:show["marketCap"]= show["marketCap"].map(fmt_money)
    if "px_now" in show.columns:   show["px_now"]   = show["px_now"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    if "Score" in show.columns:    show["Score"]    = show["Score"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    return show.to_html(index=False, escape=False)

def send_email(html):
    assert FROM_EMAIL and FROM_PASS and TO_EMAIL, "Missing email env vars."
    msg = MIMEMultipart("alternative")
    msg["From"] = FROM_EMAIL
    msg["To"]   = TO_EMAIL
    msg["Subject"] = SUBJECT
    msg.attach(MIMEText(html, "html"))
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(FROM_EMAIL, FROM_PASS)
        server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())

def main():
    try:
        raw = fetch_openinsider()
        if raw is None or raw.empty:
            send_email("<p>OpenInsider fetch returned no data.</p>")
            return
        base = norm_cols(raw)
        if base.empty:
            send_email("<p>No purchase rows found today.</p>")
            return

        scored = compute_features(base)
        passed = scored[scored["eligible"]].copy()
        top = passed.sort_values("Score", ascending=False).head(TOP_K)

        html = f"""
        <h3>Daily Insider Buy Signals (screened & scored)</h3>
        <p>Criteria: Price ≥ ${MIN_PRICE}, 20D $Vol ≥ ${MIN_DVOL:,}, MarketCap ≥ ${CAP_MIN/1e9:.1f}B, Role: {'CEO' if CEO_ONLY else 'CEO/CFO'}</p>
        <p>Found {len(passed)} eligible out of {len(scored)}; showing top {min(TOP_K, len(passed))} by Score.</p>
        {email_table(top)}
        """
        send_email(html)
    except Exception as e:
        tb = traceback.format_exc()
        send_email(f"<h3>Insider bot error</h3><pre>{tb}</pre>")

if __name__ == "__main__":
    main()
