# notifier.py
# Daily insider buy notifier with OpenInsider + SEC fallback and Gmail SMTP email
# Paste this whole file into your repo. Requires: requests, pandas, lxml (or html5lib)

import os
import re
import smtplib
import traceback
from email.mime.text import MIMEText
from email.utils import formatdate

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# ------- CONFIG ----------
# =========================
DAYS_LOOKBACK   = 1          # look back N days for OpenInsider
MIN_VALUE       = 100_000    # min $ value of the purchase
REQUIRE_CEO_CFO = True       # only CEO/CFO roles
MAX_RESULTS     = 30         # max rows to email
SUBJECT_PREFIX  = "Insider buys"
YF_LINK_FMT     = "https://finance.yahoo.com/quote/{sym}"

# Gmail secrets are read from env
GMAIL_USER       = os.getenv("GMAIL_USER")
GMAIL_APP_PASS   = os.getenv("GMAIL_APP_PASSWORD")
TO_EMAIL         = os.getenv("TO_EMAIL")

# SEC wants a meaningful UA with contact info
SEC_UA = "Mozilla/5.0 (compatible; InsiderBot/1.0; +mailto:{})".format(GMAIL_USER or "you@example.com")


# =========================
# ---- HTTP / FETCHERS ----
# =========================
def _session_with_retries(total=5, backoff=0.6):
    s = requests.Session()
    r = Retry(
        total=total,
        connect=total,
        read=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


def fetch_openinsider(days=DAYS_LOOKBACK, timeout=20):
    """Scrape OpenInsider screener (purchases, officers, CEO/CFO toggled)."""
    url = (
        "https://openinsider.com/screener"
        "?s=&o=&pl=&ph=&ll=&lh=&fd=0&fdr=&td=&tdr=&fdlyl=&fdlyh="
        f"&days={days}&xp=1&vl=100&vh=&ocl=0&sicMin=&sicMax=&sortcol=0"
        "&sortdir=0&maxresults=5000&group=filing&isOfficer=1&isCEO=1&isCFO=1"
    )
    sess = _session_with_retries()
    r = sess.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    tables = pd.read_html(r.text, displayed_only=False)
    # pick the most "grid-like" table
    t = max(tables, key=lambda df: (df.shape[1], df.shape[0]))
    t.columns = [str(c).strip().replace("\xa0", " ") for c in t.columns]

    # purchase rows only
    if "Trade Type" in t.columns:
        t = t[t["Trade Type"].astype(str).str.contains("P - Purchase", na=False)]

    # normalize columns
    rename = {
        "Ticker": "Ticker", "Company Name": "Company", "Insider Name": "Insider",
        "Title": "Title", "Trade Type": "Transaction", "Price": "Price",
        "Qty": "Shares", "Value ($)": "Value", "Trade Date": "Date", "Filing Date": "Filing Date",
    }
    t = t.rename(columns={k: v for k, v in rename.items() if k in t.columns})
    if "Ticker" in t.columns:
        t = t[t["Ticker"].astype(str) != "Ticker"]  # drop header rows inside body

    # numeric/date cleanup
    def to_num(x):
        s = str(x).replace(",", "").replace("$", "").replace("%", "").replace("—", "").replace("\u2212", "-")
        s = s.replace("(", "-").replace(")", "")
        return pd.to_numeric(s, errors="coerce")

    for c in ("Price", "Shares", "Value"):
        if c in t.columns:
            t[c] = t[c].map(to_num)
    for c in ("Date", "Filing Date"):
        if c in t.columns:
            t[c] = pd.to_datetime(t[c], errors="coerce")

    return t


def fetch_sec_current(timeout=30):
    """
    Official SEC fallback: 'current insider transactions' page.
    We filter to purchase transactions and try to keep comparable fields.
    """
    url = "https://www.sec.gov/cgi-bin/own-disp?action=getcurrent"
    sess = _session_with_retries()
    r = sess.get(url, timeout=timeout, headers={"User-Agent": SEC_UA})
    r.raise_for_status()

    tables = pd.read_html(r.text, displayed_only=False)
    t = max(tables, key=lambda df: (df.shape[1], df.shape[0]))
    t.columns = [str(c).strip().replace("\xa0", " ") for c in t.columns]

    # Map into our schema where possible
    t = t.rename(columns={
        "Ticker": "Ticker",
        "Transaction Date": "Date",
        "Price": "Price",
        "Amount": "Shares",
        "Relationship": "Title",
        "Company": "Company",
        "Owner": "Insider",
        "Transaction": "Transaction",
    })

    # Purchase code is 'P'
    if "Transaction Code" in t.columns:
        t = t[t["Transaction Code"].astype(str).str.fullmatch(r"P", na=False)]

    # Roles (CEO/CFO/Pres etc.)
    if "Title" in t.columns and REQUIRE_CEO_CFO:
        t["Title"] = t["Title"].astype(str)
        role_mask = t["Title"].str.contains(r"\b(CEO|CFO|Chief|President|Pres\.?)\b", flags=re.I, regex=True)
        t = t[role_mask]

    # numeric/date cleanup
    def to_num(x):
        s = str(x).replace(",", "").replace("$", "").replace("%", "").replace("—", "").replace("\u2212", "-")
        s = s.replace("(", "-").replace(")", "")
        return pd.to_numeric(s, errors="coerce")

    for c in ("Price", "Shares"):
        if c in t.columns:
            t[c] = t[c].map(to_num)
    if "Date" in t.columns:
        t["Date"] = pd.to_datetime(t["Date"], errors="coerce")

    if {"Price", "Shares"}.issubset(t.columns):
        t["Value"] = t["Price"] * t["Shares"]

    keep = [c for c in ["Ticker", "Company", "Insider", "Title", "Transaction",
                        "Price", "Shares", "Value", "Date", "Filing Date"] if c in t.columns]
    out = t[keep].dropna(subset=["Ticker", "Date"]).reset_index(drop=True)
    return out


def fetch_insider_rows():
    """Try OpenInsider first; on error, fall back to SEC."""
    try:
        return fetch_openinsider(days=DAYS_LOOKBACK)
    except Exception as e:
        print("OpenInsider unreachable; using SEC fallback →", e)
        return fetch_sec_current()


# =========================
# ----- POST-PROCESS ------
# =========================
def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean tickers, enforce filters, rank by $Value."""
    if df.empty:
        return df

    # Ticker → Yahoo symbol format (BRK.B -> BRK-B)
    def to_yahoo(t):
        t = str(t).strip().upper()
        if not t or t == "NAN":
            return np.nan
        t = t.replace(".", "-")
        t = re.sub(r"[^A-Z0-9\-\_\.]", "", t)
        return t

    df = df.copy()
    if "Ticker" not in df.columns:
        return pd.DataFrame()

    df["Yahoo"] = df["Ticker"].map(to_yahoo)
    df = df.dropna(subset=["Yahoo"])

    # Default Event date: Filing Date if present, else Date
    if "Filing Date" in df.columns:
        df["EventDate"] = pd.to_datetime(df["Filing Date"], errors="coerce")
    else:
        df["EventDate"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df = df.dropna(subset=["EventDate"])

    # Ensure Value numeric
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # CEO/CFO role filter (when requested)
    if REQUIRE_CEO_CFO and "Title" in df.columns:
        role_mask = df["Title"].astype(str).str.contains(r"\b(CEO|CFO)\b", case=False, regex=True)
        df = df[role_mask]

    # Min $ filter
    if "Value" in df.columns:
        df = df[df["Value"] >= MIN_VALUE]

    # Keep one-most-recent per ticker
    df = (df.sort_values("EventDate")
            .groupby("Yahoo", as_index=False)
            .tail(1))

    # Simple score (percentile of log1p(Value)) so email is ranked
    if "Value" in df.columns:
        val = np.log1p(pd.to_numeric(df["Value"], errors="coerce"))
        pct = val.rank(pct=True)
        df["Score"] = pct.fillna(0.5).round(3)

    # Order for email
    preferred_cols = [
        "Yahoo", "EventDate", "Company", "Insider", "Title",
        "Price", "Shares", "Value", "Score"
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    df = df[cols].sort_values(["Score", "Value", "EventDate"], ascending=[False, False, False])
    return df.reset_index(drop=True)


# =========================
# -------- EMAIL ----------
# =========================
def send_email(subject: str, body_text: str, body_html: str = None):
    assert GMAIL_USER and GMAIL_APP_PASS and TO_EMAIL, \
        "Missing GMAIL_USER, GMAIL_APP_PASSWORD or TO_EMAIL env vars."

    msg = MIMEText(body_html or body_text, "html" if body_html else "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = TO_EMAIL
    msg["Date"]    = formatdate(localtime=True)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_USER, GMAIL_APP_PASS)
        smtp.sendmail(GMAIL_USER, [TO_EMAIL], msg.as_string())


def df_to_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No qualifying trades today.</p>"
    df = df.copy()
    # Add Yahoo links
    if "Yahoo" in df.columns:
        df["Yahoo"] = df["Yahoo"].apply(lambda s: f'<a href="{YF_LINK_FMT.format(sym=s)}">{s}</a>')
    fmt = {
        "EventDate": lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M"),
        "Price":     lambda x: f"${x:,.2f}",
        "Shares":    lambda x: f"{int(x):,}" if pd.notna(x) else "",
        "Value":     lambda x: f"${x:,.0f}" if pd.notna(x) else "",
        "Score":     lambda x: f"{x:.3f}" if pd.notna(x) else "",
    }
    for c, f in fmt.items():
        if c in df.columns:
            df[c] = df[c].apply(lambda v: f(v) if pd.notna(v) else "")
    return df.to_html(index=False, escape=False)


# =========================
# --------- MAIN ----------
# =========================
def main():
    try:
        raw = fetch_insider_rows()  # robust fetcher with SEC fallback
        clean = standardize(raw)
        top = clean.head(MAX_RESULTS)

        source = "OpenInsider" if "Filing Date" in raw.columns else "SEC"
        subject = f"{SUBJECT_PREFIX}: {len(top)} signals (src: {source})"

        html = f"""
        <h3>{SUBJECT_PREFIX} — {len(top)} qualifying trades</h3>
        <p>Source: <b>{source}</b> | Lookback: {DAYS_LOOKBACK} day(s) | Min value: ${MIN_VALUE:,} | CEO/CFO only: {REQUIRE_CEO_CFO}</p>
        {df_to_html_table(top)}
        <p style="color:#999">Auto-generated from public data. This is not investment advice.</p>
        """

        send_email(subject, body_text="HTML required", body_html=html)

    except Exception as e:
        # On error, email the traceback so you see it in your inbox
        tb = traceback.format_exc()
        subject = f"Insider bot error"
        body = f"<pre>{tb}</pre>"
        try:
            send_email(subject, body_text=tb, body_html=body)
        except Exception:
            # If even email fails, print to logs
            print("Fatal error & email send failed:\n", tb)


if __name__ == "__main__":
    main()
