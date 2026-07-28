"""
notify.py
Compose and send the digest email over Gmail SMTP.
"""
from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

import pandas as pd

from config import (GMAIL_APP_PASS, GMAIL_USER, TO_EMAIL, YF_LINK_FMT, get_logger)

log = get_logger("insider.notify")

_PREFERRED = [
    "Yahoo", "Filed", "Date", "Company", "Insider", "Title", "Price", "Shares",
    "Value", "cluster_10d", "mom_12_1", "mom_3m", "px_now", "dvol20", "Score",
]


def send_email(subject: str, html: str) -> None:
    if not (GMAIL_USER and GMAIL_APP_PASS and TO_EMAIL):
        raise RuntimeError("Missing GMAIL_USER, GMAIL_APP_PASSWORD or TO_EMAIL env vars.")
    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = TO_EMAIL
    msg["Date"] = formatdate(localtime=True)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_USER, GMAIL_APP_PASS)
        smtp.sendmail(GMAIL_USER, [TO_EMAIL], msg.as_string())
    log.info("email sent: %s", subject)


def df_to_html(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h4>{title}</h4><p>No rows.</p>"
    d = df.copy()
    if "Yahoo" in d.columns:
        d["Yahoo"] = d["Yahoo"].apply(lambda s: f'<a href="{YF_LINK_FMT.format(sym=s)}">{s}</a>')
    for col in ("Price", "px_now"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").map(
                lambda v: f"${v:,.2f}" if pd.notna(v) else "")
    if "Shares" in d.columns:
        d["Shares"] = pd.to_numeric(d["Shares"], errors="coerce").map(
            lambda v: f"{int(v):,}" if pd.notna(v) else "")
    for col in ("Value", "dvol20"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").map(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "")
    if "Score" in d.columns:
        d["Score"] = pd.to_numeric(d["Score"], errors="coerce").map(
            lambda v: f"{v:.3f}" if pd.notna(v) else "")
    for col in ("Date", "Filed"):
        if col in d.columns:
            d[col] = pd.to_datetime(d[col], errors="coerce").dt.strftime("%Y-%m-%d")
    cols = [c for c in _PREFERRED if c in d.columns]
    return f"<h4>{title}</h4>" + d[cols].to_html(index=False, escape=False)


def send_heartbeat(note: str) -> None:
    try:
        send_email("[bot heartbeat] notifier ran", f"<p>{note}</p>")
    except Exception as e:
        log.error("heartbeat email failed: %s", e)
