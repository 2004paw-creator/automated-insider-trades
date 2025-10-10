# test_mail.py — minimal Gmail SMTP test using your GH secrets
import os, smtplib
from email.mime.text import MIMEText

def _clean(x): return (x or "").strip().replace("\r","").replace("\n","")
GMAIL_USER     = _clean(os.getenv("GMAIL_USER"))
GMAIL_APP_PASS = _clean(os.getenv("GMAIL_APP_PASSWORD"))
TO_EMAIL       = _clean(os.getenv("TO_EMAIL"))

assert GMAIL_USER and GMAIL_APP_PASS and TO_EMAIL, "Missing Gmail env vars."

msg = MIMEText("Hello from test_mail.py — if you see this, SMTP is good.")
msg["Subject"] = "SMTP sanity check"
msg["From"]    = GMAIL_USER
msg["To"]      = TO_EMAIL

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(GMAIL_USER, GMAIL_APP_PASS)
    smtp.sendmail(GMAIL_USER, [TO_EMAIL], msg.as_string())

print("✅ Sent test email.")
