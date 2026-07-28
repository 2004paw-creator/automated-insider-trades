"""
main.py
Orchestration: fetch a week of Form 4 purchases once, build a "today" view
and a "this week" view from the same data, score both, and email the digest.

Run:  python main.py
"""
from __future__ import annotations

import traceback
from datetime import date

import pandas as pd

from config import CFG, SUBJECT_PREFIX, get_logger
from edgar import _dedup_latest, _empty, fetch_sec_week_raw
from notify import df_to_html, send_email
from scoring import add_features_and_score

log = get_logger("insider.main")


def run() -> None:
    # Fetch the whole window once. Today is included here, so it is pulled a
    # single time per run rather than separately for the daily and weekly views.
    raw_week_all = fetch_sec_week_raw(CFG.lookback_week)

    today_ts = pd.Timestamp(date.today()).normalize()
    raw_today = (
        _dedup_latest(raw_week_all[raw_week_all["Filed"] == today_ts])
        if not raw_week_all.empty else _empty()
    )
    raw_week = _dedup_latest(raw_week_all)

    matches_today, forced_today = add_features_and_score(raw_today)
    matches_week, forced_week = add_features_and_score(raw_week)

    n = lambda x: 0 if x is None else len(x)
    subject = (
        f"{SUBJECT_PREFIX}: today {n(matches_today)}/{n(forced_today)} | "
        f"week {n(matches_week)}/{n(forced_week)}"
    )

    html = f"""
    <h3>{SUBJECT_PREFIX}</h3>
    <p>Filters: CEO/CFO={CFG.require_ceo_cfo} | Min Value=${CFG.min_value:,.0f} |
       Min Price=${CFG.min_price} | Min $Vol(20D)=${CFG.min_dvol:,.0f}</p>

    {df_to_html(matches_today, "Today — Matches (pass filters + score)")}
    {df_to_html(forced_today,  "Today — All-ranked")}

    {df_to_html(matches_week,  "This Week (rolling 7 days) — Matches")}
    {df_to_html(forced_week,   "This Week (rolling 7 days) — All-ranked")}

    <p style="color:#999">Sources: SEC EDGAR daily master (XML) with fallback to
    SEC current filings HTML. "Filed" is the filing date; "Date" is the
    transaction date.</p>
    """
    send_email(subject, html)


def main() -> None:
    try:
        run()
    except Exception:
        tb = traceback.format_exc()
        log.error("run failed:\n%s", tb)
        try:
            send_email("Insider bot error", f"<pre>{tb}</pre>")
        except Exception as e:
            log.error("could not send error email: %s", e)


if __name__ == "__main__":
    main()
