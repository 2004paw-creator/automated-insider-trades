# Insider-purchase notifier

An automated pipeline that watches SEC Form 4 filings, isolates open-market
purchases by senior officers, ranks them with a small cross-sectional factor
model, and emails a daily digest.

Corporate insiders sell for many reasons — diversification, liquidity,
scheduled 10b5-1 plans — but they buy on the open market for essentially one:
they expect the stock to rise. Open-market purchases (Form 4 transaction code
`P`) are therefore the subset of insider activity with the most signal, and
purchases by the CEO or CFO carry the most, since those officers have the
broadest view of the firm. This tool surfaces that subset and ranks it.

## What it does

Each run:

1. Pulls the last seven days of Form 4 filings from the SEC EDGAR daily master
   index, and parses each filing's ownership XML.
2. Keeps only open-market purchases (code `P`), recording the transaction date,
   filing date, insider, title, price, shares, and dollar value.
3. Joins daily prices from Yahoo Finance and computes four factors per purchase.
4. Ranks purchases by a weighted factor score and applies liquidity/role
   filters.
5. Emails two tables — "today" and "rolling seven days" — each showing the
   filtered matches and the full ranking.

If the daily master index is unavailable, it falls back to parsing the SEC
"current filings" HTML for the current day.

## The ranking model

For each purchase, four factors are computed, **rank-normalised within the
filing date** (so a score reflects how a purchase compares to its same-day
peers, not an absolute level), and combined as a weighted sum:

| Factor | Definition | Rationale | Weight |
|---|---|---|---|
| `insider_size` | log dollar value of the purchase | larger buys signal stronger conviction | 0.35 |
| `cluster` | purchases in a trailing 10 days | independent buyers corroborate the signal | 0.15 |
| `mom_trend` | 12-1 month momentum | insiders buying into an established up-trend | 0.30 |
| `mom_contra` | negative 3-month return | insiders buying a recent sell-off | 0.20 |

The trend and contrarian factors deliberately pull in opposite directions: the
score rewards names that are either in a longer-run up-trend *or* have recently
sold off, the two regimes in which insider purchases have historically been
most informative. The weights are a starting point, not an estimated model —
they live in `config.py` and are trivial to change or sweep.

Filters (also in `config.py`) then remove noise: a minimum purchase value, a
minimum share price, a minimum 20-day average dollar volume, and (optionally) a
requirement that the insider be CEO or CFO.

## Live dashboard

The daily run also maintains three JSON files under `signals/` — this week's
filter-passing signals, a forward-filling paper book (first 50 qualifying
signals per calendar year, entered at the first close after the filing date),
and its performance against SPY over matched windows. The files are committed
by the Action, so the repo doubles as the (versioned) database, and a page on
my site renders them live: the book only ever fills forward, so there is no
hindsight selection in the track record.

## Layout

```
config.py    constants, secrets, logging, and the ScoringConfig dataclass
edgar.py     SEC fetching + Form 4 XML parsing (rate-limited, retrying)
scoring.py   price download, factor construction, and the ranking model
notify.py    email composition and Gmail SMTP delivery
main.py      orchestration (entry point)
```

## Running it

```bash
pip install -r requirements.txt

export GMAIL_USER="you@gmail.com"
export GMAIL_APP_PASSWORD="your-16-char-app-password"   # not your login password
export TO_EMAIL="where@to-send.com"

python main.py
```

`GMAIL_APP_PASSWORD` is a Google [app password](https://support.google.com/accounts/answer/185833),
not your account password. Runs on a schedule via GitHub Actions (see
`.github/workflows/`).

## Limitations

This is a research tool, and its results should be read with the usual caveats:

- **Not investment advice.** Insider-purchase signals are noisy; this ranks
  filings, it does not predict returns.
- **Data quality.** Yahoo Finance prices are convenient but unofficial;
  survivorship and occasional bad ticks are possible. Delisted or renamed
  tickers simply drop out.
- **Point-in-time care.** Factors are computed as of the transaction date. The
  digest is informational, but any backtest built on this data would need to
  join prices strictly as-of the *filing* date to avoid look-ahead, since a
  purchase is not public until it is filed.
- **Weights are unfitted.** The factor weights are chosen, not estimated. They
  are a reasonable prior, not a claim of optimality.
