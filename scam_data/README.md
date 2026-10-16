# scam_data/ — constructed scam/benign phrase set (CC0, hand-written)

`phrases.csv` columns: text, label (scam/benign), language (en/hi/ta),
notes. Every row was hand-written FOR this detector with known intent —
ground truth by construction, NOT field recordings. Treat eval numbers as a
rules/classifier sanity check on constructed data, never as deployment
metrics.

- scam rows: OTP/PIN demands, SIM/KYC threats, arrest threats, prize lures,
  remote-access demands, money-transfer demands (EN + Hindi-English + Tamil).
- benign rows: family chat, food delivery, appointment reminders, AND tricky
  cases: fraud-awareness advisories, first-person OTP accounts, legitimate
  mentions of banks/OTP without demands.

Train: `python tools/train_scam_text.py` → `models/scam_text.joblib`
(git-ignored weights) + `reports/scam_text_eval.md` (committed metrics).
