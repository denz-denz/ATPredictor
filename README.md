# ATP Match Predictor

Predicts pre-match win probability and expected set count for ATP tennis
matches (e.g. "Jannik Sinner to win in 4 sets (37%)").

## Data

Match history comes from [Tennismylife/TML-Database](https://github.com/Tennismylife/TML-Database)
(itself derived from Jeff Sackmann's `tennis_atp` work). TML's terms don't
clearly permit redistributing the raw database, so **the raw CSVs are not
committed to this repo** — `data/` is gitignored and local-only.

Instead, the repo commits **trained model artifacts** (`models/predictor.joblib`):
the fitted win-probability and set-count models plus compact per-player
rank/form/activity lookup tables derived from the data — not the underlying
match-by-match records. The deployed app loads only this artifact and never
touches the raw dataset.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # loads the committed models/predictor.joblib
```

## Retrain

Requires the raw data in `data/atp_matches_YYYY.csv` (fetch from TML-Database
yourself, matching that naming convention):

```bash
python src/build_model.py     # trains from data/, writes models/predictor.joblib
python src/train_baseline.py  # prints train/valid/test eval metrics
```

Commit the updated `models/predictor.joblib` after retraining; `data/` stays
untracked.

See `CLAUDE.md` for architecture details.
