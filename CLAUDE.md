# Tennis Predictor

Pre-match ATP predictor trained on ATP match CSVs from
[Tennismylife/TML-Database](https://github.com/Tennismylife/TML-Database)
(2018–2026, refreshed periodically). Given two player names and a tournament,
it predicts both the winner and how many sets the match takes (e.g. "Jannik
Sinner to win in 4 sets (37%)"), via a Streamlit UI (`app.py`).

## Repo layout

```
data/                       ATP match CSVs, one per year (atp_matches_2018.csv ... 2026.csv)
                             NOT tracked in git (see .gitignore) — TML-Database's terms
                             don't clearly permit redistribution, so this stays local-only.
                             50 cols incl. winner_name, loser_name, tourney_name, surface,
                             tourney_date (YYYYMMDD), tourney_level, best_of, score,
                             winner_rank, loser_rank. (Historically Sackmann-format-compatible;
                             TML adds an `indoor` column and ATP-ID-based winner_id/loser_id,
                             neither used here.)
models/
  predictor.joblib            Committed trained-model artifact (see below) — this, not
                               data/, is what the deployed app actually loads.
src/
  predictor.py                 TennisPredictor class — the real, current model code
  build_model.py                Trains from data/ and writes models/predictor.joblib
  train_baseline.py            Standalone training/eval script (train/valid/test split + metrics)
  run_predictor.py              Manual smoke-test entrypoint (hardcoded example match)
app.py                       Streamlit UI (dropdowns for Player A/B + tournament, Predict button)
requirements.txt             pandas, numpy, scikit-learn, streamlit, joblib
.venv/                       Local virtualenv (not tracked)
```

## How prediction works (src/predictor.py)

`TennisPredictor(data_dir="data")` (used by `build_model.py`/`train_baseline.py`/
`run_predictor.py` — requires local `data/`), on construction:
1. Loads every `data/atp_matches_20*.csv`, concatenates, parses `tourney_date`,
   sorts chronologically.
2. Builds a **rank index** per player (date → rank), a **form index** per
   player (date, win/loss, surface), a **tournament index**
   (`tourney_name -> {surface, tourney_level, best_of}`, from each
   tournament's most recent observed row), and a trimmed **activity table**
   (date, winner, loser only — backs `list_recent_players`, deliberately
   excludes match stats so the persisted artifact stays a "model", not a copy
   of the dataset).
3. Trains **two** `HistGradientBoostingClassifier` pipelines from one shared
   set of training examples (`_make_training_examples`, built by randomly
   swapping each historical winner/loser into "A"/"B" so the model doesn't
   learn "A always wins", with rank/form computed strictly from matches
   *before* the match date — no leakage):
   - **Win model** (`self.model`): features `rank_diff`, `form_diff`,
     `surface_form_diff` (all A − B), one-hot `surface`/`tourney_level`.
   - **Set-count model** (`self.sets_model`): a separate, symmetric model —
     does *not* condition on who wins, since match length is driven by how
     close the matchup is, not identity. Features are the *unsigned*
     (`abs()`) versions of the same diffs, plus `best_of`. Target is number
     of sets played, parsed from the `score` column via `_parse_num_sets`
     (excludes retirements/walkovers/`RET`/`W/O`/etc., since true length is
     unknowable for those).

Both models train on `date < 2023-01-01`; `train_baseline.py` also reports
`valid` (2023) / `test` (>= 2024) metrics.

`predict_match(playerA, playerB, surface, date, tourney_level="U", best_of=3)`
looks up both players' last known rank/form strictly before `date`, runs both
models, and combines them under an independence assumption:
`P(A wins in k sets) = P(A wins) * P(num_sets=k)`. Returns a `PredictionResult`
with `prob_A_wins`, `sets_probs` (dict of set-count → probability, renormalized
to the valid range for `best_of`), `predicted_sets`, and a human-readable
`summary`. Raises `ValueError` if either player has no match history before
that date (unranked/unknown player, or name doesn't exactly match the CSV
`winner_name`/`loser_name` strings).

`predict_match_by_tournament(playerA, playerB, tourney_name, date)` is a
convenience wrapper that looks up `surface`/`tourney_level`/`best_of` from the
tournament index — this is what `app.py` uses, so the UI only needs one
"Tournament" dropdown instead of three separate fields.

`list_recent_players(months=18)` and `list_tournaments()` back the UI dropdowns.

### Persistence: `save()` / `load()`

Training is not free (a few seconds for ~20k rows × 2 models) and, more
importantly, **the deployed app should never need the raw dataset** (see
README.md for why). So:
- `predictor.save(model_dir="models")` serializes the two fitted pipelines +
  the rank/form/tournament/activity lookup tables (not raw match rows) to
  `models/predictor.joblib` via `joblib`.
- `TennisPredictor.load(model_dir="models")` is a classmethod that
  reconstructs a working predictor from that artifact alone — bypasses
  `__init__`/`_load_matches` entirely, no `data/` access.
- `app.py` calls `TennisPredictor.load(...)`, wrapped in `@st.cache_resource`.
- Retraining is a local, manual step: run `python src/build_model.py`, then
  commit the updated `models/predictor.joblib`.

## Streamlit app (app.py)

`streamlit run app.py` — Player A / Player B / Tournament dropdowns (players
filtered to the last 18 months of activity), Predict button, shows the
winner/set-count summary, a win-probability bar, and a `st.bar_chart` of
`sets_probs`. Errors (e.g. insufficient history, same player twice) render via
`st.error` rather than crashing. Loads `models/predictor.joblib` only — works
without `data/` present.

## Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py                # run the UI (needs models/predictor.joblib)
python src/build_model.py           # retrain from data/, write models/predictor.joblib
python src/run_predictor.py         # quick manual prediction smoke test (needs data/)
python src/train_baseline.py        # train + print eval metrics for both models (needs data/)
```

No test suite, linter, or CI config currently exists.
