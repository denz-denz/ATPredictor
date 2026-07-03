"""
One-off experiment: does adding Elo ratings (overall + surface) and trailing
serve/return stats improve on the current rank/form-only feature set?

Builds ONE shared training frame (so baseline and augmented models see
identical rows/splits/labels), trains a baseline and an augmented model for
both the win and set-count targets, and prints held-out metrics side by side.
Nothing here touches predictor.py / train_baseline.py — only kept if it wins.
"""
import glob
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from predictor import _parse_num_sets

SEED = 42
N_FORM = 20
K_ELO = 32
STARTING_ELO = 1500.0


def load_matches(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "atp_matches_20*.csv")))
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)

    required = ["winner_name", "loser_name", "surface", "tourney_date", "winner_rank", "loser_rank"]
    df = df.dropna(subset=required).copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"]).copy()
    df = df.sort_values("tourney_date").reset_index(drop=True)

    if "tourney_level" not in df.columns:
        df["tourney_level"] = "U"
    if "score" not in df.columns:
        df["score"] = None
    if "best_of" not in df.columns:
        df["best_of"] = 3
    df["best_of"] = df["best_of"].fillna(3)
    return df


def safe_rate(numer, denom):
    if pd.isna(numer) or pd.isna(denom) or denom <= 0:
        return None
    return float(numer) / float(denom)


def global_mean_rate(df, numer_col, denom_col):
    numer = pd.concat([df[f"w_{numer_col}"], df[f"l_{numer_col}"]])
    denom = pd.concat([df[f"w_{denom_col}"], df[f"l_{denom_col}"]])
    valid = denom > 0
    return float((numer[valid] / denom[valid]).mean())


def build_examples(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    overall_hist = defaultdict(lambda: deque(maxlen=N_FORM))
    surface_hist = defaultdict(lambda: defaultdict(lambda: deque(maxlen=N_FORM)))
    elo = defaultdict(lambda: STARTING_ELO)
    surface_elo = defaultdict(lambda: defaultdict(lambda: STARTING_ELO))

    ace_hist = defaultdict(lambda: deque(maxlen=N_FORM))
    fsw_hist = defaultdict(lambda: deque(maxlen=N_FORM))   # first-serve-won rate
    bps_hist = defaultdict(lambda: deque(maxlen=N_FORM))   # break-point-saved rate

    fallback_ace = global_mean_rate(df, "ace", "svpt")
    fallback_fsw = global_mean_rate(df, "1stWon", "1stIn")
    fallback_bps = global_mean_rate(df, "bpSaved", "bpFaced")

    rows = []
    for _, r in df.iterrows():
        date = r["tourney_date"]
        surf = r["surface"]
        level = r.get("tourney_level", "U")
        w, l = r["winner_name"], r["loser_name"]
        w_rank, l_rank = float(r["winner_rank"]), float(r["loser_rank"])
        best_of = int(r.get("best_of", 3))

        def winrates(player):
            o, s = overall_hist[player], surface_hist[player][surf]
            wr_o = (sum(o) / len(o)) if len(o) else 0.5
            wr_s = (sum(s) / len(s)) if len(s) else 0.5
            return wr_o, wr_s

        def rate_feat(hist, fallback):
            return (sum(hist) / len(hist)) if len(hist) else fallback

        w_wr_o, w_wr_s = winrates(w)
        l_wr_o, l_wr_s = winrates(l)
        w_elo, l_elo = elo[w], elo[l]
        w_selo, l_selo = surface_elo[surf][w], surface_elo[surf][l]
        w_ace_r, l_ace_r = rate_feat(ace_hist[w], fallback_ace), rate_feat(ace_hist[l], fallback_ace)
        w_fsw_r, l_fsw_r = rate_feat(fsw_hist[w], fallback_fsw), rate_feat(fsw_hist[l], fallback_fsw)
        w_bps_r, l_bps_r = rate_feat(bps_hist[w], fallback_bps), rate_feat(bps_hist[l], fallback_bps)

        swap = rng.random() < 0.5
        if not swap:
            A_rank, B_rank, A_wr_o, B_wr_o, A_wr_s, B_wr_s = w_rank, l_rank, w_wr_o, l_wr_o, w_wr_s, l_wr_s
            A_elo, B_elo, A_selo, B_selo = w_elo, l_elo, w_selo, l_selo
            A_ace, B_ace, A_fsw, B_fsw, A_bps, B_bps = w_ace_r, l_ace_r, w_fsw_r, l_fsw_r, w_bps_r, l_bps_r
            y = 1
        else:
            A_rank, B_rank, A_wr_o, B_wr_o, A_wr_s, B_wr_s = l_rank, w_rank, l_wr_o, w_wr_o, l_wr_s, w_wr_s
            A_elo, B_elo, A_selo, B_selo = l_elo, w_elo, l_selo, w_selo
            A_ace, B_ace, A_fsw, B_fsw, A_bps, B_bps = l_ace_r, w_ace_r, l_fsw_r, w_fsw_r, l_bps_r, w_bps_r
            y = 0

        best_of_val = best_of
        rows.append({
            "date": date, "surface": surf, "tourney_level": level, "best_of": best_of_val,
            "rank_diff": A_rank - B_rank,
            "form_diff": A_wr_o - B_wr_o,
            "surface_form_diff": A_wr_s - B_wr_s,
            "elo_diff": A_elo - B_elo,
            "surface_elo_diff": A_selo - B_selo,
            "ace_rate_diff": A_ace - B_ace,
            "first_serve_win_rate_diff": A_fsw - B_fsw,
            "bp_save_rate_diff": A_bps - B_bps,
            "y": y,
            "num_sets": _parse_num_sets(r.get("score"), best_of_val),
        })

        # ---- update all trailing state AFTER using it as a feature ----
        overall_hist[w].append(1); overall_hist[l].append(0)
        surface_hist[w][surf].append(1); surface_hist[l][surf].append(0)

        exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_elo) / 400.0))
        elo[w] = w_elo + K_ELO * (1 - exp_w)
        elo[l] = l_elo - K_ELO * (1 - exp_w)
        exp_w_s = 1.0 / (1.0 + 10 ** ((l_selo - w_selo) / 400.0))
        surface_elo[surf][w] = w_selo + K_ELO * (1 - exp_w_s)
        surface_elo[surf][l] = l_selo - K_ELO * (1 - exp_w_s)

        w_ace_rate = safe_rate(r.get("w_ace"), r.get("w_svpt"))
        l_ace_rate = safe_rate(r.get("l_ace"), r.get("l_svpt"))
        if w_ace_rate is not None: ace_hist[w].append(w_ace_rate)
        if l_ace_rate is not None: ace_hist[l].append(l_ace_rate)

        w_fsw_rate = safe_rate(r.get("w_1stWon"), r.get("w_1stIn"))
        l_fsw_rate = safe_rate(r.get("l_1stWon"), r.get("l_1stIn"))
        if w_fsw_rate is not None: fsw_hist[w].append(w_fsw_rate)
        if l_fsw_rate is not None: fsw_hist[l].append(l_fsw_rate)

        w_bps_rate = safe_rate(r.get("w_bpSaved"), r.get("w_bpFaced"))
        l_bps_rate = safe_rate(r.get("l_bpSaved"), r.get("l_bpFaced"))
        if w_bps_rate is not None: bps_hist[w].append(w_bps_rate)
        if l_bps_rate is not None: bps_hist[l].append(l_bps_rate)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def time_split(data):
    train = data[data["date"] < "2023-01-01"]
    valid = data[(data["date"] >= "2023-01-01") & (data["date"] < "2024-01-01")]
    test = data[data["date"] >= "2024-01-01"]
    return train, valid, test


def make_pipeline(num_cols):
    preprocess = ColumnTransformer(transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["surface", "tourney_level"]),
    ])
    return Pipeline(steps=[
        ("prep", preprocess),
        ("model", HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=SEED)),
    ])


def eval_win(clf, num_cols, split, name):
    X = split[num_cols + ["surface", "tourney_level"]]
    y = split["y"].values
    p = clf.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {"split": name, "n": len(split), "accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, p), "auc": roc_auc_score(y, p)}


def eval_sets(clf, num_cols, split, name):
    split = split[split["num_sets"].notna()]
    X = split[num_cols + ["surface", "tourney_level"]]
    y = split["num_sets"].astype(int).values
    p = clf.predict_proba(X)
    pred = clf.classes_[p.argmax(axis=1)]
    return {"split": name, "n": len(split), "accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, p, labels=clf.classes_)}


def run_comparison(label, win_cols, sets_cols_abs, data, train, valid, test):
    print(f"\n=== {label} ===")
    win_clf = make_pipeline(win_cols)
    win_clf.fit(train[win_cols + ["surface", "tourney_level"]], train["y"])
    win_results = pd.DataFrame([
        eval_win(win_clf, win_cols, train, "train"),
        eval_win(win_clf, win_cols, valid, "valid"),
        eval_win(win_clf, win_cols, test, "test"),
    ])
    print("Win model:")
    print(win_results.to_string(index=False))

    sets_train = train[train["num_sets"].notna()]
    sets_clf = make_pipeline(sets_cols_abs)
    sets_clf.fit(sets_train[sets_cols_abs + ["surface", "tourney_level"]], sets_train["num_sets"].astype(int))
    sets_results = pd.DataFrame([
        eval_sets(sets_clf, sets_cols_abs, train, "train"),
        eval_sets(sets_clf, sets_cols_abs, valid, "valid"),
        eval_sets(sets_clf, sets_cols_abs, test, "test"),
    ])
    print("Sets model:")
    print(sets_results.to_string(index=False))
    return win_results, sets_results


def main():
    df = load_matches("data")
    data = build_examples(df)
    data["abs_rank_diff"] = data["rank_diff"].abs()
    data["abs_form_diff"] = data["form_diff"].abs()
    data["abs_surface_form_diff"] = data["surface_form_diff"].abs()
    data["abs_elo_diff"] = data["elo_diff"].abs()
    data["abs_surface_elo_diff"] = data["surface_elo_diff"].abs()
    data["abs_ace_rate_diff"] = data["ace_rate_diff"].abs()
    data["abs_first_serve_win_rate_diff"] = data["first_serve_win_rate_diff"].abs()
    data["abs_bp_save_rate_diff"] = data["bp_save_rate_diff"].abs()

    train, valid, test = time_split(data)

    baseline_win_cols = ["rank_diff", "form_diff", "surface_form_diff"]
    baseline_sets_cols = ["abs_rank_diff", "abs_form_diff", "abs_surface_form_diff", "best_of"]

    augmented_win_cols = baseline_win_cols + [
        "elo_diff", "surface_elo_diff", "ace_rate_diff", "first_serve_win_rate_diff", "bp_save_rate_diff",
    ]
    augmented_sets_cols = baseline_sets_cols + [
        "abs_elo_diff", "abs_surface_elo_diff", "abs_ace_rate_diff",
        "abs_first_serve_win_rate_diff", "abs_bp_save_rate_diff",
    ]

    run_comparison("BASELINE (current production features)", baseline_win_cols, baseline_sets_cols, data, train, valid, test)
    run_comparison("AUGMENTED (+ Elo, + trailing serve/return stats)", augmented_win_cols, augmented_sets_cols, data, train, valid, test)


if __name__ == "__main__":
    main()
