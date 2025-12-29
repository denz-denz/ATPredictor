import glob
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def load_matches(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "atp_matches_20*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No match files found in {data_dir}. Expected files like atp_matches_2019.csv"
        )
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)

    # required columns in Sackmann dataset
    required = ["winner_name", "loser_name", "surface", "tourney_date", "winner_rank", "loser_rank"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSVs: {missing}\nColumns found: {list(df.columns)}")

    df = df.dropna(subset=required).copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"]).copy()
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # some files might not have tourney_level; create placeholder
    if "tourney_level" not in df.columns:
        df["tourney_level"] = "U"

    return df


def make_examples(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _, r in df.iterrows():
        w = r["winner_name"]
        l = r["loser_name"]
        surf = r["surface"]
        level = r.get("tourney_level", "U")
        date = r["tourney_date"]

        w_rank = r["winner_rank"]
        l_rank = r["loser_rank"]

        # Randomize A/B to avoid "player A always wins" bias
        swap = rng.random() < 0.5
        if not swap:
            A_rank, B_rank = w_rank, l_rank
            y = 1
        else:
            A_rank, B_rank = l_rank, w_rank
            y = 0

        rows.append(
            {
                "date": date,
                "surface": surf,
                "tourney_level": level,
                "rank_diff": float(A_rank) - float(B_rank),
                "y": y,
            }
        )

    data = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return data


def time_split(data: pd.DataFrame):
    # Adjust these cutoffs however you like
    train = data[data["date"] < "2023-01-01"]
    valid = data[(data["date"] >= "2023-01-01") & (data["date"] < "2024-01-01")]
    test = data[data["date"] >= "2024-01-01"]
    return train, valid, test


def evaluate(clf, split: pd.DataFrame, name: str):
    X = split[["rank_diff", "surface", "tourney_level"]]
    y = split["y"].values
    p = clf.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "split": name,
        "n": len(split),
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, p),
        "auc": roc_auc_score(y, p),
    }


def main():
    df = load_matches("data")
    data = make_examples(df)

    train, valid, test = time_split(data)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["rank_diff"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["surface", "tourney_level"]),
        ]
    )

    clf = Pipeline(
        steps=[
            ("prep", preprocess),
            ("model", LogisticRegression(max_iter=300)),
        ]
    )

    clf.fit(train[["rank_diff", "surface", "tourney_level"]], train["y"])

    results = pd.DataFrame(
        [
            evaluate(clf, train, "train"),
            evaluate(clf, valid, "valid"),
            evaluate(clf, test, "test"),
        ]
    )

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()