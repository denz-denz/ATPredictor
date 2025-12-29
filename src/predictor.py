import glob
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, Union, Dict, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

N_FORM = 20 # last 20 matches for recent form


DateLike = Union[str, pd.Timestamp]


def _to_timestamp(d: DateLike) -> pd.Timestamp:
    return pd.to_datetime(d)


@dataclass
class PredictionResult:
    playerA: str
    playerB: str
    date: pd.Timestamp
    surface: str
    tourney_level: str
    rankA: Optional[float]
    rankB: Optional[float]
    prob_A_wins: float


class TennisPredictor:
    """
    Baseline pre-match tennis predictor (ATP):
    - Features: rank_diff + one-hot(surface, tourney_level)
    - Ranking for inference: last known rank from match history strictly BEFORE the given date
    """

    def __init__(self, data_dir: str = "data", seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed

        self.matches = self._load_matches()
        self.rank_index = self._build_rank_index(self.matches)
        self.form_index = self._build_form_index(self.matches)

        self.model = self._train_baseline_model(self.matches)


    def _load_matches(self) -> pd.DataFrame:
        paths = sorted(glob.glob(os.path.join(self.data_dir, "atp_matches_20*.csv")))
        if not paths:
            raise FileNotFoundError(
                f"No match files found in {self.data_dir}. Expected atp_matches_2015.csv etc."
            )

        df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)

        required = ["winner_name", "loser_name", "surface", "tourney_date", "winner_rank", "loser_rank"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

        df = df.dropna(subset=required).copy()
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["tourney_date"]).copy()
        df = df.sort_values("tourney_date").reset_index(drop=True)

        if "tourney_level" not in df.columns:
            df["tourney_level"] = "U"

        return df

    # ---------- Rank indexing (for inference) ----------

    def _build_rank_index(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        For each player, build a tiny table of (date, rank) observations based on their matches.
        We'll use this to look up the last known rank BEFORE a target date.
        """
        records: Dict[str, List[tuple]] = {}

        for _, r in df.iterrows():
            d = r["tourney_date"]

            w = r["winner_name"]
            l = r["loser_name"]

            records.setdefault(w, []).append((d, float(r["winner_rank"])))
            records.setdefault(l, []).append((d, float(r["loser_rank"])))

        # Convert to DataFrames and sort
        out: Dict[str, pd.DataFrame] = {}
        for player, lst in records.items():
            tmp = pd.DataFrame(lst, columns=["date", "rank"]).sort_values("date").reset_index(drop=True)

            # If multiple rank observations on same date, keep the last
            tmp = tmp.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            out[player] = tmp

        return out

    def _last_known_rank(self, player: str, date: pd.Timestamp) -> Optional[float]:
        """
        Returns the player's most recent known rank strictly BEFORE `date`.
        If not found, returns None.
        """
        hist = self.rank_index.get(player)
        if hist is None or hist.empty:
            return None

        dates = hist["date"].values  # numpy datetime64
        # find insertion point for `date`, then take the previous index
        idx = np.searchsorted(dates, np.datetime64(date), side="left") - 1
        if idx < 0:
            return None
        return float(hist.iloc[idx]["rank"])

    # ---------- Training ----------
    

    def _make_training_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert (winner, loser) rows into randomized (A, B) training rows
        and add time-safe recent form features computed from matches strictly before each match.
        """
        rng = np.random.default_rng(self.seed)

        overall_hist = defaultdict(lambda: deque(maxlen=N_FORM))  # player -> deque of 1/0
        surface_hist = defaultdict(lambda: defaultdict(lambda: deque(maxlen=N_FORM)))  # player -> surface -> deque

        rows = []

        for _, r in df.iterrows():
            date = r["tourney_date"]
            surf = r["surface"]
            level = r.get("tourney_level", "U")

            w = r["winner_name"]
            l = r["loser_name"]
            w_rank = float(r["winner_rank"])
            l_rank = float(r["loser_rank"])

            def winrates(player: str):
                o = overall_hist[player]
                s = surface_hist[player][surf]

                # if no history, use 0.5 so we don't bias early matches
                wr_o = (sum(o) / len(o)) if len(o) else 0.5
                wr_s = (sum(s) / len(s)) if len(s) else 0.5
                return wr_o, wr_s

            # compute form BEFORE updating with this match (prevents leakage)
            w_wr_o, w_wr_s = winrates(w)
            l_wr_o, l_wr_s = winrates(l)

            # Randomize A/B so A isn't always the winner
            swap = rng.random() < 0.5
            if not swap:
                A_rank, B_rank = w_rank, l_rank
                A_wr_o, B_wr_o = w_wr_o, l_wr_o
                A_wr_s, B_wr_s = w_wr_s, l_wr_s
                y = 1
            else:
                A_rank, B_rank = l_rank, w_rank
                A_wr_o, B_wr_o = l_wr_o, w_wr_o
                A_wr_s, B_wr_s = l_wr_s, w_wr_s
                y = 0

            rows.append({
                "date": date,
                "surface": surf,
                "tourney_level": level,
                "rank_diff": A_rank - B_rank,
                "form_diff": A_wr_o - B_wr_o,
                "surface_form_diff": A_wr_s - B_wr_s,
                "y": y,
            })

            # update histories AFTER creating features
            overall_hist[w].append(1); overall_hist[l].append(0)
            surface_hist[w][surf].append(1); surface_hist[l][surf].append(0)
    
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


    def _train_baseline_model(self, df: pd.DataFrame) -> Pipeline:
        data = self._make_training_examples(df)

        # time split (train on older data)
        train = data[data["date"] < "2023-01-01"].copy()

        preprocess = ColumnTransformer(
            transformers=[
                ("num", "passthrough", ["rank_diff", "form_diff", "surface_form_diff"]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["surface", "tourney_level"]),
            ]
        )

        clf = Pipeline(
            steps=[
                ("prep", preprocess),
                ("model", LogisticRegression(max_iter=300)),
            ]
        )

        X_train = train[["rank_diff", "form_diff", "surface_form_diff", "surface", "tourney_level"]]
        y_train = train["y"]
        clf.fit(X_train, y_train)
        return clf

    def _build_form_index(self, df: pd.DataFrame) -> dict:
        """
        Build per-player chronological history of outcomes for fast inference-time form computation.
        Stores one table per player: columns [date, win, surface]
        """
        records = {}

        for _, r in df.iterrows():
            d = r["tourney_date"]
            surf = r["surface"]
            w = r["winner_name"]
            l = r["loser_name"]

            records.setdefault(w, []).append((d, 1, surf))
            records.setdefault(l, []).append((d, 0, surf))

        out = {}
        for player, lst in records.items():
            tmp = pd.DataFrame(lst, columns=["date", "win", "surface"]).sort_values("date").reset_index(drop=True)
            out[player] = tmp
        return out


    def _recent_form(self, player: str, date: pd.Timestamp, n: int = N_FORM) -> float:
        """
        Win rate over last n matches strictly BEFORE date. Returns 0.5 if insufficient history.
        """
        hist = self.form_index.get(player)
        if hist is None or hist.empty:
            return 0.5

        dates = hist["date"].values
        idx = np.searchsorted(dates, np.datetime64(date), side="left")
        if idx <= 0:
            return 0.5

        window = hist.iloc[max(0, idx - n):idx]
        return float(window["win"].mean()) if len(window) else 0.5


    def _recent_surface_form(self, player: str, date: pd.Timestamp, surface: str, n: int = N_FORM) -> float:
        """
        Win rate over last n matches on `surface` strictly BEFORE date. Returns 0.5 if insufficient history.
        """
        hist = self.form_index.get(player)
        if hist is None or hist.empty:
            return 0.5

        # filter to matches on this surface first, then take last n before date
        surf_hist = hist[hist["surface"] == surface]
        if surf_hist.empty:
            return 0.5

        dates = surf_hist["date"].values
        idx = np.searchsorted(dates, np.datetime64(date), side="left")
        if idx <= 0:
            return 0.5

        window = surf_hist.iloc[max(0, idx - n):idx]
        return float(window["win"].mean()) if len(window) else 0.5

    def predict_match(
        self,
        playerA: str,
        playerB: str,
        surface: str,
        date: DateLike,
        tourney_level: str = "U",
    ) -> PredictionResult:
        """
        Returns P(Player A wins) for a pre-match prediction.

        """
        d = _to_timestamp(date)

        rankA = self._last_known_rank(playerA, d)
        rankB = self._last_known_rank(playerB, d)

        if rankA is None or rankB is None:
            missing = []
            if rankA is None:
                missing.append(playerA)
            if rankB is None:
                missing.append(playerB)
            raise ValueError(
                f"Not enough history to find pre-match ranks for: {', '.join(missing)} "
                f"before {d.date()}. Try an earlier date or check name spelling."
            )

        rank_diff = rankA - rankB

        formA = self._recent_form(playerA, d)
        formB = self._recent_form(playerB, d)
        surfFormA = self._recent_surface_form(playerA, d, surface)
        surfFormB = self._recent_surface_form(playerB, d, surface)

        form_diff = formA - formB
        surface_form_diff = surfFormA - surfFormB

        X = pd.DataFrame(
            [{
                "rank_diff": rank_diff,
                "form_diff": form_diff,
                "surface_form_diff": surface_form_diff,
                "surface": surface,
                "tourney_level": tourney_level,
            }]
        )

        prob_A_wins = float(self.model.predict_proba(X)[:, 1][0])

        return PredictionResult(
            playerA=playerA,
            playerB=playerB,
            date=d,
            surface=surface,
            tourney_level=tourney_level,
            rankA=rankA,
            rankB=rankB,
            prob_A_wins=prob_A_wins
        )