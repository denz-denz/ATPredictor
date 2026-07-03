import glob
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, Union, Dict, List

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

N_FORM = 20 # last 20 matches for recent form

K_ELO = 32
STARTING_ELO = 1500.0

RETIRED_MARKERS = ("RET", "W/O", "WO", "DEF", "ABN", "ABD")


def _safe_rate(numer, denom) -> Optional[float]:
    if pd.isna(numer) or pd.isna(denom) or denom <= 0:
        return None
    return float(numer) / float(denom)


def _global_mean_rate(df: pd.DataFrame, numer_col: str, denom_col: str) -> float:
    numer = pd.concat([df[f"w_{numer_col}"], df[f"l_{numer_col}"]])
    denom = pd.concat([df[f"w_{denom_col}"], df[f"l_{denom_col}"]])
    valid = denom > 0
    return float((numer[valid] / denom[valid]).mean())


DateLike = Union[str, pd.Timestamp]


def _to_timestamp(d: DateLike) -> pd.Timestamp:
    return pd.to_datetime(d)


def _parse_num_sets(score, best_of) -> Optional[int]:
    """
    Number of sets actually played, or None if unknowable (retirement/walkover/
    missing score) or inconsistent with best_of. Excluding these from the
    sets-model target avoids training on matches that didn't run their course.
    """
    if not isinstance(score, str) or pd.isna(best_of):
        return None

    s = score.strip()
    su = s.upper()
    if not s or any(marker in su for marker in RETIRED_MARKERS):
        return None

    n = sum(1 for tok in s.split() if "-" in tok)
    bo = int(best_of)
    lo = bo // 2 + 1
    if n < lo or n > bo:
        return None
    return n


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
    best_of: int
    sets_probs: Dict[int, float]
    predicted_sets: int
    summary: str


class TennisPredictor:
    """
    Baseline pre-match tennis predictor (ATP):
    - Features: rank_diff, recent form and surface form, one-hot(surface, tourney_level)
    - Ranking for inference: last known rank from match history strictly BEFORE the given date
    """

    def __init__(self, data_dir: str = "data", seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed

        self.matches = self._load_matches()
        self.rank_index = self._build_rank_index(self.matches)
        self.form_index = self._build_form_index(self.matches)
        self.tournament_index = self._build_tournament_index(self.matches)
        self._activity = self.matches[["tourney_date", "winner_name", "loser_name"]].copy()

        self.elo_index, self.surface_elo_index = self._build_elo_index(self.matches)
        (self.ace_index, self.fsw_index, self.bps_index,
         self._fallback_ace, self._fallback_fsw, self._fallback_bps) = self._build_serve_stat_index(self.matches)

        train_examples = self._make_training_examples(self.matches)
        self.model = self._train_win_model(train_examples)
        self.sets_model = self._train_sets_model(train_examples)

    @classmethod
    def load(cls, model_dir: str = "models") -> "TennisPredictor":
        """
        Reconstructs a trained predictor from artifacts written by `save()`,
        without touching raw match data. Used at deploy time so the app never
        needs the underlying dataset, only the trained model + compact
        (player, date, rank/outcome) lookup tables derived from it.
        """
        obj = cls.__new__(cls)
        state = joblib.load(os.path.join(model_dir, "predictor.joblib"))
        obj.data_dir = None
        obj.matches = None
        obj.seed = state["seed"]
        obj.rank_index = state["rank_index"]
        obj.form_index = state["form_index"]
        obj.tournament_index = state["tournament_index"]
        obj._activity = state["activity"]
        obj.elo_index = state["elo_index"]
        obj.surface_elo_index = state["surface_elo_index"]
        obj.ace_index = state["ace_index"]
        obj.fsw_index = state["fsw_index"]
        obj.bps_index = state["bps_index"]
        obj._fallback_ace = state["fallback_ace"]
        obj._fallback_fsw = state["fallback_fsw"]
        obj._fallback_bps = state["fallback_bps"]
        obj.model = state["model"]
        obj.sets_model = state["sets_model"]
        return obj

    def save(self, model_dir: str = "models") -> None:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(
            {
                "seed": self.seed,
                "rank_index": self.rank_index,
                "form_index": self.form_index,
                "tournament_index": self.tournament_index,
                "activity": self._activity,
                "elo_index": self.elo_index,
                "surface_elo_index": self.surface_elo_index,
                "ace_index": self.ace_index,
                "fsw_index": self.fsw_index,
                "bps_index": self.bps_index,
                "fallback_ace": self._fallback_ace,
                "fallback_fsw": self._fallback_fsw,
                "fallback_bps": self._fallback_bps,
                "model": self.model,
                "sets_model": self.sets_model,
            },
            os.path.join(model_dir, "predictor.joblib"),
        )


    def _load_matches(self) -> pd.DataFrame:
        paths = sorted(glob.glob(os.path.join(self.data_dir, "atp_matches_20*.csv")))
        if not paths:
            raise FileNotFoundError(
                f"No match files found in {self.data_dir}. Expected atp_matches_2015.csv etc."
            )

        df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)

        required = [
            "winner_name", "loser_name", "surface", "tourney_date",
            "winner_rank", "loser_rank", "tourney_name",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

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

        for prefix in ("w_", "l_"):
            for stat in ("ace", "svpt", "1stIn", "1stWon", "bpSaved", "bpFaced"):
                col = f"{prefix}{stat}"
                if col not in df.columns:
                    df[col] = np.nan

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

    # ---------- Elo indexing (for inference) ----------

    def _build_elo_index(self, df: pd.DataFrame):
        """
        Sequentially replays match history to compute overall + per-surface
        Elo ratings, snapshotting each player's rating *after* every match so
        we can later look up "rating strictly before date" at inference time.
        Standard Elo update: everyone starts at STARTING_ELO, K=K_ELO.
        """
        elo = defaultdict(lambda: STARTING_ELO)
        surface_elo = defaultdict(lambda: defaultdict(lambda: STARTING_ELO))

        overall_records: Dict[str, List[tuple]] = {}
        surface_records: Dict[str, List[tuple]] = {}

        for _, r in df.iterrows():
            d = r["tourney_date"]
            surf = r["surface"]
            w, l = r["winner_name"], r["loser_name"]

            w_elo, l_elo = elo[w], elo[l]
            w_selo, l_selo = surface_elo[surf][w], surface_elo[surf][l]

            exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_elo) / 400.0))
            elo[w] = w_elo + K_ELO * (1 - exp_w)
            elo[l] = l_elo - K_ELO * (1 - exp_w)

            exp_w_s = 1.0 / (1.0 + 10 ** ((l_selo - w_selo) / 400.0))
            surface_elo[surf][w] = w_selo + K_ELO * (1 - exp_w_s)
            surface_elo[surf][l] = l_selo - K_ELO * (1 - exp_w_s)

            overall_records.setdefault(w, []).append((d, elo[w]))
            overall_records.setdefault(l, []).append((d, elo[l]))
            surface_records.setdefault(w, []).append((d, surf, surface_elo[surf][w]))
            surface_records.setdefault(l, []).append((d, surf, surface_elo[surf][l]))

        overall_index = {
            player: pd.DataFrame(lst, columns=["date", "elo"]).sort_values("date").reset_index(drop=True)
            for player, lst in overall_records.items()
        }
        surface_index = {
            player: pd.DataFrame(lst, columns=["date", "surface", "elo"]).sort_values("date").reset_index(drop=True)
            for player, lst in surface_records.items()
        }
        return overall_index, surface_index

    def _last_known_elo(self, player: str, date: pd.Timestamp) -> float:
        hist = self.elo_index.get(player)
        if hist is None or hist.empty:
            return STARTING_ELO
        dates = hist["date"].values
        idx = np.searchsorted(dates, np.datetime64(date), side="left") - 1
        if idx < 0:
            return STARTING_ELO
        return float(hist.iloc[idx]["elo"])

    def _last_known_surface_elo(self, player: str, surface: str, date: pd.Timestamp) -> float:
        hist = self.surface_elo_index.get(player)
        if hist is None or hist.empty:
            return STARTING_ELO
        surf_hist = hist[hist["surface"] == surface]
        if surf_hist.empty:
            return STARTING_ELO
        dates = surf_hist["date"].values
        idx = np.searchsorted(dates, np.datetime64(date), side="left") - 1
        if idx < 0:
            return STARTING_ELO
        return float(surf_hist.iloc[idx]["elo"])

    # ---------- Trailing serve/return stats (for inference) ----------

    def _build_serve_stat_index(self, df: pd.DataFrame):
        """
        Per-player chronological history of three serve/return rates
        (ace rate, first-serve-points-won rate, break-point-saved rate),
        one row per match where that rate was computable. Also returns the
        dataset-wide mean of each rate as a fallback for players with no
        (or insufficient) history.
        """
        fallback_ace = _global_mean_rate(df, "ace", "svpt")
        fallback_fsw = _global_mean_rate(df, "1stWon", "1stIn")
        fallback_bps = _global_mean_rate(df, "bpSaved", "bpFaced")

        ace_records: Dict[str, List[tuple]] = {}
        fsw_records: Dict[str, List[tuple]] = {}
        bps_records: Dict[str, List[tuple]] = {}

        for _, r in df.iterrows():
            d = r["tourney_date"]
            for side, prefix in (("winner_name", "w_"), ("loser_name", "l_")):
                player = r[side]
                ace_rate = _safe_rate(r.get(f"{prefix}ace"), r.get(f"{prefix}svpt"))
                if ace_rate is not None:
                    ace_records.setdefault(player, []).append((d, ace_rate))
                fsw_rate = _safe_rate(r.get(f"{prefix}1stWon"), r.get(f"{prefix}1stIn"))
                if fsw_rate is not None:
                    fsw_records.setdefault(player, []).append((d, fsw_rate))
                bps_rate = _safe_rate(r.get(f"{prefix}bpSaved"), r.get(f"{prefix}bpFaced"))
                if bps_rate is not None:
                    bps_records.setdefault(player, []).append((d, bps_rate))

        def to_index(records):
            return {
                player: pd.DataFrame(lst, columns=["date", "rate"]).sort_values("date").reset_index(drop=True)
                for player, lst in records.items()
            }

        return to_index(ace_records), to_index(fsw_records), to_index(bps_records), fallback_ace, fallback_fsw, fallback_bps

    def _trailing_rate(self, index: dict, player: str, date: pd.Timestamp, fallback: float, n: int = N_FORM) -> float:
        hist = index.get(player)
        if hist is None or hist.empty:
            return fallback
        dates = hist["date"].values
        idx = np.searchsorted(dates, np.datetime64(date), side="left")
        if idx <= 0:
            return fallback
        window = hist.iloc[max(0, idx - n):idx]
        return float(window["rate"].mean()) if len(window) else fallback

    # Training
    
    def _make_training_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert (winner, loser) rows into randomized (A, B) training rows
        and add time-safe recent form features computed from matches strictly before each match.
        """
        rng = np.random.default_rng(self.seed)

        overall_hist = defaultdict(lambda: deque(maxlen=N_FORM))  # player -> deque of 1/0
        surface_hist = defaultdict(lambda: defaultdict(lambda: deque(maxlen=N_FORM)))  # player -> surface -> deque
        elo = defaultdict(lambda: STARTING_ELO)
        surface_elo = defaultdict(lambda: defaultdict(lambda: STARTING_ELO))
        ace_hist = defaultdict(lambda: deque(maxlen=N_FORM))
        fsw_hist = defaultdict(lambda: deque(maxlen=N_FORM))
        bps_hist = defaultdict(lambda: deque(maxlen=N_FORM))

        fallback_ace = _global_mean_rate(df, "ace", "svpt")
        fallback_fsw = _global_mean_rate(df, "1stWon", "1stIn")
        fallback_bps = _global_mean_rate(df, "bpSaved", "bpFaced")

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

            def rate_feat(hist, fallback):
                return (sum(hist) / len(hist)) if len(hist) else fallback

            # compute form/elo/serve-rates BEFORE updating with this match (prevents leakage)
            w_wr_o, w_wr_s = winrates(w)
            l_wr_o, l_wr_s = winrates(l)
            w_elo, l_elo = elo[w], elo[l]
            w_selo, l_selo = surface_elo[surf][w], surface_elo[surf][l]
            w_ace_r, l_ace_r = rate_feat(ace_hist[w], fallback_ace), rate_feat(ace_hist[l], fallback_ace)
            w_fsw_r, l_fsw_r = rate_feat(fsw_hist[w], fallback_fsw), rate_feat(fsw_hist[l], fallback_fsw)
            w_bps_r, l_bps_r = rate_feat(bps_hist[w], fallback_bps), rate_feat(bps_hist[l], fallback_bps)

            # Randomize so A isn't always the winner
            swap = rng.random() < 0.5
            if not swap:
                A_rank, B_rank = w_rank, l_rank
                A_wr_o, B_wr_o = w_wr_o, l_wr_o
                A_wr_s, B_wr_s = w_wr_s, l_wr_s
                A_elo, B_elo = w_elo, l_elo
                A_selo, B_selo = w_selo, l_selo
                A_ace, B_ace = w_ace_r, l_ace_r
                A_fsw, B_fsw = w_fsw_r, l_fsw_r
                A_bps, B_bps = w_bps_r, l_bps_r
                y = 1
            else:
                A_rank, B_rank = l_rank, w_rank
                A_wr_o, B_wr_o = l_wr_o, w_wr_o
                A_wr_s, B_wr_s = l_wr_s, w_wr_s
                A_elo, B_elo = l_elo, w_elo
                A_selo, B_selo = l_selo, w_selo
                A_ace, B_ace = l_ace_r, w_ace_r
                A_fsw, B_fsw = l_fsw_r, w_fsw_r
                A_bps, B_bps = l_bps_r, w_bps_r
                y = 0

            best_of = int(r.get("best_of", 3))
            rows.append({
                "date": date,
                "surface": surf,
                "tourney_level": level,
                "rank_diff": A_rank - B_rank,
                "form_diff": A_wr_o - B_wr_o,
                "surface_form_diff": A_wr_s - B_wr_s,
                "elo_diff": A_elo - B_elo,
                "surface_elo_diff": A_selo - B_selo,
                "ace_rate_diff": A_ace - B_ace,
                "first_serve_win_rate_diff": A_fsw - B_fsw,
                "bp_save_rate_diff": A_bps - B_bps,
                "y": y,
                "best_of": best_of,
                "num_sets": _parse_num_sets(r.get("score"), best_of),
            })

            # update all trailing state after creating features
            overall_hist[w].append(1); overall_hist[l].append(0)
            surface_hist[w][surf].append(1); surface_hist[l][surf].append(0)

            exp_w = 1.0 / (1.0 + 10 ** ((l_elo - w_elo) / 400.0))
            elo[w] = w_elo + K_ELO * (1 - exp_w)
            elo[l] = l_elo - K_ELO * (1 - exp_w)
            exp_w_s = 1.0 / (1.0 + 10 ** ((l_selo - w_selo) / 400.0))
            surface_elo[surf][w] = w_selo + K_ELO * (1 - exp_w_s)
            surface_elo[surf][l] = l_selo - K_ELO * (1 - exp_w_s)

            w_ace_rate = _safe_rate(r.get("w_ace"), r.get("w_svpt"))
            l_ace_rate = _safe_rate(r.get("l_ace"), r.get("l_svpt"))
            if w_ace_rate is not None: ace_hist[w].append(w_ace_rate)
            if l_ace_rate is not None: ace_hist[l].append(l_ace_rate)

            w_fsw_rate = _safe_rate(r.get("w_1stWon"), r.get("w_1stIn"))
            l_fsw_rate = _safe_rate(r.get("l_1stWon"), r.get("l_1stIn"))
            if w_fsw_rate is not None: fsw_hist[w].append(w_fsw_rate)
            if l_fsw_rate is not None: fsw_hist[l].append(l_fsw_rate)

            w_bps_rate = _safe_rate(r.get("w_bpSaved"), r.get("w_bpFaced"))
            l_bps_rate = _safe_rate(r.get("l_bpSaved"), r.get("l_bpFaced"))
            if w_bps_rate is not None: bps_hist[w].append(w_bps_rate)
            if l_bps_rate is not None: bps_hist[l].append(l_bps_rate)

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


    WIN_NUM_COLS = [
        "rank_diff", "form_diff", "surface_form_diff",
        "elo_diff", "surface_elo_diff", "ace_rate_diff", "first_serve_win_rate_diff", "bp_save_rate_diff",
    ]

    def _train_win_model(self, data: pd.DataFrame) -> Pipeline:
        # time split (train on older data)
        train = data[data["date"] < "2023-01-01"].copy()

        preprocess = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.WIN_NUM_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["surface", "tourney_level"]),
            ]
        )

        clf = Pipeline(
            steps=[
                ("prep", preprocess),
                ("model", HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=300,
                    random_state=self.seed
                )),
            ]
        )

        X_train = train[self.WIN_NUM_COLS + ["surface", "tourney_level"]]
        y_train = train["y"]
        clf.fit(X_train, y_train)
        return clf

    SETS_NUM_COLS = [
        "abs_rank_diff", "abs_form_diff", "abs_surface_form_diff",
        "abs_elo_diff", "abs_surface_elo_diff", "abs_ace_rate_diff",
        "abs_first_serve_win_rate_diff", "abs_bp_save_rate_diff", "best_of",
    ]

    def _train_sets_model(self, data: pd.DataFrame) -> Pipeline:
        """
        Predicts how many sets a match goes, independent of who wins: match
        length is driven by how close the matchup is (rank/form/elo/serve gap),
        not by player identity, so features are unsigned (abs) versions of the
        win model's diff features plus best_of. Rows with unknown set count
        (retirements/walkovers/missing scores) are excluded.
        """
        train = data[(data["date"] < "2023-01-01") & data["num_sets"].notna()].copy()
        for col in ["rank_diff", "form_diff", "surface_form_diff", "elo_diff",
                    "surface_elo_diff", "ace_rate_diff", "first_serve_win_rate_diff", "bp_save_rate_diff"]:
            train[f"abs_{col}"] = train[col].abs()

        preprocess = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.SETS_NUM_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["surface", "tourney_level"]),
            ]
        )

        clf = Pipeline(
            steps=[
                ("prep", preprocess),
                ("model", HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=300,
                    random_state=self.seed
                )),
            ]
        )

        X_train = train[self.SETS_NUM_COLS + ["surface", "tourney_level"]]
        y_train = train["num_sets"].astype(int)
        clf.fit(X_train, y_train)
        return clf

    def _build_tournament_index(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Maps tourney_name -> {surface, tourney_level, best_of} using each
        tournament's most recently observed values (tournaments occasionally
        change surface/category across years).
        """
        idx: Dict[str, dict] = {}
        for name, g in df.sort_values("tourney_date").groupby("tourney_name"):
            last = g.iloc[-1]
            idx[name] = {
                "surface": last["surface"],
                "tourney_level": last["tourney_level"],
                "best_of": int(last["best_of"]),
            }
        return idx

    def list_tournaments(self) -> List[str]:
        return sorted(self.tournament_index.keys())

    def get_tournament_meta(self, tourney_name: str) -> dict:
        meta = self.tournament_index.get(tourney_name)
        if meta is None:
            raise ValueError(f"Unknown tournament: {tourney_name}")
        return meta

    def list_recent_players(self, months: int = 18) -> List[str]:
        """Players with at least one match in the most recent `months` of data."""
        cutoff = self._activity["tourney_date"].max() - pd.DateOffset(months=months)
        recent = self._activity[self._activity["tourney_date"] >= cutoff]
        players = pd.unique(pd.concat([recent["winner_name"], recent["loser_name"]]))
        return sorted(players)

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
        best_of: int = 3,
    ) -> PredictionResult:
        """
        Returns P(Player A wins) plus a set-count distribution for a pre-match prediction.

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

        eloA = self._last_known_elo(playerA, d)
        eloB = self._last_known_elo(playerB, d)
        surfEloA = self._last_known_surface_elo(playerA, surface, d)
        surfEloB = self._last_known_surface_elo(playerB, surface, d)
        elo_diff = eloA - eloB
        surface_elo_diff = surfEloA - surfEloB

        aceA = self._trailing_rate(self.ace_index, playerA, d, self._fallback_ace)
        aceB = self._trailing_rate(self.ace_index, playerB, d, self._fallback_ace)
        fswA = self._trailing_rate(self.fsw_index, playerA, d, self._fallback_fsw)
        fswB = self._trailing_rate(self.fsw_index, playerB, d, self._fallback_fsw)
        bpsA = self._trailing_rate(self.bps_index, playerA, d, self._fallback_bps)
        bpsB = self._trailing_rate(self.bps_index, playerB, d, self._fallback_bps)
        ace_rate_diff = aceA - aceB
        first_serve_win_rate_diff = fswA - fswB
        bp_save_rate_diff = bpsA - bpsB

        X = pd.DataFrame(
            [{
                "rank_diff": rank_diff,
                "form_diff": form_diff,
                "surface_form_diff": surface_form_diff,
                "elo_diff": elo_diff,
                "surface_elo_diff": surface_elo_diff,
                "ace_rate_diff": ace_rate_diff,
                "first_serve_win_rate_diff": first_serve_win_rate_diff,
                "bp_save_rate_diff": bp_save_rate_diff,
                "surface": surface,
                "tourney_level": tourney_level,
            }]
        )

        prob_A_wins = float(self.model.predict_proba(X)[:, 1][0])

        X_sets = pd.DataFrame(
            [{
                "abs_rank_diff": abs(rank_diff),
                "abs_form_diff": abs(form_diff),
                "abs_surface_form_diff": abs(surface_form_diff),
                "abs_elo_diff": abs(elo_diff),
                "abs_surface_elo_diff": abs(surface_elo_diff),
                "abs_ace_rate_diff": abs(ace_rate_diff),
                "abs_first_serve_win_rate_diff": abs(first_serve_win_rate_diff),
                "abs_bp_save_rate_diff": abs(bp_save_rate_diff),
                "best_of": best_of,
                "surface": surface,
                "tourney_level": tourney_level,
            }]
        )
        sets_proba = self.sets_model.predict_proba(X_sets)[0]
        lo = best_of // 2 + 1
        valid_classes = set(range(lo, best_of + 1))
        raw = {
            int(cls): float(p)
            for cls, p in zip(self.sets_model.classes_, sets_proba)
            if int(cls) in valid_classes
        }
        total = sum(raw.values()) or 1.0
        sets_probs = {cls: p / total for cls, p in raw.items()}
        predicted_sets = max(sets_probs, key=sets_probs.get)

        winner = playerA if prob_A_wins >= 0.5 else playerB
        p_win = max(prob_A_wins, 1 - prob_A_wins)
        joint_prob = p_win * sets_probs[predicted_sets]
        summary = f"{winner} to win in {predicted_sets} sets ({joint_prob:.0%})"

        return PredictionResult(
            playerA=playerA,
            playerB=playerB,
            date=d,
            surface=surface,
            tourney_level=tourney_level,
            rankA=rankA,
            rankB=rankB,
            prob_A_wins=prob_A_wins,
            best_of=best_of,
            sets_probs=sets_probs,
            predicted_sets=predicted_sets,
            summary=summary,
        )

    def predict_match_by_tournament(
        self,
        playerA: str,
        playerB: str,
        tourney_name: str,
        date: DateLike,
    ) -> PredictionResult:
        """Convenience wrapper: looks up surface/level/best_of from tournament history."""
        meta = self.get_tournament_meta(tourney_name)
        return self.predict_match(
            playerA=playerA,
            playerB=playerB,
            surface=meta["surface"],
            date=date,
            tourney_level=meta["tourney_level"],
            best_of=meta["best_of"],
        )