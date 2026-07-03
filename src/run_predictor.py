from predictor import TennisPredictor

if __name__ == "__main__":
    predictor = TennisPredictor(data_dir="data")

    res = predictor.predict_match(
        playerA="Jannik Sinner",
        playerB="Carlos Alcaraz",
        surface="Hard",
        date="2026-01-20",
        tourney_level="G",
        best_of=5,
    )

    print(res)
    print(f"\nP({res.playerA} wins) = {res.prob_A_wins:.3f}")
    print(f"Sets distribution: {res.sets_probs}")
    print(res.summary)
