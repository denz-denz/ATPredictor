import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predictor import TennisPredictor  # noqa: E402


@st.cache_resource
def load_predictor() -> TennisPredictor:
    return TennisPredictor.load(model_dir="models")


st.set_page_config(page_title="ATP Match Predictor", page_icon="🎾")
st.title("🎾 ATP Match Predictor")
st.caption("Pick two players and a tournament to predict the winner and how many sets the match will take.")

predictor = load_predictor()
players = predictor.list_recent_players(months=18)
tournaments = predictor.list_tournaments()


def filtered_options(all_options, query):
    if not query:
        return all_options
    q = query.lower()
    return [o for o in all_options if q in o.lower()]


col1, col2 = st.columns(2)
with col1:
    query_a = st.text_input("Search Player A", placeholder="Type to search…")
    player_a = st.selectbox("Player A", filtered_options(players, query_a), index=None, placeholder="Choose a player")
with col2:
    query_b = st.text_input("Search Player B", placeholder="Type to search…")
    player_b = st.selectbox("Player B", filtered_options(players, query_b), index=None, placeholder="Choose a player")

query_t = st.text_input("Search Tournament", placeholder="Type to search…")
tournament = st.selectbox("Tournament", filtered_options(tournaments, query_t), index=None, placeholder="Choose a tournament")

if st.button("Predict", type="primary"):
    if not player_a or not player_b:
        st.error("Choose both Player A and Player B.")
    elif player_a == player_b:
        st.error("Choose two different players.")
    elif not tournament:
        st.error("Choose a tournament.")
    else:
        try:
            result = predictor.predict_match_by_tournament(
                playerA=player_a,
                playerB=player_b,
                tourney_name=tournament,
                date=pd.Timestamp.now(),
            )
        except ValueError as e:
            st.error(str(e))
        else:
            winner = player_a if result.prob_A_wins >= 0.5 else player_b
            p_win = max(result.prob_A_wins, 1 - result.prob_A_wins)
            p_sets = result.sets_probs[result.predicted_sets]
            joint = p_win * p_sets

            st.subheader(f"{winner} to win in {result.predicted_sets} sets")
            st.caption(
                f"P({winner} wins) {p_win:.0%} × P(match goes {result.predicted_sets} sets) {p_sets:.0%} "
                f"= {joint:.0%} combined probability of this exact outcome"
            )

            st.metric(f"P({winner} wins)", f"{p_win:.0%}")

            st.write("Win probability")
            st.progress(result.prob_A_wins, text=f"{player_a} {result.prob_A_wins:.0%} — {1 - result.prob_A_wins:.0%} {player_b}")

            st.write("Set count probabilities")
            st.caption("How many sets the match takes, independent of who wins.")
            sets_df = pd.DataFrame(
                {"sets": [str(s) for s in result.sets_probs.keys()], "probability": list(result.sets_probs.values())}
            )
            chart = (
                alt.Chart(sets_df)
                .mark_bar()
                .encode(
                    x=alt.X("sets:N", title="Sets", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("probability:Q", title="Probability"),
                )
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(" · ".join(f"{s} sets: {p:.0%}" for s, p in result.sets_probs.items()))
