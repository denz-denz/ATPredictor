import os
import sys

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
            st.subheader(result.summary)

            winner = player_a if result.prob_A_wins >= 0.5 else player_b
            st.metric(f"P({winner} wins)", f"{max(result.prob_A_wins, 1 - result.prob_A_wins):.0%}")

            st.write("Win probability")
            st.progress(result.prob_A_wins, text=f"{player_a} {result.prob_A_wins:.0%} — {1 - result.prob_A_wins:.0%} {player_b}")

            st.write("Set count probabilities")
            sets_df = pd.DataFrame(
                {"sets": list(result.sets_probs.keys()), "probability": list(result.sets_probs.values())}
            ).set_index("sets")
            st.bar_chart(sets_df)
