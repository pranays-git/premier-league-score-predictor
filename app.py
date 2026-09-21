from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy.stats import poisson

st.set_page_config(
    page_title="Premier League Score Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ Premier League Score Predictor")
st.caption("Poisson + Elo model using recent form and completed Premier League results.")

# Load the final trained models
home_model = sm.load("models/poisson_elo_home_goals.pickle")
away_model = sm.load("models/poisson_elo_away_goals.pickle")

# Load completed match history
matches = pd.read_csv(
    "data/processed/epl_matches_clean.csv",
    parse_dates=["Date"]
).sort_values("Date")

feature_columns = [
    "home_form_points_5",
    "away_form_points_5",
    "home_avg_goals_for_5",
    "away_avg_goals_for_5",
    "home_avg_goals_against_5",
    "away_avg_goals_against_5",
    "elo_difference",
]

team_history = defaultdict(list)
ratings = defaultdict(lambda: 1500.0)

K_FACTOR = 20
HOME_ADVANTAGE = 60

# Rebuild each team's current form and Elo rating
for _, match in matches.iterrows():
    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]
    home_goals = match["FTHG"]
    away_goals = match["FTAG"]

    if home_goals > away_goals:
        home_points, away_points, actual_home = 3, 0, 1.0
    elif home_goals < away_goals:
        home_points, away_points, actual_home = 0, 3, 0.0
    else:
        home_points, away_points, actual_home = 1, 1, 0.5

    expected_home = 1 / (
        1 + 10 ** ((ratings[away_team] - (ratings[home_team] + HOME_ADVANTAGE)) / 400)
    )

    ratings[home_team] += K_FACTOR * (actual_home - expected_home)
    ratings[away_team] += K_FACTOR * ((1 - actual_home) - (1 - expected_home))

    team_history[home_team].append({
        "goals_for": home_goals,
        "goals_against": away_goals,
        "points": home_points,
    })
    team_history[away_team].append({
        "goals_for": away_goals,
        "goals_against": home_goals,
        "points": away_points,
    })

def recent_stats(team, last_n=5):
    recent = team_history[team][-last_n:]

    return {
        "form_points_5": sum(game["points"] for game in recent),
        "avg_goals_for_5": np.mean([game["goals_for"] for game in recent]),
        "avg_goals_against_5": np.mean([game["goals_against"] for game in recent]),
    }

def make_prediction(home_team, away_team):
    home = recent_stats(home_team)
    away = recent_stats(away_team)

    input_data = pd.DataFrame([{
        "home_form_points_5": home["form_points_5"],
        "away_form_points_5": away["form_points_5"],
        "home_avg_goals_for_5": home["avg_goals_for_5"],
        "away_avg_goals_for_5": away["avg_goals_for_5"],
        "home_avg_goals_against_5": home["avg_goals_against_5"],
        "away_avg_goals_against_5": away["avg_goals_against_5"],
        "elo_difference": ratings[home_team] - ratings[away_team],
    }])

    X = sm.add_constant(input_data[feature_columns], has_constant="add")

    home_xg = home_model.predict(X).iloc[0]
    away_xg = away_model.predict(X).iloc[0]

    score_rows = []

    for home_goals in range(8):
        for away_goals in range(8):
            probability = (
                poisson.pmf(home_goals, home_xg) *
                poisson.pmf(away_goals, away_xg)
            )

            score_rows.append({
                "Score": f"{home_goals}-{away_goals}",
                "Probability": probability,
                "HomeGoals": home_goals,
                "AwayGoals": away_goals,
            })

    scores = pd.DataFrame(score_rows)

    home_win = scores[scores["HomeGoals"] > scores["AwayGoals"]]["Probability"].sum()
    draw = scores[scores["HomeGoals"] == scores["AwayGoals"]]["Probability"].sum()
    away_win = scores[scores["HomeGoals"] < scores["AwayGoals"]]["Probability"].sum()

    top_scores = scores.sort_values("Probability", ascending=False).head(5).copy()
    top_scores["Probability"] = (top_scores["Probability"] * 100).round(1)

    return home_xg, away_xg, home_win, draw, away_win, top_scores

teams = sorted(matches["HomeTeam"].unique())

home_team = st.selectbox("Home team", teams, index=teams.index("Arsenal"))
away_options = [team for team in teams if team != home_team]
away_team = st.selectbox(
    "Away team",
    away_options,
    index=away_options.index("Chelsea") if "Chelsea" in away_options else 0,
)

if st.button("Predict match", type="primary", use_container_width=True):
    home_xg, away_xg, home_win, draw, away_win, top_scores = make_prediction(
        home_team, away_team
    )

    st.subheader(f"{home_team} vs {away_team}")
    st.write(f"Expected goals: **{home_team} {home_xg:.2f} — {away_team} {away_xg:.2f}**")

    first, second, third = st.columns(3)
    first.metric("Home win", f"{home_win:.1%}")
    second.metric("Draw", f"{draw:.1%}")
    third.metric("Away win", f"{away_win:.1%}")

    st.subheader("Most likely exact scores")
    st.dataframe(
        top_scores[["Score", "Probability"]],
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.caption(
    f"Data current through {matches['Date'].max().date()}. "
    "Predictions are probabilities, not guarantees."
)

