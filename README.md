# Premier League Score Predictor

A machine-learning web app that estimates Premier League match scores, expected goals, and home-win/draw/away-win probabilities.

## Model

The app uses a Poisson regression model with:

- Each team's form from its previous five matches
- Recent goals scored and conceded
- Elo-based team-strength difference
- Home advantage

The model produces expected goals, exact-score probabilities, and match-result probabilities.

## Performance

On an unseen Premier League test season, the Elo-enhanced model achieved:

- Home-goal MAE: 0.970
- Away-goal MAE: 0.886
- Exact-score accuracy: 10.7%
- Match-result accuracy: 51.2%

## Data

The model uses completed Premier League results through 20 September 2026.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
