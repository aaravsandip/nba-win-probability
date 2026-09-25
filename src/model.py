"""The rating model: recency-weighted point margin + optional strength of schedule."""
import numpy as np
import pandas as pd
def team_ratings(games, as_of, decay=0.92, sos=True):
 """Rate every team using ONLY games played before `as_of`.
 A team's rating = weighted average point margin, where a game from
 w weeks ago counts decay**w as much as a game played today.
 """
 past = games[games["date"] < as_of]
 if past.empty:
 return {}
 # Look at each game from both teams' side: home margin is +m, away margin is -m.
 side = pd.DataFrame({
 "team": pd.concat([past["home"], past["away"]]),
 "opp": pd.concat([past["away"], past["home"]]),
 "margin": pd.concat([past["margin"], -past["margin"]]),
 "date": pd.concat([past["date"], past["date"]]),
 })
 weeks_ago = (as_of - side["date"]).dt.days / 7.0
 side["w"] = decay ** weeks_ago
 total_weight = side["w"].groupby(side["team"]).sum()
 raw = (side["margin"] * side["w"]).groupby(side["team"]).sum() / total_weight
 if not sos:
 return raw.to_dict()
 # Strength of schedule: beating a good team by 5 means more than beating a bad team by 5.
 # Adjusted margin = actual margin + the opponent's rating.
 side["adj"] = side["margin"] + side["opp"].map(raw).fillna(0.0)
 adj = (side["adj"] * side["w"]).groupby(side["team"]).sum() / total_weight
 adj = adj - adj.mean() # keep the league average at 0
 return adj.to_dict()
def rating_gap(ratings, home, away):
 """Home rating minus away rating (0 for a team we've never seen)."""
 return ratings.get(home, 0.0) - ratings.get(away, 0.0)
def win_probability(gap, a, b):
 """Logistic curve: P(home wins) = 1 / (1 + e^-(a*gap + b)).
 b is the home-court edge; b/a converts it into points.
 """
 return 1.0 / (1.0 + np.exp(-(a * np.asarray(gap) + b)))