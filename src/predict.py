"""Live predictions for the 2026-27 season.
1. Make picks BEFORE games are played:
 python src/predict.py picks
 Reads data/upcoming.csv (columns: date,home,away) and adds the picks to predictions/log.csv.
2. After the games are played (and you've downloaded fresh data):
 python src/predict.py score
 Matches your logged picks to real results and prints your running record.
"""
import os
import sys
import numpy as np
import pandas as pd
from data import load_games
from model import team_ratings, rating_gap, win_probability
from backtest import score
DATA = os.path.join("data", "TeamStatistics.csv")
LOG = os.path.join("predictions", "log.csv")
def best_settings():
 t = pd.read_csv(os.path.join("results", "tuning.csv")).sort_values("log_loss").iloc[0]
 return float(t["decay"]), bool(t["sos"]), float(t["a"]), float(t["b"])
def make_picks():
 decay, sos, a, b = best_settings()
 games = load_games(DATA, first_season=2025)
 upcoming = pd.read_csv(os.path.join("data", "upcoming.csv"), parse_dates=["date"])
 known = set(games["home"]) | set(games["away"])
 unknown = (set(upcoming["home"]) | set(upcoming["away"])) - known
 if unknown:
 sys.exit(f"These team names don't match the data (check spelling): {sorted(unknown)}")
 rows = []
 for day, todays in upcoming.groupby("date"):
 ratings = team_ratings(games, day, decay=decay, sos=sos)
 for _, g in todays.iterrows():
 p = float(win_probability(rating_gap(ratings, g["home"], g["away"]), a, b))
 rows.append({"date": day.date(), "home": g["home"], "away": g["away"],
 "p_home": round(p, 4), "pick": g["home"] if p >= 0.5 else g["away"],
 "made_on": pd.Timestamp.now().date()})
 new = pd.DataFrame(rows)
 os.makedirs("predictions", exist_ok=True)
 if os.path.exists(LOG):
 old = pd.read_csv(LOG)
 old_keys = set(zip(old["date"].astype(str), old["home"], old["away"]))
 keys = zip(new["date"].astype(str), new["home"], new["away"])
 new = new[[k not in old_keys for k in keys]]
 new = pd.concat([old, new], ignore_index=True)
 new.to_csv(LOG, index=False)
 print(new.tail(len(rows)).to_string(index=False))
 print(f"\nSaved to {LOG}. Commit it BEFORE tip-off so the timestamp proves you predicted first.")
def score_picks():
 games = load_games(DATA, first_season=2026)
 log = pd.read_csv(LOG, parse_dates=["date"])
 results = games[["date", "home", "away", "home_win"]]
 merged = log.merge(results, on=["date", "home", "away"], how="left")
 done = merged.dropna(subset=["home_win"])
 if done.empty:
 print("No finished games found yet. Download fresh data and try again.")
 return
 s = score(done["home_win"].values.astype(int), done["p_home"].values)
 print(f"Games scored: {len(done)} (waiting on {len(merged) - len(done)})")
 print(f"Accuracy: {s['accuracy']:.3f} Log loss: {s['log_loss']:.4f} Brier: {s['brier']:.4f}")
 done.to_csv(os.path.join("predictions", "scored.csv"), index=False)
if __name__ == "__main__":
 command = sys.argv[1] if len(sys.argv) > 1 else ""
 if command == "picks":
 make_picks()
 elif command == "score":
 score_picks()
 else:
 print("Use: python src/predict.py picks OR python src/predict.py score")