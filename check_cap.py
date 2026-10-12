import sys
sys.path.insert(0, "src")
import pandas as pd
from model import team_ratings

# One fake game: Team A beats Team B by 40
games = pd.DataFrame({
    "date": [pd.Timestamp("2026-01-01")],
    "home": ["Team A"], "away": ["Team B"],
    "margin": [40],
})
ratings = team_ratings(games, pd.Timestamp("2026-01-08"), decay=0.95, sos=False)
print(ratings)