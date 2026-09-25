"""Load the Kaggle TeamStatistics.csv file and turn it into one row per game."""
import pandas as pd
def _full_name(df, city_col, name_col):
 """'Los Angeles' + 'Lakers' -> 'Los Angeles Lakers' so LA teams stay separate."""
 return df[city_col].astype(str).str.strip() + " " + df[name_col].astype(str).str.strip()
 
def load_games(filepath, first_season=2019):
 """Return a DataFrame with ONE row per game:
 date, season, home, away, home_score, away_score, home_win (1 or 0).
 """
 df = pd.read_csv(filepath, low_memory=False)

 # The date column name depends on the dataset version.
 date_col = "gameDateTimeEst" if "gameDateTimeEst" in df.columns else "gameDate"
 dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
 df["date"] = dates.dt.tz_localize(None).dt.normalize()

 df["team"] = _full_name(df, "teamCity", "teamName")
 df["opp"] = _full_name(df, "opponentTeamCity", "opponentTeamName")

 # Keep only the row written from the HOME team's point of view.
 # This fixes the "every game counted twice" bug and tells us who was home.
 home_rows = df[df["home"] == 1].copy()

 games = pd.DataFrame({
 "game_id": home_rows["gameId"],
 "date": home_rows["date"],
 "home": home_rows["team"],
 "away": home_rows["opp"],
 "home_score": home_rows["teamScore"],
 "away_score": home_rows["opponentScore"],
 })
 games = games.dropna(subset=["date", "home_score", "away_score"])
 games = games[games["home_score"] != games["away_score"]] # skip bad rows

 # Season label: the 2025-26 season is "2026" (games from Oct 2025 to Jun 2026).
 games["season"] = games["date"].dt.year + (games["date"].dt.month >= 8).astype(int)
 games = games[games["season"] >= first_season]

 # Regular season + playoffs only (drop preseason / All-Star games) if the column exists.
 if "gameType" in df.columns:
 keep = df.loc[home_rows.index, "gameType"].isin(["Regular Season", "Playoffs"])
 games = games[keep.reindex(games.index).fillna(False)]
 games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
 games["margin"] = games["home_score"] - games["away_score"]
 return games.drop_duplicates("game_id").sort_values("date").reset_index(drop=True)