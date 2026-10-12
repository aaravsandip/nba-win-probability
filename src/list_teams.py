from data import load_games

games = load_games("data/TeamStatistics.csv", first_season=2026)
for team in sorted(set(games["home"])):
    print(team)