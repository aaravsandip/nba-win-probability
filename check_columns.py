import pandas as pd
df = pd.read_csv("data/TeamStatistics.csv", nrows=5)
print(df.columns.tolist())
print(df.head(2).T)