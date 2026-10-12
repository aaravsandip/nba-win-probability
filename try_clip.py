import pandas as pd

margins = pd.Series([35, -28, 12, -5, 22])
print(margins.clip(lower=-20, upper=20))