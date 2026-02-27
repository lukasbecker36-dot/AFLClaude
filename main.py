import pandas as pd
df = pd.read_csv("afl_team_rows.csv")
print(df["season"].value_counts().sort_index())

