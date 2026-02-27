import pandas as pd
import os

from scrape_afl_tables import build_season_dataset
# ^ adjust import to wherever your scraper lives

TEAM_CACHE = "afl_team_rows.csv"
SEASONS_TO_ADD = [2025]   # can be [2025] or [2024, 2025]

all_new = []

for season in SEASONS_TO_ADD:
    print(f"Scraping season {season}...")
    df = build_season_dataset(season)
    df["season"] = season
    all_new.append(df)

new_rows = pd.concat(all_new, ignore_index=True)

if os.path.exists(TEAM_CACHE):
    existing = pd.read_csv(TEAM_CACHE)
    combined = pd.concat([existing, new_rows], ignore_index=True)
else:
    combined = new_rows

# Critical: avoid duplicates if script is re-run
combined = combined.drop_duplicates(
    subset=["match_url", "Team"],
    keep="last"
)

combined.to_csv(TEAM_CACHE, index=False)

print("Done.")
print("Rows:", len(combined))
print("Seasons:", sorted(combined["season"].unique()))
