import re
import time
from io import StringIO
from typing import List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

AFL_ROOT = "https://afltables.com/afl/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Stat columns typically present in AFL Tables player match-stat tables.
# We'll keep only these (prevents weird columns from creeping in).
KNOWN_PLAYER_STATS = {
    "KI", "MK", "HB", "DI", "GL", "BH", "HO", "TK", "RB", "IF", "CL", "CG",
    "FF", "FA", "BR", "CP", "UP", "CM", "MI", "1%", "BO", "GA", "%P"
}


# ----------------------------
# Fetch + season links
# ----------------------------
def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(0.2)
    return r.text


def get_match_stats_links(season: int) -> List[str]:
    season_url = f"{AFL_ROOT}seas/{season}.html"
    html = fetch_html(season_url)
    soup = BeautifulSoup(html, "html.parser")

    links = []
    for a in soup.find_all("a", string="Match stats"):
        href = a.get("href")
        if not href:
            continue
        # href can be "../stats/..." or "stats/..."
        while href.startswith("../"):
            href = href[3:]
        href = href.lstrip("/")
        links.append(AFL_ROOT + href)

    # Deduplicate while keeping order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# ----------------------------
# Parse helpers
# ----------------------------
def _parse_final_score(score_str: str) -> Optional[int]:
    """Parse strings like '12.14.86' and return 86."""
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", str(score_str).strip())
    return int(m.group(3)) if m else None


def _find_top_summary_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Find the small top table (has Qrt margin / Qrt scores)."""
    for t in tables:
        flat = " ".join(t.astype(str).fillna("").values.flatten().tolist())
        if "Qrt margin" in flat or "Qrt scores" in flat:
            return t
    return None


def _clean_cell(x) -> str:
    return str(x).strip()


def _is_arrow(s: str) -> bool:
    return s in {"←", "→"}


def _extract_team_name_from_row(row_vals) -> str:
    """
    Summary table rows often start with a nav arrow column.
    Extract the first "word-like" cell that isn't the arrow, isn't a score, etc.
    """
    for cell in row_vals:
        s = _clean_cell(cell)
        if not s or _is_arrow(s):
            continue
        # Ignore score tokens like 12.14.86
        if re.search(r"\d+\.\d+\.\d+", s):
            continue
        # Ignore qtr margin like "SY by 9"
        if " by " in s.lower():
            continue
        # Ignore labels
        if s.lower() in {"qrt margin", "qrt scores", "field umpires"}:
            continue
        # Team names have letters
        if any(ch.isalpha() for ch in s):
            return s

    # Fallback
    return _clean_cell(row_vals[0])


def _extract_home_away_and_scores_from_summary(summary: pd.DataFrame) -> Tuple[str, str, Optional[int], Optional[int]]:
    """
    Use the top summary table:
      - find the two rows that contain a goals.behinds.total pattern
      - first team row is home; second is away
    """
    df = summary.copy()

    team_rows = []
    for idx in range(df.shape[0]):
        row_vals = df.iloc[idx].tolist()
        joined = " ".join(_clean_cell(x) for x in row_vals)
        if re.search(r"\d+\.\d+\.\d+", joined):
            team_rows.append(idx)

    if len(team_rows) < 2:
        raise ValueError("Could not find two team rows in summary table.")

    r1 = df.iloc[team_rows[0]].tolist()
    r2 = df.iloc[team_rows[1]].tolist()

    home_team = _extract_team_name_from_row(r1)
    away_team = _extract_team_name_from_row(r2)

    def last_total(row):
        last = None
        for cell in row:
            tot = _parse_final_score(cell)
            if tot is not None:
                last = tot
        return last

    home_score = last_total(r1)
    away_score = last_total(r2)

    return home_team, away_team, home_score, away_score


def _flatten_cols(cols) -> List[str]:
    """
    pandas.read_html sometimes returns MultiIndex columns like:
      ('Sydney Match Statistics [Season][Game by Game]', 'KI')
    We want just the last level, e.g. 'KI'
    """
    out = []
    for c in cols:
        if isinstance(c, tuple):
            out.append(str(c[-1]).strip())
        else:
            out.append(str(c).strip())
    return out


def _player_table_score(df: pd.DataFrame) -> int:
    """
    Score a table for being a player stats table.
    Higher score = more likely.
    """
    cols = [str(c).strip() for c in df.columns]
    score = 0

    # Any column containing 'player'?
    if any("player" in c.lower() for c in cols):
        score += 50

    # Known stat headers present?
    stat_hits = sum(1 for c in cols if str(c).strip() in KNOWN_PLAYER_STATS)
    score += stat_hits * 10

    # Player tables usually have many rows (18+)
    if df.shape[0] >= 18:
        score += 30
    if df.shape[0] >= 25:
        score += 20

    # Typically many columns
    if df.shape[1] >= 8:
        score += 10

    return score


def _sum_player_table(df: pd.DataFrame) -> pd.Series:
    """
    Sum numeric columns of a player table AFTER flattening header names.
    Keep only known stat columns (KI, HB, CL, IF, etc.).
    """
    df2 = df.copy()
    df2.columns = _flatten_cols(df2.columns)

    # Drop obvious non-stat columns
    drop_cols = [c for c in df2.columns if c.lower() in {"#", "no"} or "player" in c.lower()]
    keep = df2.drop(columns=drop_cols, errors="ignore")

    # Keep only stat columns we recognise
    keep = keep[[c for c in keep.columns if c in KNOWN_PLAYER_STATS]]

    keep_num = keep.apply(pd.to_numeric, errors="coerce")
    return keep_num.sum(axis=0, skipna=True)


# ----------------------------
# Match parse (2 rows per match)
# ----------------------------
def parse_match_page_to_team_totals(match_url: str, debug: bool = False) -> pd.DataFrame:
    html = fetch_html(match_url)
    tables = pd.read_html(StringIO(html))

    summary = _find_top_summary_table(tables)
    if summary is None:
        raise ValueError(f"Could not find top summary table: {match_url}")

    home_team, away_team, home_score, away_score = _extract_home_away_and_scores_from_summary(summary)

    # Score all tables and take the top 2 that look like player tables
    scored = []
    for idx, t in enumerate(tables):
        s = _player_table_score(t)
        scored.append((s, idx, t))
    scored.sort(reverse=True, key=lambda x: x[0])

    if debug:
        print("Top table scores:", [(s, idx, tables[idx].shape) for s, idx, _ in scored[:6]])

    # Prefer those above threshold
    candidates = [(s, idx, t) for s, idx, t in scored if s >= 60]
    if len(candidates) < 2:
        # fallback: top 2 with reasonable size
        candidates = [(s, idx, t) for s, idx, t in scored if t.shape[0] >= 10 and t.shape[1] >= 6][:2]

    if len(candidates) < 2:
        raise ValueError(f"Could not find two player tables on page: {match_url}")

    home_players = candidates[0][2]
    away_players = candidates[1][2]

    home_sums = _sum_player_table(home_players)
    away_sums = _sum_player_table(away_players)

    home_row = {"Team": home_team, "is_home": 1, "score_for": home_score, "score_against": away_score}
    away_row = {"Team": away_team, "is_home": 0, "score_for": away_score, "score_against": home_score}

    for k, v in home_sums.items():
        home_row[k] = v
    for k, v in away_sums.items():
        away_row[k] = v

    out = pd.DataFrame([home_row, away_row])
    out["match_url"] = match_url
    return out


# ----------------------------
# Season build
# ----------------------------
def build_season_dataset(season: int, max_games: Optional[int] = None) -> pd.DataFrame:
    links = get_match_stats_links(season)
    print(f"Found {len(links)} matches for {season}")

    rows = []
    for i, link in enumerate(links, start=1):
        try:
            df = parse_match_page_to_team_totals(link, debug=(i == 1))
            df["season"] = season
            rows.append(df)
            print(f"[{i}/{len(links)}] OK {link}")
        except Exception as e:
            print(f"[{i}/{len(links)}] FAIL {link}: {e}")

        if max_games and i >= max_games:
            break

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)

import os

TEAM_CACHE = "afl_team_rows.csv"

def load_existing_team_rows(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, encoding="utf-8", engine="python")
    return pd.DataFrame()

def update_season_incremental(season: int, out_path: str = TEAM_CACHE) -> pd.DataFrame:
    existing = load_existing_team_rows(out_path)
    existing_urls = set(existing["match_url"].dropna().astype(str)) if len(existing) else set()

    links = get_match_stats_links(season)
    new_links = [u for u in links if u not in existing_urls]

    print(f"Season {season}: {len(links)} total links, {len(new_links)} new")

    if not new_links:
        print("Nothing new to add.")
        return existing

    rows = []
    for i, url in enumerate(new_links, start=1):
        try:
            df2 = parse_match_page_to_team_totals(url, debug=(i == 1))
            df2["season"] = season
            rows.append(df2)
            print(f"[{i}/{len(new_links)}] OK {url}")
        except Exception as e:
            print(f"[{i}/{len(new_links)}] FAIL {url}: {e}")

    if not rows:
        print("No new rows parsed successfully.")
        return existing

    new_df = pd.concat(rows, ignore_index=True)

    combined = pd.concat([existing, new_df], ignore_index=True)

    # de-dupe: 2 rows per match_url
    combined = combined.drop_duplicates(subset=["match_url", "Team", "is_home"], keep="last")
    combined = combined.sort_values(["season", "match_url", "is_home"]).reset_index(drop=True)

    combined.to_csv(out_path, index=False)
    print(f"Saved updated file: {os.path.abspath(out_path)}")
    print(f"Rows now: {len(combined)} (added {len(new_df)})")
    return combined

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    update_season_incremental(2026, TEAM_CACHE)

