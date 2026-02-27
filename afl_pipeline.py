# src/afl_pipeline.py
import os
import re
import numpy as np
import pandas as pd

from typing import Optional, Tuple, Dict, List

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score


# =========================
# TEAM NORMALISATION
# =========================
def norm_team(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip()
    s_lower = s.lower()
    lower_map = {
        "gws giants": "Greater Western Sydney",
        "gws": "Greater Western Sydney",
        "gold coast suns": "Gold Coast",
        "sydney swans": "Sydney",
        "geelong cats": "Geelong",
    }
    if s_lower in lower_map:
        return lower_map[s_lower]

    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u2013", "-").replace("\u2014", "-")

    mapping = {
        "GWS": "Greater Western Sydney",
        "GWS Giants": "Greater Western Sydney",
        "Greater Western Sydney Giants": "Greater Western Sydney",
        "Footscray": "Western Bulldogs",
        "Brisbane": "Brisbane Lions",
        "Brisbane Bears": "Brisbane Lions",
        "Kangaroos": "North Melbourne",
        "St. Kilda": "St Kilda",
        "St Kilda Saints": "St Kilda",
        "Sydney Swans": "Sydney",
        "West Coast Eagles": "West Coast",
        "Adelaide Crows": "Adelaide",
        "Port Adelaide Power": "Port Adelaide",
        "Gold Coast Suns": "Gold Coast",
        "Fremantle Dockers": "Fremantle",
        "Geelong Cats": "Geelong",
        "Hawthorn Hawks": "Hawthorn",
        "Melbourne Demons": "Melbourne",
        "Richmond Tigers": "Richmond",
        "Collingwood Magpies": "Collingwood",
        "Carlton Blues": "Carlton",
        "Essendon Bombers": "Essendon",
    }

    if s in mapping:
        return mapping[s]

    s = re.sub(r"\s+", " ", s)
    s = s.replace("St.", "St").strip()
    return mapping.get(s, s)


def resolve_model_stats(team_log: pd.DataFrame, desired: List[str]) -> List[str]:
    cols = list(team_log.columns)
    cols_lower = {c.lower(): c for c in cols}

    aliases = {
        "I50": ["IF", "I50", "I50s", "Inside50", "Inside 50", "Inside 50s", "I-50"],
        "CL": ["CL", "Clr", "Clearances"],
        "DI": ["DI", "Disp", "Disposals"],
    }

    resolved = []
    for d in desired:
        if d in cols:
            resolved.append(d); continue
        if d.lower() in cols_lower:
            resolved.append(cols_lower[d.lower()]); continue

        cands = aliases.get(d, [d])
        found = None
        for c in cands:
            if c in cols:
                found = c; break
            if c.lower() in cols_lower:
                found = cols_lower[c.lower()]; break
        if found is None:
            raise ValueError(
                f"Could not resolve stat '{d}'. Update MODEL_STATS or aliases. "
                f"Example columns: {cols[:40]} (total {len(cols)})"
            )
        resolved.append(found)

    out = []
    for c in resolved:
        if c not in out:
            out.append(c)
    return out


# =========================
# IO
# =========================
def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def load_team_rows(team_cache: str) -> pd.DataFrame:
    if not os.path.exists(team_cache):
        raise FileNotFoundError(f"Missing {team_cache}. Put afl_team_rows.csv in this folder.")
    df = safe_read_csv(team_cache)
    for col in ["season", "match_url", "Team", "is_home", "score_for", "score_against"]:
        if col not in df.columns:
            raise ValueError(f"{team_cache} missing required column: {col}")
    return df


def match_url_to_date(url: str) -> pd.Timestamp:
    m = re.search(r"(\d{8})\.html$", str(url))
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")


def load_fixtures_csv(path: str) -> pd.DataFrame:
    fx = pd.read_csv(path)

    fx = fx.rename(columns={
        "Season": "season",
        "venue": "Venue",
        "round": "round",
        "date": "date",
        "home_team": "home_team",
        "away_team": "away_team",
    })

    fx["season"] = fx["season"].astype(int)
    fx["date"] = pd.to_datetime(fx["date"], dayfirst=True, errors="coerce")
    fx["home_team"] = fx["home_team"].astype(str).str.strip().map(norm_team)
    fx["away_team"] = fx["away_team"].astype(str).str.strip().map(norm_team)

    keep = ["round", "date", "Venue", "home_team", "away_team", "season"]
    fx = fx[keep].copy()
    fx = fx.dropna(subset=["date", "home_team", "away_team", "season"])
    return fx


# =========================
# MATCH-LEVEL BUILD
# =========================
def build_match_level(team_log: pd.DataFrame) -> pd.DataFrame:
    needed = {"Team", "is_home", "score_for", "score_against", "match_url", "season"}
    missing = needed - set(team_log.columns)
    if missing:
        raise ValueError(f"TEAM_LOG missing required columns: {missing}")

    tl = team_log.copy()
    tl["Team"] = tl["Team"].map(norm_team)
    tl["date"] = tl["match_url"].map(match_url_to_date)
    tl["season"] = tl["season"].astype(int)

    home = tl[tl["is_home"].astype(int) == 1].copy()
    away = tl[tl["is_home"].astype(int) == 0].copy()

    core = {"Team", "is_home", "score_for", "score_against", "match_url", "season", "date"}
    stat_cols = [c for c in tl.columns if c not in core]

    home = home.rename(columns={"Team": "home_team", "score_for": "home_score", "score_against": "away_score"})
    away = away.rename(columns={"Team": "away_team", "score_for": "away_score2", "score_against": "home_score2"})

    m = home[["match_url", "date", "season", "home_team", "home_score", "away_score"] + stat_cols].merge(
        away[["match_url", "away_team"] + stat_cols],
        on="match_url",
        how="inner",
        suffixes=("_home", "_away"),
    )

    m["home_win"] = (m["home_score"] > m["away_score"]).astype(int)
    return m


# =========================
# ODDS MERGE
# =========================
def odds_to_implied_prob(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    return 1.0 / o


def novig_two_way(home_odds: pd.Series, away_odds: pd.Series) -> pd.Series:
    h = pd.to_numeric(home_odds, errors="coerce")
    a = pd.to_numeric(away_odds, errors="coerce")
    ph = 1.0 / h
    pa = 1.0 / a
    denom = ph + pa
    return ph / denom


def merge_odds_meta(matches: pd.DataFrame, odds_file: str) -> pd.DataFrame:
    if not os.path.exists(odds_file):
        raise FileNotFoundError(f"Missing {odds_file} in this folder.")

    try:
        odds = pd.read_csv(odds_file, header=1, encoding="cp1252")
    except Exception:
        odds = safe_read_csv(odds_file)

    odds["date"] = pd.to_datetime(odds["Date"], dayfirst=True, errors="coerce")
    odds["home_team"] = odds["Home Team"].astype(str).str.strip().map(norm_team)
    odds["away_team"] = odds["Away Team"].astype(str).str.strip().map(norm_team)

    odds["p_home_open"] = odds_to_implied_prob(odds.get("Home Odds Open", np.nan))
    odds["p_home_close"] = odds_to_implied_prob(odds.get("Home Odds Close", np.nan))

    if "Home Odds" in odds.columns and odds["p_home_open"].isna().all():
        odds["p_home_open"] = odds_to_implied_prob(odds["Home Odds"])
    if "Home Odds" in odds.columns and odds["p_home_close"].isna().all():
        odds["p_home_close"] = odds_to_implied_prob(odds["Home Odds"])

    odds["p_home_open_nv"] = novig_two_way(
        odds.get("Home Odds Open", np.nan),
        odds.get("Away Odds Open", np.nan),
    )
    odds["p_home_close_nv"] = novig_two_way(
        odds.get("Home Odds Close", np.nan),
        odds.get("Away Odds Close", np.nan),
    )

    if odds["p_home_open_nv"].isna().all() and {"Home Odds", "Away Odds"}.issubset(odds.columns):
        odds["p_home_open_nv"] = novig_two_way(odds["Home Odds"], odds["Away Odds"])
    if odds["p_home_close_nv"].isna().all() and {"Home Odds", "Away Odds"}.issubset(odds.columns):
        odds["p_home_close_nv"] = novig_two_way(odds["Home Odds"], odds["Away Odds"])

    keep = [
        "date", "home_team", "away_team",
        "Venue", "Play Off Game?",
        "p_home_open", "p_home_close",
        "p_home_open_nv", "p_home_close_nv",
        "Home Odds Open", "Home Odds Close",
        "Away Odds Open", "Away Odds Close",
        "Home Odds", "Away Odds",
    ]
    keep = [c for c in keep if c in odds.columns]
    odds_small = odds[keep].copy()

    merged = matches.merge(odds_small, on=["date", "home_team", "away_team"], how="left")

    missing = merged["p_home_open"].isna() & merged["p_home_open_nv"].isna()
    if missing.any():
        swapped = matches.merge(
            odds_small.rename(columns={"home_team": "away_team", "away_team": "home_team"}),
            on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_sw"),
        )

        for col in ["Venue", "Play Off Game?"]:
            if col in swapped.columns and f"{col}_sw" in swapped.columns:
                merged.loc[missing, col] = swapped.loc[missing, f"{col}_sw"]

        if "p_home_open_nv_sw" in swapped.columns:
            merged.loc[missing, "p_home_open_nv"] = 1.0 - swapped.loc[missing, "p_home_open_nv_sw"]
        if "p_home_close_nv_sw" in swapped.columns:
            merged.loc[missing, "p_home_close_nv"] = 1.0 - swapped.loc[missing, "p_home_close_nv_sw"]

        if "p_home_open_sw" in swapped.columns:
            merged.loc[missing, "p_home_open"] = 1.0 - swapped.loc[missing, "p_home_open_sw"]
        if "p_home_close_sw" in swapped.columns:
            merged.loc[missing, "p_home_close"] = 1.0 - swapped.loc[missing, "p_home_close_sw"]

    return merged


# =========================
# ELO
# =========================
def elo_expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def compute_elo_diff(matches: pd.DataFrame, k: float, home_adv: float, base: float) -> pd.DataFrame:
    d = matches.sort_values("date").copy()
    teams = pd.unique(pd.concat([d["home_team"], d["away_team"]]))
    elos = {t: float(base) for t in teams}

    elo_home_pre = []
    elo_away_pre = []

    for _, r in d.iterrows():
        h, a = r["home_team"], r["away_team"]
        eh = elos.get(h, float(base))
        ea = elos.get(a, float(base))

        elo_home_pre.append(eh)
        elo_away_pre.append(ea)

        exp_home = elo_expected(eh + home_adv, ea)
        y = int(r["home_win"])
        elos[h] = eh + k * (y - exp_home)
        elos[a] = ea + k * (exp_home - y)

    d["elo_home_pre"] = elo_home_pre
    d["elo_away_pre"] = elo_away_pre
    d["elo_diff"] = d["elo_home_pre"] - d["elo_away_pre"]
    return d


def compute_current_elos(matches: pd.DataFrame, k: float, home_adv: float, base: float) -> Dict[str, float]:
    d = matches.sort_values("date").copy()
    teams = pd.unique(pd.concat([d["home_team"], d["away_team"]]))
    elos = {t: float(base) for t in teams}

    for _, r in d.iterrows():
        h, a = r["home_team"], r["away_team"]
        eh = elos.get(h, float(base))
        ea = elos.get(a, float(base))

        exp_home = elo_expected(eh + home_adv, ea)
        y = int(r["home_win"])
        elos[h] = eh + k * (y - exp_home)
        elos[a] = ea + k * (exp_home - y)

    return elos


# =========================
# MOMENTUM / TRAJECTORY MODEL FEATURES
# =========================
MOMENTUM_FEATURES = [
    "mom_elo_slope",
    "mom_second_half_delta",
    "mom_win_rate_last8",
    "mom_pct_trend",
]


def compute_team_momentum_features(
    matches_with_elo: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    Compute per-team momentum features from a single season's matches.

    Features (all computed from season data only):
    - mom_elo_slope: OLS slope of pre-match Elo vs match index within the season.
    - mom_second_half_delta: Win rate in 2nd half minus 1st half of season.
    - mom_win_rate_last8: Win rate over last 8 matches.
    - mom_pct_trend: Scoring percentage (PF/PA) in last 8 minus season-average percentage.

    Returns DataFrame with columns: team + 4 momentum features.
    """
    sm = matches_with_elo[matches_with_elo["season"] == season].copy()
    if len(sm) == 0:
        return pd.DataFrame(columns=["team"] + MOMENTUM_FEATURES)

    # Expand to team-level rows (one row per team per match)
    rows = []
    for _, r in sm.sort_values("date").iterrows():
        for side in ["home", "away"]:
            team = r[f"{side}_team"]
            elo_pre = r[f"elo_{side}_pre"]
            won = int(r["home_win"]) if side == "home" else 1 - int(r["home_win"])
            pf = r["home_score"] if side == "home" else r["away_score"]
            pa = r["away_score"] if side == "home" else r["home_score"]
            rows.append({
                "team": team,
                "date": r["date"],
                "elo_pre": elo_pre,
                "won": won,
                "pf": pf,
                "pa": pa,
            })
    tdf = pd.DataFrame(rows).sort_values(["team", "date"])
    tdf["match_idx"] = tdf.groupby("team").cumcount()

    results = []
    for team, g in tdf.groupby("team"):
        n = len(g)

        # 1. mom_elo_slope — OLS slope of elo_pre vs match_idx
        if n >= 2:
            x = g["match_idx"].values.astype(float)
            y = g["elo_pre"].values.astype(float)
            x_mean = x.mean()
            y_mean = y.mean()
            denom = ((x - x_mean) ** 2).sum()
            slope = ((x - x_mean) * (y - y_mean)).sum() / denom if denom > 0 else 0.0
        else:
            slope = 0.0

        # 2. mom_second_half_delta — win rate 2nd half minus 1st half
        half = n // 2
        if half > 0:
            first_half_wr = g["won"].iloc[:half].mean()
            second_half_wr = g["won"].iloc[half:].mean()
            second_half_delta = second_half_wr - first_half_wr
        else:
            second_half_delta = 0.0

        # 3. mom_win_rate_last8
        last8 = g.tail(8)
        win_rate_last8 = last8["won"].mean()

        # 4. mom_pct_trend — scoring pct last 8 minus season average pct
        season_pf = g["pf"].sum()
        season_pa = g["pa"].sum()
        season_pct = season_pf / season_pa if season_pa > 0 else 1.0

        last8_pf = last8["pf"].sum()
        last8_pa = last8["pa"].sum()
        last8_pct = last8_pf / last8_pa if last8_pa > 0 else 1.0

        pct_trend = last8_pct - season_pct

        results.append({
            "team": team,
            "mom_elo_slope": slope,
            "mom_second_half_delta": second_half_delta,
            "mom_win_rate_last8": win_rate_last8,
            "mom_pct_trend": pct_trend,
        })

    return pd.DataFrame(results)


# =========================
# LIST AGE PRIORS
# =========================
LIST_PRIOR_FEATURES = ["list_avg_age"]


def load_list_priors(path: str = "data/team_season_list_priors.csv") -> pd.DataFrame:
    """Load team-season list priors CSV. Returns DataFrame with season, team, avg_age."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run: python scrape_list_ages.py"
        )
    df = pd.read_csv(path)
    for col in ["season", "team", "avg_age"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")
    return df


def add_list_priors(matches: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """
    Merge list priors onto match rows by (season, team) for home and away.
    Creates diff columns: diff_list_avg_age = home_avg_age - away_avg_age.
    Raises ValueError if any (season, team) pairs are missing from priors.
    """
    df = matches.copy()

    # Check for missing priors
    needed = set()
    for side in ["home_team", "away_team"]:
        for _, r in df[[side, "season"]].drop_duplicates().iterrows():
            needed.add((int(r["season"]), r[side]))
    available = set(zip(priors["season"].astype(int), priors["team"]))
    missing = needed - available
    if missing:
        raise ValueError(
            f"Missing list priors for {len(missing)} (season, team) pairs: "
            + ", ".join(f"({s}, {t})" for s, t in sorted(missing))
        )

    # Merge home
    home_priors = priors.rename(columns={"team": "home_team", "avg_age": "list_avg_age_home"})
    df = df.merge(home_priors[["season", "home_team", "list_avg_age_home"]],
                  on=["season", "home_team"], how="left")

    # Merge away
    away_priors = priors.rename(columns={"team": "away_team", "avg_age": "list_avg_age_away"})
    df = df.merge(away_priors[["season", "away_team", "list_avg_age_away"]],
                  on=["season", "away_team"], how="left")

    # Compute diff
    df["diff_list_avg_age"] = df["list_avg_age_home"] - df["list_avg_age_away"]

    return df


def add_preseason_momentum(matches_with_elo: pd.DataFrame) -> pd.DataFrame:
    """
    For every match in season S, merge momentum features computed from season S-1.

    Returns the input DataFrame with 8 new columns added:
      mom_<feat>_home, mom_<feat>_away, plus diff_mom_<feat> for each feature.
    """
    df = matches_with_elo.copy()
    seasons = sorted(df["season"].unique())

    # Pre-compute momentum for each season
    mom_by_season: Dict[int, pd.DataFrame] = {}
    for s in seasons:
        mom = compute_team_momentum_features(df, s)
        if len(mom) > 0:
            mom_by_season[s] = mom

    # For each match in season S, look up momentum from S-1
    mom_frames = []
    for s in seasons:
        prior = s - 1
        mask = df["season"] == s
        chunk = df.loc[mask].copy()

        if prior in mom_by_season:
            snap = mom_by_season[prior]
            home = snap.rename(columns={"team": "home_team"})
            away = snap.rename(columns={"team": "away_team"})
            chunk = chunk.merge(home, on="home_team", how="left", suffixes=("", ""))
            # Rename to _home
            for f in MOMENTUM_FEATURES:
                chunk.rename(columns={f: f"{f}_home"}, inplace=True)
            chunk = chunk.merge(away, on="away_team", how="left", suffixes=("", ""))
            for f in MOMENTUM_FEATURES:
                chunk.rename(columns={f: f"{f}_away"}, inplace=True)
            # Compute diffs
            for f in MOMENTUM_FEATURES:
                chunk[f"diff_{f}"] = chunk[f"{f}_home"] - chunk[f"{f}_away"]
        else:
            # No prior season data — fill with NaN
            for f in MOMENTUM_FEATURES:
                chunk[f"{f}_home"] = np.nan
                chunk[f"{f}_away"] = np.nan
                chunk[f"diff_{f}"] = np.nan

        mom_frames.append(chunk)

    return pd.concat(mom_frames, ignore_index=True).sort_values("date").reset_index(drop=True)


# =========================
# FORM (shifted rolling)
# =========================
def add_form_features(
    matches: pd.DataFrame,
    team_log: pd.DataFrame,
    model_stats: List[str],
    form_windows: List[int],
) -> pd.DataFrame:
    resolved_stats = resolve_model_stats(team_log, model_stats)

    cols_needed = ["match_url", "season", "Team", "is_home"] + resolved_stats
    missing = [c for c in cols_needed if c not in team_log.columns]
    if missing:
        raise ValueError(f"TEAM_LOG missing columns needed for form: {missing}")

    tg = team_log[cols_needed].copy()
    tg["Team"] = tg["Team"].map(norm_team)
    tg["date"] = tg["match_url"].apply(match_url_to_date)
    tg = tg.dropna(subset=["date"]).sort_values(["Team", "date"])

    for s in resolved_stats:
        tg[s] = pd.to_numeric(tg[s], errors="coerce")

    for w in form_windows:
        for s in resolved_stats:
            col = f"form_{s}_{w}"
            tg[col] = tg.groupby("Team")[s].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    form_cols = [f"form_{s}_{w}" for w in form_windows for s in resolved_stats]

    home = tg.loc[tg["is_home"].astype(int) == 1, ["match_url", "Team"] + form_cols].rename(columns={"Team": "home_team"})
    away = tg.loc[tg["is_home"].astype(int) == 0, ["match_url", "Team"] + form_cols].rename(columns={"Team": "away_team"})

    m = (
        matches
        .merge(home, on=["match_url", "home_team"], how="left")
        .merge(away, on=["match_url", "away_team"], how="left", suffixes=("_home", "_away"))
    )

    for w in form_windows:
        for s in resolved_stats:
            m[f"diff_form_{s}_{w}"] = m[f"form_{s}_{w}_home"] - m[f"form_{s}_{w}_away"]

    if len(form_windows) == 1:
        w = form_windows[0]
        for s in resolved_stats:
            m[f"diff_form_{s}"] = m[f"diff_form_{s}_{w}"]

    return m


def latest_form_snapshot(
    team_log: pd.DataFrame,
    model_stats: List[str],
    form_windows: List[int],
) -> pd.DataFrame:
    resolved_stats = resolve_model_stats(team_log, model_stats)

    cols_needed = ["match_url", "Team"] + resolved_stats
    missing = [c for c in cols_needed if c not in team_log.columns]
    if missing:
        raise ValueError(f"TEAM_LOG missing columns needed for form snapshot: {missing}")

    tg = team_log[cols_needed].copy()
    tg["Team"] = tg["Team"].map(norm_team)
    tg["date"] = tg["match_url"].apply(match_url_to_date)
    tg = tg.dropna(subset=["date"]).sort_values(["Team", "date"])

    for s in resolved_stats:
        tg[s] = pd.to_numeric(tg[s], errors="coerce")

    for w in form_windows:
        for s in resolved_stats:
            col = f"form_{s}_{w}"
            tg[col] = tg.groupby("Team")[s].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    form_cols = [f"form_{s}_{w}" for w in form_windows for s in resolved_stats]

    snap = (
        tg.sort_values(["Team", "date"])
          .groupby("Team", as_index=False)[form_cols]
          .last()
          .rename(columns={"Team": "team"})
    )
    return snap


def add_fixture_features(
    fixtures: pd.DataFrame,
    elos: Dict[str, float],
    form_snap: pd.DataFrame,
    team_log: pd.DataFrame,
    model_stats: List[str],
    form_windows: List[int],
    elo_base: float,
    momentum_snap: Optional[pd.DataFrame] = None,
    list_priors_snap: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    fx = fixtures.copy()

    fx["elo_home_pre"] = fx["home_team"].map(lambda t: elos.get(t, float(elo_base)))
    fx["elo_away_pre"] = fx["away_team"].map(lambda t: elos.get(t, float(elo_base)))
    fx["elo_diff"] = fx["elo_home_pre"] - fx["elo_away_pre"]

    home = form_snap.rename(columns={"team": "home_team"})
    away = form_snap.rename(columns={"team": "away_team"})
    fx = fx.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left", suffixes=("_home", "_away"))

    resolved_stats = resolve_model_stats(team_log, model_stats)
    for w in form_windows:
        for s in resolved_stats:
            col = f"form_{s}_{w}"
            fx[f"diff_form_{s}_{w}"] = fx[f"{col}_home"] - fx[f"{col}_away"]

    if len(form_windows) == 1:
        w = form_windows[0]
        for s in resolved_stats:
            fx[f"diff_form_{s}"] = fx[f"diff_form_{s}_{w}"]

    # Merge momentum features if provided
    if momentum_snap is not None and len(momentum_snap) > 0:
        mom_home = momentum_snap.rename(columns={"team": "home_team"})
        mom_away = momentum_snap.rename(columns={"team": "away_team"})
        fx = fx.merge(mom_home, on="home_team", how="left")
        for f in MOMENTUM_FEATURES:
            fx.rename(columns={f: f"{f}_home"}, inplace=True)
        fx = fx.merge(mom_away, on="away_team", how="left")
        for f in MOMENTUM_FEATURES:
            fx.rename(columns={f: f"{f}_away"}, inplace=True)
        for f in MOMENTUM_FEATURES:
            fx[f"diff_{f}"] = fx[f"{f}_home"] - fx[f"{f}_away"]

    # Merge list priors if provided
    if list_priors_snap is not None and len(list_priors_snap) > 0:
        lp_home = list_priors_snap.rename(columns={"team": "home_team", "avg_age": "list_avg_age_home"})
        lp_away = list_priors_snap.rename(columns={"team": "away_team", "avg_age": "list_avg_age_away"})
        fx = fx.merge(lp_home[["home_team", "list_avg_age_home"]], on="home_team", how="left")
        fx = fx.merge(lp_away[["away_team", "list_avg_age_away"]], on="away_team", how="left")
        fx["diff_list_avg_age"] = fx["list_avg_age_home"] - fx["list_avg_age_away"]

    return fx


# =========================
# MODEL + CALIBRATION
# =========================
def fit_model(train_df: pd.DataFrame, feats: List[str]) -> Pipeline:
    X = train_df[feats]
    y = train_df["home_win"].astype(int)

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, C=0.2, solver="lbfgs")),
    ])
    model.fit(X, y)
    return model


def predict_proba(model: Pipeline, df: pd.DataFrame, feats: List[str]) -> np.ndarray:
    return model.predict_proba(df[feats])[:, 1]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt_calibrator(p: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    z = logit(p).reshape(-1, 1)
    clf = LogisticRegression(max_iter=500, C=1e6)
    clf.fit(z, y.astype(int))
    a = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    return a, b


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    z = logit(p)
    return inv_logit(a * z + b)


# =========================
# EVALUATION HELPERS (prob metrics + tip accuracy)
# =========================
def metrics_block(df: pd.DataFrame, pcol: str, label: str) -> dict:
    y = df["home_win"].astype(int).values
    p = df[pcol].astype(float).clip(1e-6, 1 - 1e-6).values
    return {
        "source": label,
        "n": int(len(df)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "acc@0.5": float(((p >= 0.5).astype(int) == y).mean()),
    }


def evaluate_tipping(df: pd.DataFrame, pcol: str, label: str) -> dict:
    y = df["home_win"].astype(int).values
    p = df[pcol].astype(float).values
    tips = (p >= 0.5).astype(int)
    return {
        "strategy": label,
        "n": int(len(df)),
        "accuracy": float((tips == y).mean()),
        "correct": int((tips == y).sum()),
    }


# =========================
# HIGH-LEVEL PIPELINE STEPS
# =========================
def build_historical_feature_frame(
    team_log: pd.DataFrame,
    model_stats: List[str],
    form_windows: List[int],
    elo_k: float,
    elo_home_adv: float,
    elo_base: float,
    use_momentum: bool = False,
    use_list_priors: bool = False,
    list_priors_path: str = "data/team_season_list_priors.csv",
) -> pd.DataFrame:
    matches = build_match_level(team_log)
    hist = matches.sort_values("date").reset_index(drop=True)
    hist = compute_elo_diff(hist, k=elo_k, home_adv=elo_home_adv, base=elo_base)
    hist = add_form_features(hist, team_log, model_stats=model_stats, form_windows=form_windows)
    if use_momentum:
        hist = add_preseason_momentum(hist)
    if use_list_priors:
        priors = load_list_priors(list_priors_path)
        hist = add_list_priors(hist, priors)
    return hist


def make_feature_list(
    team_log: pd.DataFrame,
    model_stats: List[str],
    form_windows: List[int],
    use_momentum: bool = False,
    use_list_priors: bool = False,
) -> List[str]:
    feats = ["elo_diff"]
    resolved_stats = resolve_model_stats(team_log, model_stats)

    if len(form_windows) == 1:
        for s in resolved_stats:
            feats.append(f"diff_form_{s}")
    else:
        for w in form_windows:
            for s in resolved_stats:
                feats.append(f"diff_form_{s}_{w}")

    if use_momentum:
        for f in MOMENTUM_FEATURES:
            feats.append(f"diff_{f}")

    if use_list_priors:
        for f in LIST_PRIOR_FEATURES:
            feats.append(f"diff_{f}")

    return feats


def train_and_calibrate(
    hist: pd.DataFrame,
    feats: List[str],
    train_through_season: int,
    calib_season: int,
) -> Tuple[Pipeline, Tuple[float, float]]:
    train_df = hist[hist["season"] <= train_through_season].copy()
    calib_df = hist[hist["season"] == calib_season].copy()

    model = fit_model(train_df, feats)

    calib_df["p_raw"] = predict_proba(model, calib_df, feats)
    a, b = fit_platt_calibrator(calib_df["p_raw"].values, calib_df["home_win"].values)

    return model, (a, b)
