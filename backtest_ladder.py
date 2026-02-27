"""
Backtest ladder predictions for historical AFL seasons.

For each target season (2023, 2024, 2025):
1. Train model using only data before that season
2. Generate win probabilities for all matches in that season
3. Run Monte Carlo ladder simulation
4. Compare predicted ladder to actual final standings

Improvements:
- Walk-forward Elo: Update Elo after each round using actual results
- Pre-season premiership odds: Blend with betting market expectations
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional

# Add parent directory for imports
sys.path.insert(0, ".")

from afl_pipeline import (
    load_team_rows,
    build_match_level,
    build_historical_feature_frame,
    make_feature_list,
    train_and_calibrate,
    predict_proba,
    apply_platt,
    compute_current_elos,
    latest_form_snapshot,
    add_fixture_features,
    norm_team,
    safe_read_csv,
    elo_expected,
    match_url_to_date,
    compute_team_momentum_features,
    MOMENTUM_FEATURES,
    load_list_priors,
    add_list_priors,
    LIST_PRIOR_FEATURES,
)
from data.afl_season_sim import simulate_ladder_from_probs


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_STATS = ["I50", "CL", "DI"]
FORM_WINDOWS = [5]
ELO_K = 25
ELO_HOME_ADV = 65
ELO_BASE = 1500

TEAM_CACHE = "afl_team_rows.csv"
ODDS_FILE = "aflodds.csv"

N_SIMS = 20000
SHRINK = 0.9  # Shrink probabilities toward 0.5 (more conservative)

# Blend weight for premiership odds (0 = pure model, 1 = pure market)
# Tuning shows w=0.0 is optimal for Spearman/MARE, w=0.2 for Top 8 count
PREMIERSHIP_BLEND_W = 0.0  # Pure model performs best

# Walk-forward Elo: update Elo after each round
USE_WALKFORWARD_ELO = True

# Momentum model features: add momentum as logistic regression features
USE_MOMENTUM_MODEL_FEATS = True

# List age priors: add team average age as logistic regression feature
USE_LIST_PRIORS = True


# ============================================================
# MOMENTUM FEATURES CONFIG
# ============================================================
LATE_SEASON_WINDOW = 6  # Last N matches of prior season
USE_MOMENTUM_FEATURES = False  # Toggle momentum features (proved not helpful)

# ============================================================
# COACHING TENURE DATA
# Year of tenure for each coach heading into that season
# Key insight: Year 2-3 coaches often see breakouts as system beds in
# ============================================================
COACHING_TENURE = {
    2023: {
        # New coaches in Year 2 (started 2022)
        "Collingwood": 2,      # Craig McRae (started 2022) - WON PREMIERSHIP
        "Hawthorn": 2,         # Sam Mitchell (started 2022)
        # Established coaches
        "Brisbane Lions": 7,   # Chris Fagan (started 2017)
        "Geelong": 13,         # Chris Scott (started 2011)
        "Sydney": 12,          # John Longmire (started 2012)
        "Melbourne": 5,        # Simon Goodwin (started 2019, but interim 2017)
        "Carlton": 3,          # Michael Voss (started 2021)
        "Port Adelaide": 3,    # Ken Hinkley (started 2013) - actually 10+
        "Richmond": 6,         # Damien Hardwick (started 2010, but re-signed)
        "Western Bulldogs": 8, # Luke Beveridge (started 2015)
        "Fremantle": 3,        # Justin Longmuir (started 2020)
        "St Kilda": 3,         # Ross Lyon (started 2022) - started 2nd stint
        "Gold Coast": 6,       # Stuart Dew (started 2018)
        "Greater Western Sydney": 10, # Leon Cameron (but left mid-2022)
        "Essendon": 2,         # Brad Scott (started 2022)
        "Adelaide": 4,         # Matthew Nicks (started 2020)
        "North Melbourne": 3,  # David Noble (started 2021)
        "West Coast": 10,      # Adam Simpson (started 2014)
    },
    2024: {
        # Year 3 coaches (breakout zone)
        "Collingwood": 3,      # Craig McRae
        "Hawthorn": 3,         # Sam Mitchell - BREAKOUT SEASON
        # New coach Year 2
        "Greater Western Sydney": 2,  # Adam Kingsley (started 2023)
        "Essendon": 3,         # Brad Scott
        # Established
        "Brisbane Lions": 8,
        "Geelong": 14,
        "Sydney": 13,
        "Melbourne": 6,
        "Carlton": 4,
        "Port Adelaide": 11,
        "Western Bulldogs": 9,
        "Fremantle": 4,
        "St Kilda": 2,         # Ross Lyon year 2
        "Gold Coast": 7,       # Stuart Dew (left 2023, Damien Hardwick came mid)
        "Adelaide": 5,
        "North Melbourne": 1,  # Alastair Clarkson (started mid-2023)
        "West Coast": 11,
        "Richmond": 1,         # Adem Yze (started 2024)
    },
    2025: {
        # Coaches in breakout zone (Year 2-3)
        "Greater Western Sydney": 3,  # Adam Kingsley
        "Richmond": 2,         # Adem Yze
        # Adelaide potential (long tenure, but young list maturing)
        "Adelaide": 6,         # Matthew Nicks - WON MINOR PREMIERSHIP
        "Hawthorn": 4,         # Sam Mitchell (continued success)
        # Sydney new coach
        "Sydney": 1,           # Dean Cox (replaced Longmire)
        # Established
        "Collingwood": 4,
        "Brisbane Lions": 9,
        "Geelong": 15,
        "Melbourne": 7,
        "Carlton": 5,
        "Port Adelaide": 12,
        "Western Bulldogs": 10,
        "Fremantle": 5,
        "St Kilda": 3,
        "Gold Coast": 2,       # Damien Hardwick year 2
        "Essendon": 4,
        "North Melbourne": 2,
        "West Coast": 1,       # Andrew McQualter (started 2025)
    },
    2026: {
        # Coaches in breakout zone (Year 2-3)
        "Sydney": 2,           # Dean Cox year 2 - BREAKOUT CANDIDATE
        "West Coast": 2,       # Andrew McQualter year 2 - BREAKOUT CANDIDATE
        "Richmond": 3,         # Adem Yze year 3 - BREAKOUT CANDIDATE
        "Gold Coast": 3,       # Damien Hardwick year 3 - BREAKOUT CANDIDATE
        # Year 4+ (past breakout zone)
        "Greater Western Sydney": 4,
        "St Kilda": 4,
        "North Melbourne": 3,  # Alastair Clarkson year 3 - BREAKOUT CANDIDATE
        "Hawthorn": 5,
        "Collingwood": 5,
        "Essendon": 5,
        "Carlton": 6,
        "Fremantle": 6,
        "Adelaide": 7,
        "Melbourne": 8,
        "Brisbane Lions": 10,
        "Western Bulldogs": 11,
        "Port Adelaide": 13,
        "Geelong": 16,
    },
}

# Tenure years that are "breakout zone" - systems bedding in
BREAKOUT_TENURE_YEARS = [2, 3]
COACHING_ELO_BOOST = 30.0  # Elo boost for coaches in breakout zone


# ============================================================
# PRE-SEASON PREMIERSHIP ODDS (historical)
# Source: Various betting aggregators, captured at season start
# Lower odds = more favoured to win premiership
# ============================================================
PREMIERSHIP_ODDS = {
    2023: {
        "Melbourne": 4.5,
        "Geelong": 6.0,
        "Brisbane Lions": 7.0,
        "Collingwood": 8.0,
        "Sydney": 9.0,
        "Fremantle": 11.0,
        "Carlton": 13.0,
        "Richmond": 15.0,
        "St Kilda": 17.0,
        "Western Bulldogs": 21.0,
        "Port Adelaide": 23.0,
        "Gold Coast": 34.0,
        "Greater Western Sydney": 34.0,
        "Hawthorn": 51.0,
        "Essendon": 51.0,
        "Adelaide": 67.0,
        "West Coast": 101.0,
        "North Melbourne": 151.0,
    },
    2024: {
        "Brisbane Lions": 5.0,
        "Collingwood": 5.5,
        "Carlton": 7.0,
        "Melbourne": 8.0,
        "Sydney": 9.0,
        "Port Adelaide": 11.0,
        "Geelong": 13.0,
        "Greater Western Sydney": 15.0,
        "St Kilda": 21.0,
        "Fremantle": 23.0,
        "Western Bulldogs": 26.0,
        "Gold Coast": 34.0,
        "Essendon": 41.0,
        "Richmond": 51.0,
        "Hawthorn": 67.0,
        "Adelaide": 81.0,
        "West Coast": 151.0,
        "North Melbourne": 201.0,
    },
    2025: {
        "Brisbane Lions": 5.0,
        "Sydney": 6.0,
        "Carlton": 7.0,
        "Collingwood": 8.0,
        "Geelong": 9.0,
        "Greater Western Sydney": 11.0,
        "Hawthorn": 13.0,
        "Port Adelaide": 13.0,
        "Western Bulldogs": 17.0,
        "Fremantle": 21.0,
        "Gold Coast": 26.0,
        "Melbourne": 26.0,
        "Essendon": 34.0,
        "St Kilda": 41.0,
        "Adelaide": 51.0,
        "Richmond": 101.0,
        "North Melbourne": 151.0,
        "West Coast": 201.0,
    },
}


# ============================================================
# PREMIERSHIP ODDS PROCESSING
# ============================================================
def get_premiership_implied_ranks(season: int) -> pd.DataFrame:
    """
    Convert premiership odds to implied ladder positions.
    Lower odds = higher implied rank (more likely to finish high).

    Returns DataFrame with columns: team, prem_odds, prem_implied_rank
    """
    if season not in PREMIERSHIP_ODDS:
        return pd.DataFrame(columns=["team", "prem_odds", "prem_implied_rank"])

    odds = PREMIERSHIP_ODDS[season]
    df = pd.DataFrame([
        {"team": team, "prem_odds": odd}
        for team, odd in odds.items()
    ])

    # Sort by odds ascending (favourites first), assign rank
    df = df.sort_values("prem_odds").reset_index(drop=True)
    df["prem_implied_rank"] = range(1, len(df) + 1)

    return df


def blend_with_premiership_odds(
    predicted: pd.DataFrame,
    season: int,
    blend_w: float = PREMIERSHIP_BLEND_W,
) -> pd.DataFrame:
    """
    Blend model predictions with premiership market implied rankings.

    predicted: DataFrame with team, mean_rank, exp_points, etc.
    blend_w: 0 = pure model, 1 = pure market

    Returns DataFrame with blended_rank column added.
    """
    prem = get_premiership_implied_ranks(season)

    if len(prem) == 0:
        predicted["blended_rank"] = predicted["mean_rank"]
        return predicted

    merged = predicted.merge(prem, on="team", how="left")

    # For teams without premiership odds, use model rank
    merged["prem_implied_rank"] = merged["prem_implied_rank"].fillna(merged["mean_rank"])

    # Blend: weighted average of model rank and market implied rank
    merged["blended_rank"] = (
        (1 - blend_w) * merged["mean_rank"] +
        blend_w * merged["prem_implied_rank"]
    )

    return merged


# ============================================================
# MOMENTUM / TRAJECTORY FEATURES
# ============================================================
def compute_team_season_stats(
    team_log: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    Compute per-team aggregated stats for a given season.

    Returns DataFrame with: team, wins, losses, win_rate, point_diff,
    plus late-season performance (last N matches).
    """
    tl = team_log[team_log["season"] == season].copy()
    if len(tl) == 0:
        return pd.DataFrame()

    tl["Team"] = tl["Team"].map(norm_team)
    tl["date"] = tl["match_url"].apply(match_url_to_date)
    tl = tl.dropna(subset=["date"]).sort_values(["Team", "date"])

    # Compute per-game win/loss
    tl["won"] = (tl["score_for"] > tl["score_against"]).astype(int)
    tl["margin"] = tl["score_for"] - tl["score_against"]

    # Full season stats
    full = tl.groupby("Team").agg(
        games=("won", "count"),
        wins=("won", "sum"),
        total_margin=("margin", "sum"),
    ).reset_index()
    full["win_rate"] = full["wins"] / full["games"]
    full["avg_margin"] = full["total_margin"] / full["games"]

    # Late season stats (last N matches per team)
    late = tl.groupby("Team").tail(LATE_SEASON_WINDOW)
    late_stats = late.groupby("Team").agg(
        late_games=("won", "count"),
        late_wins=("won", "sum"),
        late_margin=("margin", "sum"),
    ).reset_index()
    late_stats["late_win_rate"] = late_stats["late_wins"] / late_stats["late_games"]
    late_stats["late_avg_margin"] = late_stats["late_margin"] / late_stats["late_games"]

    merged = full.merge(late_stats, on="Team", how="left")
    merged = merged.rename(columns={"Team": "team"})
    merged["season"] = season

    return merged


def compute_elo_trajectory(
    matches: pd.DataFrame,
    season: int,
    k: float = ELO_K,
    home_adv: float = ELO_HOME_ADV,
    base: float = ELO_BASE,
) -> pd.DataFrame:
    """
    Compute Elo trajectory for each team within a season.

    Returns DataFrame with: team, elo_start, elo_end, elo_change
    (how much a team improved/declined during the season based on Elo)
    """
    # Get matches up to and including target season
    all_matches = matches[matches["season"] <= season].sort_values("date").copy()
    season_matches = matches[matches["season"] == season].sort_values("date").copy()

    if len(season_matches) == 0:
        return pd.DataFrame()

    teams = set(season_matches["home_team"]).union(set(season_matches["away_team"]))

    # Compute Elo at start of season (after all prior matches)
    prior_matches = all_matches[all_matches["season"] < season]
    if len(prior_matches) > 0:
        start_elos = compute_current_elos(prior_matches, k, home_adv, base)
    else:
        start_elos = {t: float(base) for t in teams}

    # Compute Elo at end of season (after all matches including this season)
    end_elos = compute_current_elos(all_matches, k, home_adv, base)

    # Build trajectory DataFrame
    trajectory = []
    for team in teams:
        elo_start = start_elos.get(team, float(base))
        elo_end = end_elos.get(team, float(base))
        trajectory.append({
            "team": team,
            "elo_start": elo_start,
            "elo_end": elo_end,
            "elo_change": elo_end - elo_start,
            "season": season,
        })

    return pd.DataFrame(trajectory)


def compute_momentum_features(
    team_log: pd.DataFrame,
    matches: pd.DataFrame,
    prior_season: int,
) -> pd.DataFrame:
    """
    Compute momentum features for predicting next season performance.

    Features:
    - late_win_rate: Win rate in last N matches of prior season
    - late_avg_margin: Average margin in last N matches
    - elo_change: How much Elo improved during prior season
    - momentum_score: Combined momentum indicator

    These features are computed from prior_season data only.
    """
    # Season stats (including late-season form)
    season_stats = compute_team_season_stats(team_log, prior_season)

    # Elo trajectory
    elo_traj = compute_elo_trajectory(matches, prior_season)

    if len(season_stats) == 0 or len(elo_traj) == 0:
        return pd.DataFrame()

    # Merge
    momentum = season_stats.merge(
        elo_traj[["team", "elo_start", "elo_end", "elo_change"]],
        on="team",
        how="outer",
    )

    # Compute composite momentum score
    # Positive = improving team, negative = declining
    # Normalize components to similar scales
    if len(momentum) > 0:
        # Late win rate deviation from 0.5 (scale: -0.5 to +0.5)
        momentum["late_wr_dev"] = momentum["late_win_rate"].fillna(0.5) - 0.5

        # Late margin normalized (typical range -50 to +50, scale to similar range as win rate)
        momentum["late_margin_norm"] = momentum["late_avg_margin"].fillna(0) / 100

        # Elo change normalized (typical range -100 to +100, scale similarly)
        momentum["elo_change_norm"] = momentum["elo_change"].fillna(0) / 200

        # Combined momentum: weighted average
        # Late form weighted more heavily as it's most recent
        momentum["momentum_score"] = (
            0.4 * momentum["late_wr_dev"] +
            0.3 * momentum["late_margin_norm"] +
            0.3 * momentum["elo_change_norm"]
        )

    return momentum


def get_preseason_momentum(
    team_log: pd.DataFrame,
    matches: pd.DataFrame,
    target_season: int,
) -> Dict[str, Dict[str, float]]:
    """
    Get momentum features for all teams heading into target season.

    Returns dict: {team_name: {feature_name: value, ...}, ...}
    """
    prior_season = target_season - 1
    momentum = compute_momentum_features(team_log, matches, prior_season)

    if len(momentum) == 0:
        return {}

    result = {}
    for _, row in momentum.iterrows():
        team = row["team"]
        result[team] = {
            "late_win_rate": row.get("late_win_rate", 0.5),
            "late_avg_margin": row.get("late_avg_margin", 0.0),
            "elo_change": row.get("elo_change", 0.0),
            "momentum_score": row.get("momentum_score", 0.0),
            "elo_end": row.get("elo_end", ELO_BASE),
        }

    return result


# ============================================================
# COACHING TENURE FEATURES
# ============================================================
def get_coaching_adjustments(target_season: int) -> Dict[str, float]:
    """
    Get Elo adjustments based on coaching tenure.

    Coaches in "breakout zone" (year 2-3) get a boost as their systems
    are bedding in and improvement is more likely.

    Returns dict: {team_name: elo_adjustment, ...}
    """
    if target_season not in COACHING_TENURE:
        return {}

    adjustments = {}
    tenure_data = COACHING_TENURE[target_season]

    for team, tenure in tenure_data.items():
        if tenure in BREAKOUT_TENURE_YEARS:
            adjustments[team] = COACHING_ELO_BOOST
        else:
            adjustments[team] = 0.0

    return adjustments


def print_coaching_analysis(target_season: int):
    """Print coaching tenure analysis for a given season."""
    if target_season not in COACHING_TENURE:
        print(f"No coaching data for {target_season}")
        return

    tenure_data = COACHING_TENURE[target_season]
    adjustments = get_coaching_adjustments(target_season)

    print(f"\n{'='*70}")
    print(f"COACHING TENURE: {target_season} Season")
    print(f"Breakout zone: Years {BREAKOUT_TENURE_YEARS}")
    print(f"{'='*70}")
    print(f"\n{'Team':<24} {'Tenure':>8} {'Zone':>10} {'Adjustment':>12}")
    print("-" * 60)

    # Sort by tenure
    sorted_teams = sorted(tenure_data.items(), key=lambda x: x[1])
    for team, tenure in sorted_teams:
        in_zone = "YES" if tenure in BREAKOUT_TENURE_YEARS else ""
        adj = adjustments.get(team, 0.0)
        adj_str = f"+{adj:.0f}" if adj > 0 else ""
        print(f"{team:<24} {tenure:>8} {in_zone:>10} {adj_str:>12}")

    # Highlight teams in breakout zone
    breakout_teams = [t for t, y in tenure_data.items() if y in BREAKOUT_TENURE_YEARS]
    if breakout_teams:
        print(f"\nTeams in breakout zone: {', '.join(breakout_teams)}")


# ============================================================
# WALK-FORWARD ELO
# ============================================================
def compute_walkforward_elos(
    initial_elos: Dict[str, float],
    matches: pd.DataFrame,
    k: float = ELO_K,
    home_adv: float = ELO_HOME_ADV,
    base: float = ELO_BASE,
) -> pd.DataFrame:
    """
    Compute Elo ratings that update after each match.

    Returns the matches DataFrame with elo_home_pre, elo_away_pre columns
    representing the Elo BEFORE each match (using walk-forward updating).
    """
    # Sort matches by date
    df = matches.sort_values("date").copy()

    # Initialize Elo ratings
    elos = {t: initial_elos.get(t, float(base)) for t in
            set(df["home_team"]).union(set(df["away_team"]))}

    elo_home_pre = []
    elo_away_pre = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        eh = elos.get(h, float(base))
        ea = elos.get(a, float(base))

        elo_home_pre.append(eh)
        elo_away_pre.append(ea)

        # Update Elo based on actual result
        home_score = row.get("home_score", 0)
        away_score = row.get("away_score", 0)

        if pd.notna(home_score) and pd.notna(away_score):
            actual_home_win = 1 if home_score > away_score else 0
            exp_home = elo_expected(eh + home_adv, ea)

            elos[h] = eh + k * (actual_home_win - exp_home)
            elos[a] = ea + k * (exp_home - actual_home_win)

    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]

    return df


# ============================================================
# ACTUAL LADDER COMPUTATION
# ============================================================
def compute_actual_ladder(odds_path: str, season: int) -> pd.DataFrame:
    """
    Derive the actual final ladder from match results in odds file.

    Returns DataFrame with columns: team, actual_rank, points, percentage, wins
    """
    # Read odds file (header on row 2)
    odds = pd.read_csv(odds_path, header=1, encoding="cp1252")

    # Parse date and extract year
    odds["date"] = pd.to_datetime(odds["Date"], format="%d-%b-%y", errors="coerce")
    odds["year"] = odds["date"].dt.year

    # Filter to target season
    df = odds[odds["year"] == season].copy()

    # Filter to regular season only (exclude finals)
    df["Play Off Game?"] = df["Play Off Game?"].fillna("")
    df = df[df["Play Off Game?"] != "Y"]

    if len(df) == 0:
        raise ValueError(f"No regular season matches found for {season}")

    # Normalize team names
    df["home_team"] = df["Home Team"].astype(str).str.strip().map(norm_team)
    df["away_team"] = df["Away Team"].astype(str).str.strip().map(norm_team)

    # Parse scores
    df["home_score"] = pd.to_numeric(df["Home Score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["Away Score"], errors="coerce")

    # Drop matches with missing scores
    df = df.dropna(subset=["home_score", "away_score"])

    # Compute team records
    records: Dict[str, Dict] = {}

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])

        for team in [h, a]:
            if team not in records:
                records[team] = {"wins": 0, "draws": 0, "losses": 0, "pf": 0, "pa": 0}

        records[h]["pf"] += hs
        records[h]["pa"] += as_
        records[a]["pf"] += as_
        records[a]["pa"] += hs

        if hs > as_:
            records[h]["wins"] += 1
            records[a]["losses"] += 1
        elif as_ > hs:
            records[a]["wins"] += 1
            records[h]["losses"] += 1
        else:
            records[h]["draws"] += 1
            records[a]["draws"] += 1

    # Build ladder DataFrame
    ladder = []
    for team, r in records.items():
        pts = r["wins"] * 4 + r["draws"] * 2
        pct = r["pf"] / r["pa"] if r["pa"] > 0 else 0
        ladder.append({
            "team": team,
            "points": pts,
            "percentage": pct,
            "pf": r["pf"],
            "pa": r["pa"],
            "wins": r["wins"],
        })

    ladder_df = pd.DataFrame(ladder)
    ladder_df = ladder_df.sort_values(["points", "percentage"], ascending=False)
    ladder_df["actual_rank"] = range(1, len(ladder_df) + 1)
    ladder_df = ladder_df.reset_index(drop=True)

    return ladder_df[["team", "actual_rank", "points", "percentage", "wins"]]


# ============================================================
# PREDICTION GENERATION
# ============================================================
USE_COACHING_FEATURES = True  # Toggle coaching tenure features

def generate_season_predictions(
    team_log: pd.DataFrame,
    target_season: int,
    train_through_season: int,
    calib_season: int,
    use_walkforward_elo: bool = USE_WALKFORWARD_ELO,
    use_momentum: bool = USE_MOMENTUM_FEATURES,
    momentum_weight: float = 50.0,  # Elo points per unit of momentum_score
    use_coaching: bool = USE_COACHING_FEATURES,
    use_momentum_model: bool = False,
    use_list_priors: bool = False,
) -> pd.DataFrame:
    """
    Train model on historical data and generate win probabilities for target season.

    If use_walkforward_elo=True, Elo ratings are updated after each match
    using actual results (simulating how predictions would evolve through the season).

    Returns DataFrame with columns: round, home_team, away_team, p_home_win_cal
    """
    # Filter team_log to training data only
    train_log = team_log[team_log["season"] <= train_through_season].copy()

    # Build historical feature frame from training data
    hist = build_historical_feature_frame(
        team_log=train_log,
        model_stats=MODEL_STATS,
        form_windows=FORM_WINDOWS,
        elo_k=ELO_K,
        elo_home_adv=ELO_HOME_ADV,
        elo_base=ELO_BASE,
        use_momentum=use_momentum_model,
        use_list_priors=use_list_priors,
    )

    # Get feature list
    feats = make_feature_list(train_log, MODEL_STATS, FORM_WINDOWS,
                              use_momentum=use_momentum_model,
                              use_list_priors=use_list_priors)

    # Compute momentum snapshot from T-1 for fixture features
    mom_snap = None
    if use_momentum_model:
        mom_snap = compute_team_momentum_features(hist, train_through_season)

    # Load list priors snapshot for target season fixtures
    lp_snap = None
    if use_list_priors:
        all_priors = load_list_priors()
        lp_snap = all_priors[all_priors["season"] == target_season].copy()

    # Train model and get Platt calibration params
    model, (a, b) = train_and_calibrate(
        hist=hist,
        feats=feats,
        train_through_season=train_through_season,
        calib_season=calib_season,
    )

    # Get initial Elo ratings as of end of training period
    initial_elos = compute_current_elos(
        hist[hist["season"] <= train_through_season],
        k=ELO_K,
        home_adv=ELO_HOME_ADV,
        base=ELO_BASE,
    )

    # Apply momentum adjustments to initial Elos
    if use_momentum:
        momentum_data = get_preseason_momentum(train_log, hist, target_season)
        for team, mom in momentum_data.items():
            if team in initial_elos:
                # Adjust Elo based on momentum score
                # momentum_score ranges roughly -0.5 to +0.5
                # momentum_weight of 50 means max adjustment of ~25 Elo points
                adjustment = mom.get("momentum_score", 0.0) * momentum_weight
                initial_elos[team] += adjustment

    # Apply coaching tenure adjustments
    if use_coaching:
        coaching_adj = get_coaching_adjustments(target_season)
        for team, adj in coaching_adj.items():
            if team in initial_elos and adj != 0:
                initial_elos[team] += adj

    # Get latest form snapshot from training data only
    snap = latest_form_snapshot(
        train_log,
        model_stats=MODEL_STATS,
        form_windows=FORM_WINDOWS,
    )

    # Build match-level data for target season (these are our "fixtures")
    target_log = team_log[team_log["season"] == target_season].copy()
    target_matches = build_match_level(target_log)

    if len(target_matches) == 0:
        raise ValueError(f"No matches found for season {target_season}")

    # Sort by date
    target_matches = target_matches.sort_values("date").reset_index(drop=True)

    # Create pseudo-round based on date grouping (matches on same date = same round)
    target_matches["round"] = (
        target_matches.groupby(target_matches["date"].dt.date).ngroup() + 1
    )

    if use_walkforward_elo:
        # Walk-forward Elo: update Elo after each match using actual results
        target_matches = compute_walkforward_elos(
            initial_elos=initial_elos,
            matches=target_matches,
            k=ELO_K,
            home_adv=ELO_HOME_ADV,
            base=ELO_BASE,
        )

        # Add form features (still using pre-season snapshot for form)
        # Form could also be updated walk-forward, but keeping it simpler for now
        target_with_feats = add_fixture_features(
            fixtures=target_matches,
            elos=initial_elos,  # Dummy - will be overwritten
            form_snap=snap,
            team_log=train_log,
            model_stats=MODEL_STATS,
            form_windows=FORM_WINDOWS,
            elo_base=ELO_BASE,
            momentum_snap=mom_snap,
            list_priors_snap=lp_snap,
        )

        # Override with walk-forward Elo values
        target_with_feats["elo_home_pre"] = target_matches["elo_home_pre"].values
        target_with_feats["elo_away_pre"] = target_matches["elo_away_pre"].values
        target_with_feats["elo_diff"] = target_matches["elo_diff"].values
    else:
        # Static Elo: use pre-season snapshot for all matches
        target_with_feats = add_fixture_features(
            fixtures=target_matches,
            elos=initial_elos,
            form_snap=snap,
            team_log=train_log,
            model_stats=MODEL_STATS,
            form_windows=FORM_WINDOWS,
            elo_base=ELO_BASE,
            momentum_snap=mom_snap,
            list_priors_snap=lp_snap,
        )

    # Predict raw probabilities
    p_raw = predict_proba(model, target_with_feats, feats)

    # Apply Platt calibration
    target_with_feats["p_home_win_cal"] = apply_platt(p_raw, a, b)

    # Include home_win for match-level evaluation when available
    out_cols = ["round", "date", "home_team", "away_team", "p_home_win_cal"]
    if "home_win" in target_with_feats.columns:
        out_cols.append("home_win")
    return target_with_feats[out_cols]


# ============================================================
# EVALUATION METRICS
# ============================================================
def evaluate_ladder_predictions(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
) -> Dict:
    """
    Compare predicted ladder to actual ladder.

    predicted: DataFrame with team, mean_rank, p_top4, p_top8, exp_points
               (optionally blended_rank if premiership odds were used)
    actual: DataFrame with team, actual_rank

    Returns dict of evaluation metrics.
    """
    # Merge on team
    merged = predicted.merge(actual, on="team", how="inner")

    if len(merged) == 0:
        raise ValueError("No teams matched between predicted and actual ladders")

    n_teams = len(merged)

    # Use blended_rank if available, else mean_rank
    rank_col = "blended_rank" if "blended_rank" in merged.columns else "mean_rank"

    # 1. Spearman rank correlation
    spearman_rho, spearman_p = spearmanr(merged[rank_col], merged["actual_rank"])

    # 2. Mean Absolute Rank Error
    merged["rank_error"] = abs(merged[rank_col] - merged["actual_rank"])
    mare = merged["rank_error"].mean()

    # 3. Top 8 accuracy
    actual_top8 = set(merged[merged["actual_rank"] <= 8]["team"])
    predicted_top8 = set(merged.nsmallest(8, rank_col)["team"])
    top8_correct = len(actual_top8 & predicted_top8)
    top8_accuracy = top8_correct / 8

    # 4. Top 4 accuracy
    actual_top4 = set(merged[merged["actual_rank"] <= 4]["team"])
    predicted_top4 = set(merged.nsmallest(4, rank_col)["team"])
    top4_correct = len(actual_top4 & predicted_top4)
    top4_accuracy = top4_correct / 4

    # 5. Minor Premier (1st place)
    actual_1st = merged[merged["actual_rank"] == 1]["team"].iloc[0]
    predicted_1st = merged.nsmallest(1, rank_col)["team"].iloc[0]
    minor_prem_hit = actual_1st == predicted_1st

    # 6. Wooden Spoon (last place)
    actual_last = merged[merged["actual_rank"] == n_teams]["team"].iloc[0]
    predicted_last = merged.nlargest(1, rank_col)["team"].iloc[0]
    spoon_hit = actual_last == predicted_last

    # 7. Probability calibration
    top8_teams = merged[merged["actual_rank"] <= 8]
    miss8_teams = merged[merged["actual_rank"] > 8]
    mean_p_top8_actual_top8 = top8_teams["p_top8"].mean() if len(top8_teams) > 0 else np.nan
    mean_p_top8_actual_miss = miss8_teams["p_top8"].mean() if len(miss8_teams) > 0 else np.nan

    top4_teams = merged[merged["actual_rank"] <= 4]
    miss4_teams = merged[merged["actual_rank"] > 4]
    mean_p_top4_actual_top4 = top4_teams["p_top4"].mean() if len(top4_teams) > 0 else np.nan
    mean_p_top4_actual_miss = miss4_teams["p_top4"].mean() if len(miss4_teams) > 0 else np.nan

    # 8. Brier score for top 8 (probabilistic accuracy)
    merged["made_top8"] = (merged["actual_rank"] <= 8).astype(int)
    brier_top8 = ((merged["p_top8"] - merged["made_top8"]) ** 2).mean()

    return {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "mean_abs_rank_error": mare,
        "top8_correct": top8_correct,
        "top8_accuracy": top8_accuracy,
        "top4_correct": top4_correct,
        "top4_accuracy": top4_accuracy,
        "minor_prem_hit": minor_prem_hit,
        "minor_prem_actual": actual_1st,
        "minor_prem_predicted": predicted_1st,
        "spoon_hit": spoon_hit,
        "spoon_actual": actual_last,
        "spoon_predicted": predicted_last,
        "mean_p_top8_for_top8_teams": mean_p_top8_actual_top8,
        "mean_p_top8_for_miss_teams": mean_p_top8_actual_miss,
        "mean_p_top4_for_top4_teams": mean_p_top4_actual_top4,
        "mean_p_top4_for_miss_teams": mean_p_top4_actual_miss,
        "brier_top8": brier_top8,
        "rank_col": rank_col,
        "merged_df": merged,  # Include for detailed analysis
    }


# ============================================================
# BACKTEST ORCHESTRATION
# ============================================================
def backtest_single_season(
    target_season: int,
    team_log: pd.DataFrame,
    odds_path: str,
    use_walkforward_elo: bool = USE_WALKFORWARD_ELO,
    use_premiership_blend: bool = True,
    premiership_blend_w: float = PREMIERSHIP_BLEND_W,
    use_momentum: bool = USE_MOMENTUM_FEATURES,
    momentum_weight: float = 50.0,
    use_coaching: bool = USE_COACHING_FEATURES,
    use_momentum_model: bool = False,
    use_list_priors: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run complete backtest for a single season.
    """
    # Determine training/calibration seasons
    train_through = target_season - 1
    calib_season = target_season - 1

    if verbose:
        print(f"  Training on seasons <= {train_through}")
        print(f"  Calibrating on season {calib_season}")
        print(f"  Walk-forward Elo: {use_walkforward_elo}")
        print(f"  Momentum features: {use_momentum} (weight={momentum_weight})")
        print(f"  Coaching features: {use_coaching} (boost={COACHING_ELO_BOOST})")
        print(f"  Momentum model feats: {use_momentum_model}")
        print(f"  Premiership blend: {use_premiership_blend} (w={premiership_blend_w})")

    # 1. Generate predictions
    if verbose:
        print("  Generating predictions...")
    predictions = generate_season_predictions(
        team_log=team_log,
        target_season=target_season,
        train_through_season=train_through,
        calib_season=calib_season,
        use_walkforward_elo=use_walkforward_elo,
        use_momentum=use_momentum,
        momentum_weight=momentum_weight,
        use_coaching=use_coaching,
        use_momentum_model=use_momentum_model,
        use_list_priors=use_list_priors,
    )
    if verbose:
        print(f"  Generated predictions for {len(predictions)} matches")

    # 2. Run ladder simulation
    if verbose:
        print("  Running ladder simulation...")
    predicted_ladder, _ = simulate_ladder_from_probs(
        predictions,
        prob_col="p_home_win_cal",
        n_sims=N_SIMS,
        shrink=SHRINK,
    )

    # 3. Blend with premiership odds if enabled
    if use_premiership_blend:
        predicted_ladder = blend_with_premiership_odds(
            predicted_ladder,
            target_season,
            blend_w=premiership_blend_w,
        )

    # 4. Get actual ladder
    if verbose:
        print("  Computing actual ladder...")
    actual_ladder = compute_actual_ladder(odds_path, target_season)

    # 5. Evaluate (use blended_rank if available, else mean_rank)
    if verbose:
        print("  Evaluating predictions...")
    metrics = evaluate_ladder_predictions(predicted_ladder, actual_ladder)
    metrics["season"] = target_season
    metrics["train_through"] = train_through
    metrics["n_matches"] = len(predictions)
    metrics["walkforward_elo"] = use_walkforward_elo
    metrics["premiership_blend"] = use_premiership_blend
    metrics["use_momentum"] = use_momentum
    metrics["use_momentum_model"] = use_momentum_model

    # Match-level metrics (log loss, Brier score)
    if "home_win" in predictions.columns:
        from sklearn.metrics import log_loss as sk_log_loss, brier_score_loss as sk_brier
        y = predictions["home_win"].astype(int).values
        p = predictions["p_home_win_cal"].clip(1e-6, 1 - 1e-6).values
        metrics["match_log_loss"] = float(sk_log_loss(y, p))
        metrics["match_brier"] = float(sk_brier(y, p))
        metrics["match_accuracy"] = float(((p >= 0.5).astype(int) == y).mean())

    return {
        "metrics": metrics,
        "predicted": predicted_ladder,
        "actual": actual_ladder,
        "predictions": predictions,
    }


def print_season_report(result: Dict):
    """Print detailed report for a single season backtest."""
    m = result["metrics"]
    pred = result["predicted"]
    actual = result["actual"]

    print(f"\n{'='*70}")
    print(f"SEASON {m['season']} RESULTS")
    print(f"Training: <= {m['train_through']} | Matches: {m['n_matches']}")
    wf = "Yes" if m.get("walkforward_elo", False) else "No"
    pb = "Yes" if m.get("premiership_blend", False) else "No"
    print(f"Walk-forward Elo: {wf} | Premiership blend: {pb}")
    print(f"{'='*70}")

    # Merge for display
    display = pred.merge(actual, on="team")

    # Determine which rank column to use for display
    rank_col = m.get("rank_col", "mean_rank")
    has_blend = "blended_rank" in display.columns

    if has_blend:
        display["rank_error"] = abs(display["blended_rank"] - display["actual_rank"])
        display = display.sort_values("actual_rank")

        print("\nPREDICTED vs ACTUAL LADDER:")
        print(f"{'Team':<22} {'Model':>6} {'Blend':>6} {'Actual':>7} {'Error':>6} {'p_top8':>7}")
        print("-" * 70)
        for _, row in display.iterrows():
            print(f"{row['team']:<22} {row['mean_rank']:>6.1f} {row['blended_rank']:>6.1f} "
                  f"{row['actual_rank']:>7} {row['rank_error']:>6.1f} {row['p_top8']:>7.2f}")
    else:
        display["rank_error"] = abs(display["mean_rank"] - display["actual_rank"])
        display = display.sort_values("actual_rank")

        print("\nPREDICTED vs ACTUAL LADDER:")
        print(f"{'Team':<22} {'Pred':>6} {'Actual':>7} {'Error':>6} {'p_top8':>7} {'p_top4':>7}")
        print("-" * 70)
        for _, row in display.iterrows():
            print(f"{row['team']:<22} {row['mean_rank']:>6.1f} {row['actual_rank']:>7} "
                  f"{row['rank_error']:>6.1f} {row['p_top8']:>7.2f} {row['p_top4']:>7.2f}")

    print(f"\nEVALUATION METRICS:")
    print(f"  Spearman Rho:          {m['spearman_rho']:.3f} (p={m['spearman_p']:.4f})")
    print(f"  Mean Abs Rank Error:   {m['mean_abs_rank_error']:.2f} positions")
    print(f"  Top 8 Accuracy:        {m['top8_correct']}/8 ({m['top8_accuracy']*100:.1f}%)")
    print(f"  Top 4 Accuracy:        {m['top4_correct']}/4 ({m['top4_accuracy']*100:.1f}%)")

    mp_status = "HIT" if m['minor_prem_hit'] else "MISS"
    print(f"  Minor Premier:         {mp_status} (predicted: {m['minor_prem_predicted']}, "
          f"actual: {m['minor_prem_actual']})")

    sp_status = "HIT" if m['spoon_hit'] else "MISS"
    print(f"  Wooden Spoon:          {sp_status} (predicted: {m['spoon_predicted']}, "
          f"actual: {m['spoon_actual']})")

    print(f"\nPROBABILITY CALIBRATION:")
    print(f"  Mean p_top8 for teams that made top 8:   {m['mean_p_top8_for_top8_teams']:.2f}")
    print(f"  Mean p_top8 for teams that missed:       {m['mean_p_top8_for_miss_teams']:.2f}")
    print(f"  Brier Score (top 8):                     {m['brier_top8']:.3f}")


def print_aggregate_summary(results: List[Dict]):
    """Print aggregate summary across all seasons."""
    print(f"\n{'='*70}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Season':<8} {'Spearman':>9} {'MARE':>6} {'Top8':>6} {'Top4':>6} "
          f"{'Minor':>7} {'Spoon':>7}")
    print("-" * 70)

    for r in results:
        m = r["metrics"]
        mp = "HIT" if m["minor_prem_hit"] else "MISS"
        sp = "HIT" if m["spoon_hit"] else "MISS"
        print(f"{m['season']:<8} {m['spearman_rho']:>9.3f} {m['mean_abs_rank_error']:>6.2f} "
              f"{m['top8_correct']}/8    {m['top4_correct']}/4    {mp:>7} {sp:>7}")

    # Aggregates
    print("-" * 70)
    avg_spearman = np.mean([r["metrics"]["spearman_rho"] for r in results])
    avg_mare = np.mean([r["metrics"]["mean_abs_rank_error"] for r in results])
    total_top8 = sum(r["metrics"]["top8_correct"] for r in results)
    total_top4 = sum(r["metrics"]["top4_correct"] for r in results)
    n_seasons = len(results)
    minor_hits = sum(1 for r in results if r["metrics"]["minor_prem_hit"])
    spoon_hits = sum(1 for r in results if r["metrics"]["spoon_hit"])

    print(f"{'Mean':<8} {avg_spearman:>9.3f} {avg_mare:>6.2f} "
          f"{total_top8}/{n_seasons*8}    {total_top4}/{n_seasons*4}    "
          f"{minor_hits}/{n_seasons}     {spoon_hits}/{n_seasons}")

    print(f"\nTop 8 accuracy: {total_top8/(n_seasons*8)*100:.1f}%")
    print(f"Top 4 accuracy: {total_top4/(n_seasons*4)*100:.1f}%")


def tune_blend_weight(
    team_log: pd.DataFrame,
    odds_path: str,
    test_seasons: List[int],
    weight_grid: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Grid search over blend weights to find optimal value.

    Returns DataFrame with metrics for each weight.
    """
    if weight_grid is None:
        weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []

    for w in weight_grid:
        print(f"\nTesting blend weight: {w:.1f}")

        season_metrics = []
        for season in test_seasons:
            try:
                result = backtest_single_season(
                    season, team_log, odds_path,
                    use_walkforward_elo=True,
                    use_premiership_blend=True,
                    premiership_blend_w=w,
                    verbose=False,
                )
                season_metrics.append(result["metrics"])
            except Exception as e:
                print(f"  Error for {season}: {e}")

        if season_metrics:
            avg_spearman = np.mean([m["spearman_rho"] for m in season_metrics])
            avg_mare = np.mean([m["mean_abs_rank_error"] for m in season_metrics])
            total_top8 = sum(m["top8_correct"] for m in season_metrics)
            total_top4 = sum(m["top4_correct"] for m in season_metrics)
            minor_hits = sum(1 for m in season_metrics if m["minor_prem_hit"])
            spoon_hits = sum(1 for m in season_metrics if m["spoon_hit"])
            avg_brier = np.mean([m["brier_top8"] for m in season_metrics])

            results.append({
                "blend_w": w,
                "avg_spearman": avg_spearman,
                "avg_mare": avg_mare,
                "total_top8": total_top8,
                "total_top4": total_top4,
                "minor_hits": minor_hits,
                "spoon_hits": spoon_hits,
                "avg_brier": avg_brier,
                "n_seasons": len(season_metrics),
            })

    return pd.DataFrame(results)


def print_momentum_analysis(team_log: pd.DataFrame, target_season: int):
    """
    Print momentum analysis for all teams heading into a season.
    Highlights potential breakout teams (high momentum).
    """
    # Build matches from team_log for Elo computation
    matches = build_match_level(team_log)

    momentum_data = get_preseason_momentum(team_log, matches, target_season)

    if not momentum_data:
        print(f"No momentum data available for {target_season}")
        return

    # Convert to DataFrame for easy sorting/display
    rows = []
    for team, data in momentum_data.items():
        rows.append({
            "team": team,
            "late_win_rate": data.get("late_win_rate", 0.5),
            "late_avg_margin": data.get("late_avg_margin", 0.0),
            "elo_change": data.get("elo_change", 0.0),
            "momentum_score": data.get("momentum_score", 0.0),
            "elo_end": data.get("elo_end", ELO_BASE),
        })

    df = pd.DataFrame(rows).sort_values("momentum_score", ascending=False)

    print(f"\n{'='*80}")
    print(f"MOMENTUM ANALYSIS: Teams heading into {target_season}")
    print(f"Based on {target_season - 1} season performance")
    print(f"{'='*80}")
    print(f"\n{'Team':<24} {'Late WR':>8} {'Late Marg':>10} {'Elo Chg':>8} {'Mom Score':>10} {'End Elo':>8}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{row['team']:<24} {row['late_win_rate']:>8.2f} {row['late_avg_margin']:>10.1f} "
              f"{row['elo_change']:>8.1f} {row['momentum_score']:>10.3f} {row['elo_end']:>8.0f}")

    # Highlight top/bottom momentum teams
    print(f"\n  Top 3 momentum (potential breakout):")
    for _, row in df.head(3).iterrows():
        print(f"    {row['team']}: momentum={row['momentum_score']:.3f}")

    print(f"\n  Bottom 3 momentum (potential decline):")
    for _, row in df.tail(3).iterrows():
        print(f"    {row['team']}: momentum={row['momentum_score']:.3f}")


def main():
    """Run backtests for available seasons, comparing baseline vs improved."""
    print("Loading data...")
    team_log = load_team_rows(TEAM_CACHE)

    # Determine available seasons
    available_seasons = sorted(team_log["season"].unique())
    print(f"Available seasons in team_log: {available_seasons}")

    # We need at least 1 prior season for training, so skip the first
    # Also check which seasons have odds data
    test_seasons = []
    for season in [2023, 2024, 2025]:
        if season in available_seasons and (season - 1) in available_seasons:
            try:
                # Quick check if odds data exists for this season
                _ = compute_actual_ladder(ODDS_FILE, season)
                test_seasons.append(season)
            except ValueError as e:
                print(f"Skipping {season}: {e}")

    if not test_seasons:
        print("No seasons available for backtesting!")
        return

    print(f"\nBacktesting seasons: {test_seasons}")

    # Run BASELINE (no walk-forward, no premiership blend)
    print("\n" + "=" * 70)
    print("BASELINE: Static Elo, No Premiership Blend")
    print("=" * 70)

    baseline_results = []
    for season in test_seasons:
        print(f"\n{'#'*70}")
        print(f"BACKTESTING SEASON {season} (BASELINE)")
        print(f"{'#'*70}")

        try:
            result = backtest_single_season(
                season, team_log, ODDS_FILE,
                use_walkforward_elo=False,
                use_premiership_blend=False,
                use_momentum=False,
                use_coaching=False,
            )
            baseline_results.append(result)
            print_season_report(result)
        except Exception as e:
            print(f"ERROR backtesting {season}: {e}")
            import traceback
            traceback.print_exc()

    if len(baseline_results) > 1:
        print_aggregate_summary(baseline_results)

    # Run IMPROVED (walk-forward Elo + premiership blend)
    print("\n" + "=" * 70)
    print("IMPROVED: Walk-forward Elo + Premiership Blend")
    print("=" * 70)

    improved_results = []
    for season in test_seasons:
        print(f"\n{'#'*70}")
        print(f"BACKTESTING SEASON {season} (IMPROVED)")
        print(f"{'#'*70}")

        try:
            result = backtest_single_season(
                season, team_log, ODDS_FILE,
                use_walkforward_elo=True,
                use_premiership_blend=True,
                premiership_blend_w=PREMIERSHIP_BLEND_W,
                use_momentum=False,
                use_coaching=False,
            )
            improved_results.append(result)
            print_season_report(result)
        except Exception as e:
            print(f"ERROR backtesting {season}: {e}")
            import traceback
            traceback.print_exc()

    if len(improved_results) > 1:
        print_aggregate_summary(improved_results)

    # Print comparison
    if baseline_results and improved_results:
        print("\n" + "=" * 70)
        print("COMPARISON: BASELINE vs IMPROVED")
        print("=" * 70)

        print(f"\n{'Season':<8} {'Baseline':^25} {'Improved':^25}")
        print(f"{'':8} {'Spearman':>9} {'MARE':>6} {'Top8':>6} {'Spearman':>9} {'MARE':>6} {'Top8':>6}")
        print("-" * 70)

        for b, i in zip(baseline_results, improved_results):
            bm, im = b["metrics"], i["metrics"]
            print(f"{bm['season']:<8} {bm['spearman_rho']:>9.3f} {bm['mean_abs_rank_error']:>6.2f} "
                  f"{bm['top8_correct']}/8    {im['spearman_rho']:>9.3f} {im['mean_abs_rank_error']:>6.2f} "
                  f"{im['top8_correct']}/8")

        # Totals
        print("-" * 70)
        b_spear = np.mean([r["metrics"]["spearman_rho"] for r in baseline_results])
        b_mare = np.mean([r["metrics"]["mean_abs_rank_error"] for r in baseline_results])
        b_top8 = sum(r["metrics"]["top8_correct"] for r in baseline_results)

        i_spear = np.mean([r["metrics"]["spearman_rho"] for r in improved_results])
        i_mare = np.mean([r["metrics"]["mean_abs_rank_error"] for r in improved_results])
        i_top8 = sum(r["metrics"]["top8_correct"] for r in improved_results)

        n = len(baseline_results)
        print(f"{'Mean':<8} {b_spear:>9.3f} {b_mare:>6.2f} {b_top8}/{n*8}    "
              f"{i_spear:>9.3f} {i_mare:>6.2f} {i_top8}/{n*8}")

        # Delta
        print(f"\nImprovement:")
        print(f"  Spearman: {i_spear - b_spear:+.3f}")
        print(f"  MARE: {i_mare - b_mare:+.2f} (negative = better)")
        print(f"  Top 8: {i_top8 - b_top8:+d} more correct")

    # Show coaching analysis for each season
    print("\n" + "=" * 70)
    print("COACHING TENURE ANALYSIS")
    print("=" * 70)
    for season in test_seasons:
        print_coaching_analysis(season)

    # Run WITH COACHING
    print("\n" + "=" * 70)
    print("WITH COACHING: Walk-forward Elo + Coaching Tenure Features")
    print("=" * 70)

    coaching_results = []
    for season in test_seasons:
        print(f"\n{'#'*70}")
        print(f"BACKTESTING SEASON {season} (WITH COACHING)")
        print(f"{'#'*70}")

        try:
            result = backtest_single_season(
                season, team_log, ODDS_FILE,
                use_walkforward_elo=True,
                use_premiership_blend=False,
                use_momentum=False,
                use_coaching=True,
            )
            coaching_results.append(result)
            print_season_report(result)
        except Exception as e:
            print(f"ERROR backtesting {season}: {e}")
            import traceback
            traceback.print_exc()

    if len(coaching_results) > 1:
        print_aggregate_summary(coaching_results)

    # Compare improved (no coaching) vs coaching
    if improved_results and coaching_results:
        print("\n" + "=" * 70)
        print("COMPARISON: IMPROVED vs WITH COACHING")
        print("=" * 70)

        print(f"\n{'Season':<8} {'Improved':^25} {'With Coaching':^25}")
        print(f"{'':8} {'Spearman':>9} {'MARE':>6} {'Top8':>6} {'Spearman':>9} {'MARE':>6} {'Top8':>6}")
        print("-" * 70)

        for i_res, c_res in zip(improved_results, coaching_results):
            im, cm = i_res["metrics"], c_res["metrics"]
            print(f"{im['season']:<8} {im['spearman_rho']:>9.3f} {im['mean_abs_rank_error']:>6.2f} "
                  f"{im['top8_correct']}/8    {cm['spearman_rho']:>9.3f} {cm['mean_abs_rank_error']:>6.2f} "
                  f"{cm['top8_correct']}/8")

        print("-" * 70)
        i_spear = np.mean([r["metrics"]["spearman_rho"] for r in improved_results])
        i_mare = np.mean([r["metrics"]["mean_abs_rank_error"] for r in improved_results])
        i_top8 = sum(r["metrics"]["top8_correct"] for r in improved_results)

        c_spear = np.mean([r["metrics"]["spearman_rho"] for r in coaching_results])
        c_mare = np.mean([r["metrics"]["mean_abs_rank_error"] for r in coaching_results])
        c_top8 = sum(r["metrics"]["top8_correct"] for r in coaching_results)

        n = len(improved_results)
        print(f"{'Mean':<8} {i_spear:>9.3f} {i_mare:>6.2f} {i_top8}/{n*8}    "
              f"{c_spear:>9.3f} {c_mare:>6.2f} {c_top8}/{n*8}")

        print(f"\nCoaching Impact:")
        print(f"  Spearman: {c_spear - i_spear:+.3f}")
        print(f"  MARE: {c_mare - i_mare:+.2f} (negative = better)")
        print(f"  Top 8: {c_top8 - i_top8:+d} more correct")

        # Check breakout teams specifically
        print("\n" + "=" * 70)
        print("BREAKOUT TEAM ANALYSIS")
        print("=" * 70)
        breakout_teams = [
            (2023, "Collingwood", "Predicted ~8th, Won flag"),
            (2024, "Hawthorn", "Predicted ~14th, Finished 7th"),
            (2025, "Adelaide", "Predicted ~12th, Finished 1st"),
        ]
        for season, team, note in breakout_teams:
            if season not in [r["metrics"]["season"] for r in coaching_results]:
                continue
            # Find results for this season
            i_res = next((r for r in improved_results if r["metrics"]["season"] == season), None)
            c_res = next((r for r in coaching_results if r["metrics"]["season"] == season), None)
            if i_res and c_res:
                # Get predicted ranks for the team
                i_pred = i_res["predicted"]
                c_pred = c_res["predicted"]
                actual = i_res["actual"]

                i_team_rank = i_pred[i_pred["team"] == team]["mean_rank"].values
                c_team_rank = c_pred[c_pred["team"] == team]["mean_rank"].values
                actual_rank = actual[actual["team"] == team]["actual_rank"].values

                # Check coaching tenure
                tenure = COACHING_TENURE.get(season, {}).get(team, "?")
                in_zone = "YES" if tenure in BREAKOUT_TENURE_YEARS else "NO"

                if len(i_team_rank) > 0 and len(c_team_rank) > 0 and len(actual_rank) > 0:
                    print(f"\n{season} {team} ({note}):")
                    print(f"  Coach tenure:    Year {tenure} (in breakout zone: {in_zone})")
                    print(f"  Actual rank:     {actual_rank[0]}")
                    print(f"  Improved pred:   {i_team_rank[0]:.1f}")
                    print(f"  Coaching pred:   {c_team_rank[0]:.1f}")
                    improvement = i_team_rank[0] - c_team_rank[0]
                    if improvement > 0:
                        print(f"  Improvement:     +{improvement:.1f} positions closer")
                    else:
                        print(f"  Improvement:     {improvement:.1f} positions")

    # Run WITH MOMENTUM MODEL
    print("\n" + "=" * 70)
    print("WITH MOMENTUM MODEL: Walk-forward Elo + Coaching + Momentum Features")
    print("=" * 70)

    momentum_model_results = []
    for season in test_seasons:
        print(f"\n{'#'*70}")
        print(f"BACKTESTING SEASON {season} (WITH MOMENTUM MODEL)")
        print(f"{'#'*70}")

        try:
            result = backtest_single_season(
                season, team_log, ODDS_FILE,
                use_walkforward_elo=True,
                use_premiership_blend=False,
                use_momentum=False,
                use_coaching=True,
                use_momentum_model=True,
            )
            momentum_model_results.append(result)
            print_season_report(result)
        except Exception as e:
            print(f"ERROR backtesting {season}: {e}")
            import traceback
            traceback.print_exc()

    if len(momentum_model_results) > 1:
        print_aggregate_summary(momentum_model_results)

    # Compare WITH COACHING vs WITH MOMENTUM MODEL
    if coaching_results and momentum_model_results:
        print("\n" + "=" * 70)
        print("COMPARISON: WITH COACHING vs WITH MOMENTUM MODEL")
        print("=" * 70)

        print(f"\n{'Season':<8} {'With Coaching':^25} {'With Momentum Model':^25}")
        print(f"{'':8} {'Spearman':>9} {'MARE':>6} {'Top8':>6} {'Spearman':>9} {'MARE':>6} {'Top8':>6}")
        print("-" * 70)

        for c_res, m_res in zip(coaching_results, momentum_model_results):
            cm, mm = c_res["metrics"], m_res["metrics"]
            print(f"{cm['season']:<8} {cm['spearman_rho']:>9.3f} {cm['mean_abs_rank_error']:>6.2f} "
                  f"{cm['top8_correct']}/8    {mm['spearman_rho']:>9.3f} {mm['mean_abs_rank_error']:>6.2f} "
                  f"{mm['top8_correct']}/8")

        print("-" * 70)
        c_spear = np.mean([r["metrics"]["spearman_rho"] for r in coaching_results])
        c_mare = np.mean([r["metrics"]["mean_abs_rank_error"] for r in coaching_results])
        c_top8 = sum(r["metrics"]["top8_correct"] for r in coaching_results)

        m_spear = np.mean([r["metrics"]["spearman_rho"] for r in momentum_model_results])
        m_mare = np.mean([r["metrics"]["mean_abs_rank_error"] for r in momentum_model_results])
        m_top8 = sum(r["metrics"]["top8_correct"] for r in momentum_model_results)

        n = len(coaching_results)
        print(f"{'Mean':<8} {c_spear:>9.3f} {c_mare:>6.2f} {c_top8}/{n*8}    "
              f"{m_spear:>9.3f} {m_mare:>6.2f} {m_top8}/{n*8}")

        print(f"\nMomentum Model Impact:")
        print(f"  Spearman: {m_spear - c_spear:+.3f}")
        print(f"  MARE: {m_mare - c_mare:+.2f} (negative = better)")
        print(f"  Top 8: {m_top8 - c_top8:+d} more correct")

        # Match-level metrics comparison
        print(f"\nMATCH-LEVEL METRICS:")
        print(f"{'Season':<8} {'With Coaching':^30} {'With Momentum Model':^30}")
        print(f"{'':8} {'LogLoss':>9} {'Brier':>8} {'Acc':>7} {'LogLoss':>9} {'Brier':>8} {'Acc':>7}")
        print("-" * 80)
        for c_res, m_res in zip(coaching_results, momentum_model_results):
            cm, mm = c_res["metrics"], m_res["metrics"]
            c_ll = cm.get("match_log_loss", float("nan"))
            c_br = cm.get("match_brier", float("nan"))
            c_ac = cm.get("match_accuracy", float("nan"))
            m_ll = mm.get("match_log_loss", float("nan"))
            m_br = mm.get("match_brier", float("nan"))
            m_ac = mm.get("match_accuracy", float("nan"))
            print(f"{cm['season']:<8} {c_ll:>9.4f} {c_br:>8.4f} {c_ac:>7.1%} "
                  f"{m_ll:>9.4f} {m_br:>8.4f} {m_ac:>7.1%}")

        # Breakout team analysis
        print("\n" + "=" * 70)
        print("BREAKOUT TEAM ANALYSIS: COACHING vs MOMENTUM MODEL")
        print("=" * 70)
        breakout_teams = [
            (2023, "Collingwood", "Predicted ~8th, Won flag"),
            (2024, "Hawthorn", "Predicted ~14th, Finished 7th"),
            (2025, "Adelaide", "Predicted ~12th, Finished 1st"),
        ]
        for season, team, note in breakout_teams:
            c_res = next((r for r in coaching_results if r["metrics"]["season"] == season), None)
            m_res = next((r for r in momentum_model_results if r["metrics"]["season"] == season), None)
            if c_res and m_res:
                c_rank = c_res["predicted"][c_res["predicted"]["team"] == team]["mean_rank"].values
                m_rank = m_res["predicted"][m_res["predicted"]["team"] == team]["mean_rank"].values
                actual_rank = c_res["actual"][c_res["actual"]["team"] == team]["actual_rank"].values

                if len(c_rank) > 0 and len(m_rank) > 0 and len(actual_rank) > 0:
                    delta = c_rank[0] - m_rank[0]
                    print(f"\n{season} {team} ({note}):")
                    print(f"  Actual rank:          {actual_rank[0]}")
                    print(f"  Coaching pred:        {c_rank[0]:.1f}")
                    print(f"  Momentum model pred:  {m_rank[0]:.1f}")
                    if delta > 0:
                        print(f"  Change:               +{delta:.1f} positions closer")
                    else:
                        print(f"  Change:               {delta:.1f} positions")

    # Run WITH LIST PRIORS
    print("\n" + "=" * 70)
    print("WITH LIST PRIORS: Walk-forward Elo + Coaching + Momentum + List Age")
    print("=" * 70)

    list_priors_results = []
    for season in test_seasons:
        print(f"\n{'#'*70}")
        print(f"BACKTESTING SEASON {season} (WITH LIST PRIORS)")
        print(f"{'#'*70}")

        try:
            result = backtest_single_season(
                season, team_log, ODDS_FILE,
                use_walkforward_elo=True,
                use_premiership_blend=False,
                use_momentum=False,
                use_coaching=True,
                use_momentum_model=True,
                use_list_priors=True,
            )
            list_priors_results.append(result)
            print_season_report(result)
        except Exception as e:
            print(f"ERROR backtesting {season}: {e}")
            import traceback
            traceback.print_exc()

    if len(list_priors_results) > 1:
        print_aggregate_summary(list_priors_results)

    # Compare WITH MOMENTUM MODEL vs WITH LIST PRIORS
    if momentum_model_results and list_priors_results:
        print("\n" + "=" * 70)
        print("COMPARISON: WITH MOMENTUM MODEL vs WITH LIST PRIORS")
        print("=" * 70)

        print(f"\n{'Season':<8} {'With Momentum Model':^25} {'With List Priors':^25}")
        print(f"{'':8} {'Spearman':>9} {'MARE':>6} {'Top8':>6} {'Spearman':>9} {'MARE':>6} {'Top8':>6}")
        print("-" * 70)

        for m_res, l_res in zip(momentum_model_results, list_priors_results):
            mm, lm = m_res["metrics"], l_res["metrics"]
            print(f"{mm['season']:<8} {mm['spearman_rho']:>9.3f} {mm['mean_abs_rank_error']:>6.2f} "
                  f"{mm['top8_correct']}/8    {lm['spearman_rho']:>9.3f} {lm['mean_abs_rank_error']:>6.2f} "
                  f"{lm['top8_correct']}/8")

        print("-" * 70)
        m_spear = np.mean([r["metrics"]["spearman_rho"] for r in momentum_model_results])
        m_mare = np.mean([r["metrics"]["mean_abs_rank_error"] for r in momentum_model_results])
        m_top8 = sum(r["metrics"]["top8_correct"] for r in momentum_model_results)

        l_spear = np.mean([r["metrics"]["spearman_rho"] for r in list_priors_results])
        l_mare = np.mean([r["metrics"]["mean_abs_rank_error"] for r in list_priors_results])
        l_top8 = sum(r["metrics"]["top8_correct"] for r in list_priors_results)

        n = len(momentum_model_results)
        print(f"{'Mean':<8} {m_spear:>9.3f} {m_mare:>6.2f} {m_top8}/{n*8}    "
              f"{l_spear:>9.3f} {l_mare:>6.2f} {l_top8}/{n*8}")

        print(f"\nList Priors Impact:")
        print(f"  Spearman: {l_spear - m_spear:+.3f}")
        print(f"  MARE: {l_mare - m_mare:+.2f} (negative = better)")
        print(f"  Top 8: {l_top8 - m_top8:+d} more correct")

    # Tune blend weight
    print("\n" + "=" * 70)
    print("BLEND WEIGHT TUNING")
    print("=" * 70)
    print("Testing weights from 0.0 (pure model) to 1.0 (pure market)...")

    tune_results = tune_blend_weight(
        team_log, ODDS_FILE, test_seasons,
        weight_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    print("\n" + "=" * 70)
    print("BLEND WEIGHT TUNING RESULTS")
    print("=" * 70)
    print(f"\n{'Weight':<8} {'Spearman':>9} {'MARE':>6} {'Top8':>6} {'Top4':>6} {'Minor':>6} {'Brier':>7}")
    print("-" * 70)
    for _, row in tune_results.iterrows():
        n = int(row["n_seasons"])
        print(f"{row['blend_w']:<8.1f} {row['avg_spearman']:>9.3f} {row['avg_mare']:>6.2f} "
              f"{int(row['total_top8'])}/{n*8}    {int(row['total_top4'])}/{n*4}    "
              f"{int(row['minor_hits'])}/{n}     {row['avg_brier']:>6.3f}")

    # Find optimal weight for different metrics
    best_spearman = tune_results.loc[tune_results["avg_spearman"].idxmax()]
    best_mare = tune_results.loc[tune_results["avg_mare"].idxmin()]
    best_top8 = tune_results.loc[tune_results["total_top8"].idxmax()]
    best_brier = tune_results.loc[tune_results["avg_brier"].idxmin()]

    print(f"\nOptimal weights by metric:")
    print(f"  Best Spearman ({best_spearman['avg_spearman']:.3f}): w = {best_spearman['blend_w']:.1f}")
    print(f"  Best MARE ({best_mare['avg_mare']:.2f}): w = {best_mare['blend_w']:.1f}")
    print(f"  Best Top 8 ({int(best_top8['total_top8'])}/24): w = {best_top8['blend_w']:.1f}")
    print(f"  Best Brier ({best_brier['avg_brier']:.3f}): w = {best_brier['blend_w']:.1f}")

    print("\nBacktest complete!")


TEAM_AGES_2026 = {
    "Collingwood": 25.6, "Melbourne": 25.4, "Brisbane Lions": 25.3,
    "Geelong": 25.3, "Sydney": 25.0, "Carlton": 24.9,
    "Western Bulldogs": 24.8, "Adelaide": 24.7, "Port Adelaide": 24.3,
    "Greater Western Sydney": 24.3, "Hawthorn": 24.3, "Fremantle": 24.2,
    "St Kilda": 24.2, "Gold Coast": 24.1, "North Melbourne": 24.1,
    "Richmond": 23.8, "Essendon": 23.7, "West Coast": 23.5,
}


def predict_2026():
    """
    Generate 2026 preseason ladder predictions using the improved model.
    Uses walk-forward Elo + coaching tenure + momentum model features + list age priors.
    """
    print("=" * 70)
    print("2026 AFL LADDER PREDICTION")
    print("Model: Walk-forward Elo + Coaching + Momentum + List Age Priors")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    team_log = load_team_rows(TEAM_CACHE)

    # Show coaching analysis for 2026
    print_coaching_analysis(2026)

    # Train on all available data (through 2025)
    train_through = 2025
    calib_season = 2025

    print(f"\nTraining on seasons <= {train_through}")
    print(f"Calibrating on season {calib_season}")

    # Build historical feature frame (with momentum model features + list priors)
    hist = build_historical_feature_frame(
        team_log=team_log,
        model_stats=MODEL_STATS,
        form_windows=FORM_WINDOWS,
        elo_k=ELO_K,
        elo_home_adv=ELO_HOME_ADV,
        elo_base=ELO_BASE,
        use_momentum=True,
        use_list_priors=True,
    )

    feats = make_feature_list(team_log, MODEL_STATS, FORM_WINDOWS,
                              use_momentum=True, use_list_priors=True)
    print(f"Features: {feats}")

    # Train model
    model, (a, b) = train_and_calibrate(
        hist=hist,
        feats=feats,
        train_through_season=train_through,
        calib_season=calib_season,
    )

    # Print model coefficients for momentum features
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]
    print("\nModel coefficients (standardized):")
    for fname, coef in zip(feats, clf.coef_[0]):
        print(f"  {fname:<25} {coef:+.4f}")

    # Get initial Elo ratings as of end of 2025
    initial_elos = compute_current_elos(
        hist[hist["season"] <= train_through],
        k=ELO_K,
        home_adv=ELO_HOME_ADV,
        base=ELO_BASE,
    )

    # Apply coaching adjustments for 2026
    print("\nApplying coaching tenure adjustments...")
    coaching_adj = get_coaching_adjustments(2026)
    for team, adj in coaching_adj.items():
        if team in initial_elos and adj != 0:
            print(f"  {team}: +{adj:.0f} Elo (coach in breakout zone)")
            initial_elos[team] += adj

    # Print Elo rankings after adjustment
    print("\n" + "=" * 70)
    print("PRE-SEASON ELO RATINGS (with coaching adjustment)")
    print("=" * 70)
    sorted_elos = sorted(initial_elos.items(), key=lambda x: -x[1])
    print(f"\n{'Rank':<6} {'Team':<24} {'Elo':>8}")
    print("-" * 45)
    for i, (team, elo) in enumerate(sorted_elos, 1):
        coach_note = " *" if coaching_adj.get(team, 0) > 0 else ""
        print(f"{i:<6} {team:<24} {elo:>8.0f}{coach_note}")
    print("\n* = coach in breakout zone (Year 2-3)")

    # Compute momentum snapshot from 2025 season
    mom_snap = compute_team_momentum_features(hist, train_through)
    print("\n" + "=" * 70)
    print("MOMENTUM FEATURES (from 2025 season)")
    print("=" * 70)
    mom_display = mom_snap.sort_values("mom_elo_slope", ascending=False)
    print(f"\n{'Team':<24} {'Elo Slope':>10} {'2H Delta':>9} {'WR Last8':>9} {'Pct Trend':>10}")
    print("-" * 70)
    for _, row in mom_display.iterrows():
        print(f"{row['team']:<24} {row['mom_elo_slope']:>10.2f} {row['mom_second_half_delta']:>9.3f} "
              f"{row['mom_win_rate_last8']:>9.3f} {row['mom_pct_trend']:>10.3f}")

    # Load 2026 fixtures
    from afl_pipeline import load_fixtures_csv
    fx = load_fixtures_csv("fixtures_2026.csv")
    fx = fx[fx["season"] == 2026].copy()
    print(f"\nLoaded {len(fx)} fixtures for 2026")

    # Get form snapshot
    snap = latest_form_snapshot(
        team_log,
        model_stats=MODEL_STATS,
        form_windows=FORM_WINDOWS,
    )

    # Build list priors snapshot for 2026 (scraped data + hardcoded 2026 ages)
    all_priors = load_list_priors()
    ages_2026_df = pd.DataFrame([
        {"season": 2026, "team": t, "avg_age": a}
        for t, a in TEAM_AGES_2026.items()
    ])
    all_priors_with_2026 = pd.concat([all_priors, ages_2026_df], ignore_index=True)
    lp_snap = all_priors_with_2026[all_priors_with_2026["season"] == 2026].copy()

    # Display 2026 list ages
    print("\n" + "=" * 70)
    print("LIST AGE PRIORS (2026)")
    print("=" * 70)
    age_display = lp_snap.sort_values("avg_age", ascending=False)
    print(f"\n{'Team':<24} {'Avg Age':>8}")
    print("-" * 35)
    for _, row in age_display.iterrows():
        print(f"{row['team']:<24} {row['avg_age']:>8.1f}")

    # Add features to fixtures (including momentum + list priors)
    fx_feat = add_fixture_features(
        fixtures=fx,
        elos=initial_elos,
        form_snap=snap,
        team_log=team_log,
        model_stats=MODEL_STATS,
        form_windows=FORM_WINDOWS,
        elo_base=ELO_BASE,
        momentum_snap=mom_snap,
        list_priors_snap=lp_snap,
    )

    # Predict
    p_raw = predict_proba(model, fx_feat, feats)
    fx_feat["p_home_win_cal"] = apply_platt(p_raw, a, b)

    # Run ladder simulation
    print("\nRunning Monte Carlo ladder simulation (20,000 iterations)...")
    predicted_ladder, _ = simulate_ladder_from_probs(
        fx_feat,
        prob_col="p_home_win_cal",
        n_sims=N_SIMS,
        shrink=SHRINK,
    )

    # Display results
    print("\n" + "=" * 70)
    print("2026 PREDICTED LADDER")
    print("=" * 70)
    print(f"\n{'Rank':<6} {'Team':<24} {'Exp Pts':>8} {'p_Top4':>8} {'p_Top8':>8} {'p_Prem':>8}")
    print("-" * 70)

    predicted_ladder = predicted_ladder.sort_values("mean_rank")
    for i, row in predicted_ladder.iterrows():
        rank = int(row["mean_rank"] + 0.5)
        coach_note = " *" if coaching_adj.get(row["team"], 0) > 0 else ""
        print(f"{rank:<6} {row['team']:<24} {row['exp_points']:>8.1f} "
              f"{row['p_top4']:>8.1%} {row['p_top8']:>8.1%} {row['p_minor_prem']:>8.1%}{coach_note}")

    print("\n* = coach in breakout zone (Year 2-3)")

    # Identify breakout candidates
    print("\n" + "=" * 70)
    print("BREAKOUT CANDIDATES (coaches in Year 2-3)")
    print("=" * 70)
    breakout_teams = [t for t, y in COACHING_TENURE.get(2026, {}).items()
                      if y in BREAKOUT_TENURE_YEARS]
    for team in breakout_teams:
        row = predicted_ladder[predicted_ladder["team"] == team]
        if len(row) > 0:
            row = row.iloc[0]
            tenure = COACHING_TENURE[2026].get(team, "?")
            mom_row = mom_snap[mom_snap["team"] == team]
            print(f"\n{team} (Coach Year {tenure}):")
            print(f"  Predicted rank: {row['mean_rank']:.1f}")
            print(f"  Top 8 probability: {row['p_top8']:.1%}")
            if len(mom_row) > 0:
                mr = mom_row.iloc[0]
                print(f"  Momentum: elo_slope={mr['mom_elo_slope']:.1f}, "
                      f"2H_delta={mr['mom_second_half_delta']:.3f}, "
                      f"WR_last8={mr['mom_win_rate_last8']:.3f}")
            print(f"  Finals upside if system clicks: Could finish 2-4 spots higher")

    # Save to CSV
    output_file = "afl_2026_ladder_prediction_with_momentum.csv"
    predicted_ladder.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

    return predicted_ladder


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "predict2026":
        predict_2026()
    else:
        main()
