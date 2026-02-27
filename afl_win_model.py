import os
import re
import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from typing import Optional


# =========================
# CONFIG
# =========================

FIXTURES_FILE = "fixtures_2026.csv"   # your fixtures CSV
OUT_FIXTURE_PREDS = "afl_2026_fixture_preds.csv"
PREDICT_SEASON = 2026
CALIB_SEASON = 2025                  # last completed season to fit Platt
TRAIN_THROUGH_SEASON = 2025          # train on all games up to this season

TEAM_CACHE = "afl_team_rows.csv"          # cached team rows (from AFL Tables scrape)
ODDS_FILE = "aflodds.csv"                # your odds csv
OUT_WITH_ODDS = "afl_preds_with_odds.csv"

TRAIN_END_SEASON = 2024
TEST_SEASON = 2025

# Elo parameters
ELO_K = 25
ELO_HOME_ADV = 65
ELO_BASE = 1500
USE_MARGIN_ELO = True        # Use margin-adjusted Elo instead of binary win/loss
ELO_MARGIN_SCALE = 100       # Margin divisor (margin/scale gives outcome adjustment)
ELO_SEASON_REGRESS = 0.33    # Regress toward mean between seasons (0 = no regression, 1 = full reset)

# Rolling form windows
FORM_WINDOWS = [5]  # keep simple

# Team stats to use for form (must exist in afl_team_rows.csv)
MODEL_STATS = ["I50", "CL", "DI"]  # change to your preferred set

# Feature toggles
USE_VENUE_REST_TRAVEL = False  # start False to get baseline stable

# Optional: blend weight (market-heavy is usually best)
BLEND_W = 0.85

# Venue-based home/away features
USE_VENUE_HOME_AWAY = False  # Toggle for venue-based features (disabled - no edge found)
VENUE_MIN_GAMES = 5          # Min games at venue for venue-specific rate
HOME_AWAY_WINDOW = 20        # Rolling window for home/away records
GLOBAL_HOME_WIN_RATE = 0.57  # Cold-start fallback for venue home win rate

# External data features
USE_EXTERNAL_DATA = True     # Toggle for external data features
USE_SPREAD_FEATURES = True   # Use bookmaker spread/line data
USE_TOTAL_FEATURES = True    # Use over/under total score data
USE_KICKOFF_FEATURES = True  # Use day/night, day of week features
USE_WEATHER_FEATURES = True  # Fetch and use weather data (requires internet)
WEATHER_CACHE_FILE = "weather_cache.csv"  # Cache for weather API calls

# Head-to-head and ladder features
USE_H2H_FEATURES = True      # Historical head-to-head records
H2H_MIN_GAMES = 3            # Min H2H games for reliable rate
USE_LADDER_FEATURES = True   # Ladder position and finals race
FINALS_CUTOFF = 8            # Top N teams make finals

# Venue state mapping (extend as needed)
VENUE_STATE_MAP = {
    # VIC
    "MCG": "VIC",
    "Marvel Stadium": "VIC",
    "GMHBA Stadium": "VIC",
    "Kardinia Park": "VIC",
    "Mars Stadium": "VIC",
    "Marvl": "VIC",  # typo in your odds
    # NSW/ACT
    "SCG": "NSW",
    "Sydney Showground": "NSW",
    "ENGIE Stadium": "NSW",
    "Accor Stadium": "NSW",
    "Manuka Oval": "ACT",
    # QLD
    "Gabba": "QLD",
    "People First Stadium": "QLD",
    "Cazaly’s Stadium": "QLD",
    # SA
    "Adelaide Oval": "SA",
    "Norwood Oval": "SA",
    "Adelaide Hills": "SA",
    "Barossa Park": "SA",
    "Hands Oval": "SA",
    # WA
    "Optus Stadium": "WA",
    # TAS
    "Blundstone Arena": "TAS",
    "UTAS Stadium": "TAS",
    "Ninja Stadium": "TAS",
    # NT
    "TIO Stadium": "NT",
    "Traeger Park": "NT",
}

TEAM_STATE_MAP = {
    "Adelaide": "SA",
    "Brisbane Lions": "QLD",
    "Carlton": "VIC",
    "Collingwood": "VIC",
    "Essendon": "VIC",
    "Fremantle": "WA",
    "Geelong": "VIC",
    "Gold Coast": "QLD",
    "Greater Western Sydney": "NSW",
    "Hawthorn": "VIC",
    "Melbourne": "VIC",
    "North Melbourne": "VIC",
    "Port Adelaide": "SA",
    "Richmond": "VIC",
    "St Kilda": "VIC",
    "Sydney": "NSW",
    "West Coast": "WA",
    "Western Bulldogs": "VIC",
}


# =========================
# TEAM NORMALISATION
# =========================
def norm_team(x: str) -> str:
    """Normalize team names between AFLTables and odds feed."""
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
        # add any others you see in your fixtures file
    }
    if s_lower in lower_map:
        return lower_map[s_lower]

    # common punctuation/unicode
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u2013", "-").replace("\u2014", "-")

    # canonical mapping for known variants
    mapping = {
        "GWS": "Greater Western Sydney",
        "GWS Giants": "Greater Western Sydney",
        "Greater Western Sydney Giants": "Greater Western Sydney",
        "Western Bulldogs": "Western Bulldogs",
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

    # exact mapping if found
    if s in mapping:
        return mapping[s]

    # light cleanups
    s = re.sub(r"\s+", " ", s)
    s = s.replace("St.", "St").strip()

    # if still a known key after cleanup
    return mapping.get(s, s)
def resolve_model_stats(team_log: pd.DataFrame, desired: list[str]) -> list[str]:
    """
    Map desired stat names (MODEL_STATS) to actual columns in team_log.
    Tries exact match, case-insensitive match, then aliases.
    """
    cols = list(team_log.columns)
    cols_lower = {c.lower(): c for c in cols}

    aliases = {
        # Inside 50s
        "I50": ["IF", "I50", "I50s", "Inside50", "Inside 50", "Inside 50s", "I-50"],
        # Clearances
        "CL": ["CL", "Clr", "Clearances"],
        # Disposals
        "DI": ["DI", "Disp", "Disposals"],
    }

    resolved = []
    for d in desired:
        # exact
        if d in cols:
            resolved.append(d)
            continue
        # case-insensitive
        if d.lower() in cols_lower:
            resolved.append(cols_lower[d.lower()])
            continue
        # alias candidates
        cands = aliases.get(d, [d])
        found = None
        for c in cands:
            if c in cols:
                found = c
                break
            if c.lower() in cols_lower:
                found = cols_lower[c.lower()]
                break
        if found is None:
            raise ValueError(
                f"Could not resolve stat '{d}'. "
                f"Update MODEL_STATS or aliases. Example columns: {cols[:40]} (total {len(cols)})"
            )
        resolved.append(found)

    # de-dup preserve order
    out = []
    for c in resolved:
        if c not in out:
            out.append(c)
    return out


# =========================
# UTIL / METRICS
# =========================
# =========================
# UTIL / METRICS
# =========================
def safe_read_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV/TSV reader:
    - auto-detects delimiter (comma/tab/semicolon)
    - handles utf-8 / cp1252
    - strips junk 'Unnamed' columns
    """
    for enc in (None, "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(path, encoding="cp1252", sep=None, engine="python")

    # Drop junk unnamed columns (index artifacts etc)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    return df

def read_odds_file(path: str) -> pd.DataFrame:
    """
    OddsPortal file has a junk first row, real headers are on row 2.
    """
    try:
        df = pd.read_csv(path, header=1, encoding="cp1252")
    except Exception:
        df = safe_read_csv(path)

    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    return df




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
from math import comb

def mcnemar_exact(b: int, c: int) -> float:
    """
    Exact two-sided McNemar p-value using binomial test.
    b = A only correct
    c = B only correct
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_tip_compare(
    df: pd.DataFrame,
    p_a: str,
    p_b: str,
    label_a: str,
    label_b: str,
):
    """
    Paired comparison of two tipping strategies on the same games.
    Prints the 2x2 table and net advantage on disagreement games.
    """
    y = df["home_win"].astype(int).values

    a = (df[p_a].astype(float).values >= 0.5).astype(int)
    b = (df[p_b].astype(float).values >= 0.5).astype(int)

    a_ok = (a == y)
    b_ok = (b == y)

    both_ok = int((a_ok & b_ok).sum())
    both_bad = int((~a_ok & ~b_ok).sum())
    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())

    print(f"\n=== Paired tipping compare: {label_a} vs {label_b} ===")
    print(f"Both correct: {both_ok}")
    print(f"Both wrong:   {both_bad}")
    print(f"{label_a} only correct: {a_only}")
    print(f"{label_b} only correct: {b_only}")
    print(f"Net advantage ({label_a} - {label_b}): {a_only - b_only}")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def fit_platt_calibrator(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Fit Platt scaling: calibrated = sigmoid(a * logit(p) + b)
    Returns (a, b).
    """
    z = logit(p).reshape(-1, 1)
    clf = LogisticRegression(max_iter=500, C=1e6)  # essentially unregularised for 1D
    clf.fit(z, y.astype(int))
    a = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    return a, b

def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    z = logit(p)
    return inv_logit(a * z + b)


def add_hybrid_tip(
    df: pd.DataFrame,
    model_col: str,
    market_col: str,
    delta_thresh: float = 0.15,
    market_band: tuple[float, float] = (0.35, 0.65),
    max_delta: Optional[float] = None,
    out_col: str = "p_hybrid",
) -> pd.DataFrame:
    """
    Hybrid probability used for tipping:
      - default to market probability
      - override to model probability only if:
          (a) model side != market side
          (b) |model - market| >= delta_thresh
          (c) market probability is within market_band (avoid fading strong favourites)
          (d) optional: |model - market| <= max_delta (cap extreme disagreements)

    IMPORTANT: this implementation uses NumPy arrays to avoid pandas dtype/alignment edge cases.
    """
    d = df.copy()

    pm_s = pd.to_numeric(d[model_col], errors="coerce")
    pk_s = pd.to_numeric(d[market_col], errors="coerce")

    pm = pm_s.to_numpy(dtype=float)
    pk = pk_s.to_numpy(dtype=float)

    dt = float(delta_thresh)
    lo, hi = float(market_band[0]), float(market_band[1])
    md = None if max_delta is None else float(max_delta)

    delta = pm - pk
    abs_delta = np.abs(delta)

    model_side = pm >= 0.5
    market_side = pk >= 0.5
    disagree = model_side != market_side

    in_band = (pk >= lo) & (pk <= hi)

    finite = np.isfinite(pm) & np.isfinite(pk) & np.isfinite(delta)

    override = finite & disagree & (abs_delta >= dt) & in_band
    if md is not None:
        override = override & (abs_delta <= md)

    # Hybrid probability
    d[out_col] = np.where(override, pm, pk)

    # Helpful flags for debugging
    flag_base = out_col.replace("p_", "")  # e.g. p_hybrid_tuned -> hybrid_tuned
    d[f"{flag_base}_override"] = override.astype(int)
    d[f"{flag_base}_delta"] = delta

    return d



# =========================
# 1) INSERT THIS FUNCTION (tuner) RIGHT AFTER add_hybrid_tip()
# =========================
def tune_hybrid_rule(
    df: pd.DataFrame,
    model_col: str,
    market_col: str,
    delta_grid=None,
    band_grid=None,
    max_delta_grid=None,
) -> dict:
    """
    Tune (delta_thresh, market_band, max_delta) to maximise tipping correctness on df.

    Objective:
      1) maximise correct (integer count)
      2) tie-break: fewer overrides (simpler rule)
      3) tie-break: higher acc (should be redundant if n fixed, but safe)
    """

    if delta_grid is None:
        delta_grid = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    if band_grid is None:
        band_grid = [(0.47, 0.53), (0.45, 0.55), (0.40, 0.60), (0.35, 0.65), (0.30, 0.70)]
    if max_delta_grid is None:
        max_delta_grid = [None, 0.25, 0.30, 0.40, 0.50]

    # Defensive: required columns
    for c in [model_col, market_col, "home_win"]:
        if c not in df.columns:
            raise KeyError(f"tune_hybrid_rule: missing required column '{c}'")

    best = None

    for dt in delta_grid:
        for band in band_grid:
            for md in max_delta_grid:
                tmp = add_hybrid_tip(
                    df,
                    model_col=model_col,
                    market_col=market_col,
                    delta_thresh=dt,
                    market_band=band,
                    max_delta=md,
                    out_col="p_hybrid_tmp",
                )

                t = evaluate_tipping(tmp, "p_hybrid_tmp", "tmp")
                correct = int(t["correct"])
                acc = float(t["accuracy"])

                override_col = "hybrid_tmp_override"
                overrides = int(tmp[override_col].sum())

                # Override hit rate (optional)
                if overrides > 0:
                    tips_bin = (tmp["p_hybrid_tmp"] >= 0.5).astype(int)
                    hit = float(
                        (
                            tips_bin[tmp[override_col] == 1]
                            == tmp.loc[tmp[override_col] == 1, "home_win"].astype(int)
                        ).mean()
                    )
                else:
                    hit = np.nan

                cand = {
                    "correct": correct,
                    "acc": acc,
                    "delta_thresh": float(dt),
                    "band": (float(band[0]), float(band[1])),
                    "max_delta": (None if md is None else float(md)),
                    "overrides": overrides,
                    "override_hit": hit,
                }

                if best is None:
                    best = cand
                else:
                    if cand["correct"] > best["correct"]:
                        best = cand
                    elif cand["correct"] == best["correct"]:
                        # tie-break 1: fewer overrides
                        if cand["overrides"] < best["overrides"]:
                            best = cand
                        elif cand["overrides"] == best["overrides"]:
                            # tie-break 2: higher accuracy (belt + braces)
                            if cand["acc"] > best["acc"]:
                                best = cand

    return best

def tune_blend_weight(
    df: pd.DataFrame,
    model_col: str,
    market_col: str,
    w_grid=None,
) -> dict:
    """
    Tune blend weight w to maximise tipping correctness.
    blended = w * market + (1 - w) * model
    """

    if w_grid is None:
        w_grid = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, ..., 1.0

    for c in [model_col, market_col, "home_win"]:
        if c not in df.columns:
            raise KeyError(f"tune_blend_weight: missing required column '{c}'")

    best = None

    for w in w_grid:
        p_blend = w * df[market_col].astype(float) + (1 - w) * df[model_col].astype(float)
        tips = (p_blend >= 0.5).astype(int)
        correct = int((tips == df["home_win"].astype(int)).sum())
        acc = correct / len(df)

        cand = {"w": float(w), "correct": correct, "acc": acc}

        if best is None or cand["correct"] > best["correct"]:
            best = cand

    return best



def rank_hybrid_grid(df, model_col, market_col, delta_grid, band_grid, max_delta_grid, top_n=10):
    rows = []
    for dt in delta_grid:
        for band in band_grid:
            for md in max_delta_grid:
                tmp = add_hybrid_tip(
                    df,
                    model_col=model_col,
                    market_col=market_col,
                    delta_thresh=dt,
                    market_band=band,
                    max_delta=md,
                    out_col="p_hybrid_tmp",
                )
                t = evaluate_tipping(tmp, "p_hybrid_tmp", "tmp")
                rows.append({
                    "correct": int(t["correct"]),
                    "acc": float(t["accuracy"]),
                    "dt": float(dt),
                    "band": band,
                    "md": md,
                    "overrides": int(tmp["hybrid_tmp_override"].sum()),
                })

    out = (
        pd.DataFrame(rows)
        .sort_values(["correct", "overrides"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    return out



# =========================
# LOAD + BUILD MATCH LEVEL
# =========================
def load_team_rows() -> pd.DataFrame:
    if not os.path.exists(TEAM_CACHE):
        raise FileNotFoundError(f"Missing {TEAM_CACHE}. Put afl_team_rows.csv in this folder.")
    df = safe_read_csv(TEAM_CACHE)
    # Ensure expected columns
    for col in ["season", "match_url", "Team", "is_home", "score_for", "score_against"]:
        if col not in df.columns:
            raise ValueError(f"{TEAM_CACHE} missing required column: {col}")
    return df


def match_url_to_date(url: str) -> pd.Timestamp:
    # AFLTables urls contain YYYYMMDD near end: .../YYYY/xxxxYYYYMMDD.html
    m = re.search(r"(\d{8})\.html$", str(url))
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")


def build_match_level(team_log: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level rows (2 per match_url) into match-level rows (1 per match_url).

    Requires team_log columns:
      Team, is_home, score_for, score_against, match_url, season
    """
    needed = {"Team", "is_home", "score_for", "score_against", "match_url", "season"}
    missing = needed - set(team_log.columns)
    if missing:
        raise ValueError(f"TEAM_CACHE missing required columns: {missing}")

    tl = team_log.copy()
    tl["Team"] = tl["Team"].map(norm_team)
    tl["date"] = tl["match_url"].map(match_url_to_date)
    tl["season"] = tl["season"].astype(int)

    home = tl[tl["is_home"] == 1].copy()
    away = tl[tl["is_home"] == 0].copy()

    # stat columns = everything except the core identifiers
    core = {"Team", "is_home", "score_for", "score_against", "match_url", "season", "date"}
    stat_cols = [c for c in tl.columns if c not in core]

    # rename home/away sides
    home = home.rename(columns={"Team": "home_team", "score_for": "home_score", "score_against": "away_score"})
    away = away.rename(columns={"Team": "away_team", "score_for": "away_score2", "score_against": "home_score2"})

    # merge to get one row per match
    m = home[["match_url", "date", "season", "home_team", "home_score", "away_score"] + stat_cols].merge(
        away[["match_url", "away_team"] + stat_cols],
        on="match_url",
        how="inner",
        suffixes=("_home", "_away"),
    )

    # outcome
    m["home_win"] = (m["home_score"] > m["away_score"]).astype(int)
    return m


# =========================
# ODDS MERGE
# =========================
def odds_to_implied_prob(odds: pd.Series) -> pd.Series:
    o = pd.to_numeric(odds, errors="coerce")
    return 1.0 / o


def novig_two_way(home_odds: pd.Series, away_odds: pd.Series) -> pd.Series:
    """2-way normalisation removes bookmaker margin for head-to-head odds."""
    h = pd.to_numeric(home_odds, errors="coerce")
    a = pd.to_numeric(away_odds, errors="coerce")
    ph = 1.0 / h
    pa = 1.0 / a
    denom = ph + pa
    return ph / denom


def merge_odds_meta(matches: pd.DataFrame, odds_file: str) -> pd.DataFrame:
    if not os.path.exists(odds_file):
        raise FileNotFoundError(f"Missing {odds_file} in this folder.")

    # Your file has headers on row 2 (index 1)
    # Robust odds load (handles tab/comma + weird encodings)
    odds = safe_read_csv(odds_file)

    # If headers are on row 2 in some exports, detect and re-read
    if "Home Team" not in odds.columns and "Home Odds" not in odds.columns:
        odds = pd.read_csv(odds_file, header=1, encoding="cp1252")

    # Normalize types
    odds["date"] = pd.to_datetime(odds["Date"], format="%d-%b-%y", errors="coerce")

    odds["home_team"] = odds["Home Team"].astype(str).str.strip().map(norm_team)
    odds["away_team"] = odds["Away Team"].astype(str).str.strip().map(norm_team)

    # raw implied probs from open/close (home only)
    odds["p_home_open"] = odds_to_implied_prob(odds.get("Home Odds Open", np.nan))
    odds["p_home_close"] = odds_to_implied_prob(odds.get("Home Odds Close", np.nan))

    # If open/close missing, fall back to "Home Odds"
    if "Home Odds" in odds.columns and odds["p_home_open"].isna().all():
        odds["p_home_open"] = odds_to_implied_prob(odds["Home Odds"])
    if "Home Odds" in odds.columns and odds["p_home_close"].isna().all():
        odds["p_home_close"] = odds_to_implied_prob(odds["Home Odds"])

    # No-vig (2-way normalised) market probabilities
    odds["p_home_open_nv"] = novig_two_way(
        odds.get("Home Odds Open", np.nan),
        odds.get("Away Odds Open", np.nan),
    )
    odds["p_home_close_nv"] = novig_two_way(
        odds.get("Home Odds Close", np.nan),
        odds.get("Away Odds Close", np.nan),
    )

    # Fallback: if open/close missing, use midpoint columns if present
    if odds["p_home_open_nv"].isna().all() and "Home Odds" in odds.columns and "Away Odds" in odds.columns:
        odds["p_home_open_nv"] = novig_two_way(odds["Home Odds"], odds["Away Odds"])
    if odds["p_home_close_nv"].isna().all() and "Home Odds" in odds.columns and "Away Odds" in odds.columns:
        odds["p_home_close_nv"] = novig_two_way(odds["Home Odds"], odds["Away Odds"])

    keep = [
        "date", "home_team", "away_team",
        "Venue", "Play Off Game?",
        "p_home_open", "p_home_close",
        "p_home_open_nv", "p_home_close_nv",
        "Home Odds Open", "Home Odds Close",
        "Away Odds Open", "Away Odds Close",
        "Home Odds", "Away Odds",
        # Spread/line data
        "Home Line Open", "Home Line Close",
        "Away Line Open", "Away Line Close",
        # Total score data
        "Total Score Open", "Total Score Close",
        # Kickoff time
        "Kick Off (local)",
    ]
    keep = [c for c in keep if c in odds.columns]
    odds_small = odds[keep].copy()

    merged = matches.merge(odds_small, on=["date", "home_team", "away_team"], how="left")

    # Some rows might be reversed in the odds feed (rare). Try a second merge swap as fallback.
    missing = merged["p_home_open"].isna() & merged["p_home_open_nv"].isna()
    if missing.any():
        swapped = matches.merge(
            odds_small.rename(columns={"home_team": "away_team", "away_team": "home_team"}),
            on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_sw"),
        )

        # Fill meta
        for col in ["Venue", "Play Off Game?"]:
            if col in swapped.columns and f"{col}_sw" in swapped.columns:
                merged.loc[missing, col] = swapped.loc[missing, f"{col}_sw"]

        # Fill no-vig probs (safe): swapped "home" is actually our away => p_home = 1 - p_swapped_home
        if "p_home_open_nv_sw" in swapped.columns:
            merged.loc[missing, "p_home_open_nv"] = 1.0 - swapped.loc[missing, "p_home_open_nv_sw"]
        if "p_home_close_nv_sw" in swapped.columns:
            merged.loc[missing, "p_home_close_nv"] = 1.0 - swapped.loc[missing, "p_home_close_nv_sw"]

        # Optional: fill raw implied as well (still margin-included)
        if "p_home_open_sw" in swapped.columns:
            merged.loc[missing, "p_home_open"] = 1.0 - swapped.loc[missing, "p_home_open_sw"]
        if "p_home_close_sw" in swapped.columns:
            merged.loc[missing, "p_home_close"] = 1.0 - swapped.loc[missing, "p_home_close_sw"]

    return merged


# =========================
# ELO + FORM FEATURES
# =========================
def elo_expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def margin_to_outcome(margin: float, scale: float = ELO_MARGIN_SCALE) -> float:
    """
    Convert game margin to outcome value for Elo update.

    Instead of binary 1 (win) or 0 (loss), uses margin-scaled value:
    - Big win (+60 points) -> ~1.0
    - Close win (+5 points) -> ~0.55
    - Close loss (-5 points) -> ~0.45
    - Big loss (-60 points) -> ~0.0

    Uses sigmoid-like scaling clipped to [0.05, 0.95] to prevent extreme updates.
    """
    # Linear scaling: 0.5 + margin/scale, clipped
    outcome = 0.5 + margin / scale
    return np.clip(outcome, 0.05, 0.95)


def compute_elo_diff(matches: pd.DataFrame, k=ELO_K, home_adv=ELO_HOME_ADV, base=ELO_BASE,
                     use_margin=USE_MARGIN_ELO, margin_scale=ELO_MARGIN_SCALE,
                     season_regress=ELO_SEASON_REGRESS) -> pd.DataFrame:
    """
    Compute Elo ratings with optional margin adjustment and season regression.

    Parameters:
    - use_margin: If True, use margin-adjusted outcomes instead of binary win/loss
    - margin_scale: Divisor for margin (higher = less weight to margin)
    - season_regress: Fraction to regress toward base between seasons (0 = none, 1 = full reset)
    """
    d = matches.sort_values("date").copy()
    teams = pd.unique(pd.concat([d["home_team"], d["away_team"]]))
    elos = {t: float(base) for t in teams}

    elo_home_pre = []
    elo_away_pre = []

    current_season = None

    for idx, r in d.iterrows():
        # Season regression: regress ratings toward mean at start of new season
        game_season = r.get("season", None)
        if game_season is not None and current_season is not None and game_season != current_season:
            if season_regress > 0:
                for t in elos:
                    elos[t] = elos[t] * (1 - season_regress) + base * season_regress
        current_season = game_season

        h, a = r["home_team"], r["away_team"]
        eh = elos.get(h, float(base))
        ea = elos.get(a, float(base))

        # store pre-game
        elo_home_pre.append(eh)
        elo_away_pre.append(ea)

        # Calculate outcome
        if use_margin and "home_score" in r and "away_score" in r:
            margin = float(r["home_score"]) - float(r["away_score"])
            y = margin_to_outcome(margin, margin_scale)
        else:
            y = float(r["home_win"])

        # update
        exp_home = elo_expected(eh + home_adv, ea)
        elos[h] = eh + k * (y - exp_home)
        elos[a] = ea + k * (exp_home - y)

    d["elo_home_pre"] = elo_home_pre
    d["elo_away_pre"] = elo_away_pre
    d["elo_diff"] = d["elo_home_pre"] - d["elo_away_pre"]

    if use_margin:
        print(f"Using margin-adjusted Elo (scale={margin_scale}, season_regress={season_regress})")

    return d

def load_fixtures_csv(path: str) -> pd.DataFrame:
    fx = pd.read_csv(path)

    # Standardise column names you showed: round, date, venue, home_team, away_team, Season
    fx = fx.rename(columns={
        "Season": "season",
        "home_team": "home_team",
        "away_team": "away_team",
        "venue": "Venue",
        "round": "round",
        "date": "date",
    })

    fx["season"] = fx["season"].astype(int)
    fx["date"] = pd.to_datetime(fx["date"], dayfirst=True, errors="coerce")

    # Normalise teams (IMPORTANT: your fixtures have variants like "GWS GIANTS", "Gold Coast SUNS", "Sydney Swans")
    fx["home_team"] = fx["home_team"].astype(str).str.strip().map(norm_team)
    fx["away_team"] = fx["away_team"].astype(str).str.strip().map(norm_team)

    # Convert odds if present
    for c in ["home_odds", "away_odds"]:
        if c in fx.columns:
            fx[c] = pd.to_numeric(fx[c], errors="coerce")

    # Minimal columns needed (plus odds if present)
    keep = ["round", "date", "Venue", "home_team", "away_team", "season"]
    for c in ["home_odds", "away_odds"]:
        if c in fx.columns:
            keep.append(c)

    fx = fx[keep].copy()

    fx = fx.dropna(subset=["date", "home_team", "away_team", "season"])
    return fx

def compute_current_elos(matches: pd.DataFrame, k=ELO_K, home_adv=ELO_HOME_ADV, base=ELO_BASE,
                         use_margin=USE_MARGIN_ELO, margin_scale=ELO_MARGIN_SCALE,
                         season_regress=ELO_SEASON_REGRESS) -> dict:
    """
    Compute latest Elo ratings per team after processing all completed matches (sorted by date).
    Uses the SAME update rule as compute_elo_diff().
    """
    d = matches.sort_values("date").copy()
    teams = pd.unique(pd.concat([d["home_team"], d["away_team"]]))
    elos = {t: float(base) for t in teams}

    current_season = None

    for _, r in d.iterrows():
        # Season regression
        game_season = r.get("season", None)
        if game_season is not None and current_season is not None and game_season != current_season:
            if season_regress > 0:
                for t in elos:
                    elos[t] = elos[t] * (1 - season_regress) + base * season_regress
        current_season = game_season

        h, a = r["home_team"], r["away_team"]
        eh = elos.get(h, float(base))
        ea = elos.get(a, float(base))

        # Calculate outcome
        if use_margin and "home_score" in r and "away_score" in r:
            margin = float(r["home_score"]) - float(r["away_score"])
            y = margin_to_outcome(margin, margin_scale)
        else:
            y = float(r["home_win"])

        exp_home = elo_expected(eh + home_adv, ea)
        elos[h] = eh + k * (y - exp_home)
        elos[a] = ea + k * (exp_home - y)

    return elos


def latest_form_snapshot(team_log: pd.DataFrame) -> pd.DataFrame:
    """
    Build the latest rolling-form values per team (as of last game played).
    Output columns: Team + form_{stat}_{w} for resolved MODEL_STATS.
    """
    resolved_stats = resolve_model_stats(team_log, MODEL_STATS)

    cols_needed = ["match_url", "Team"] + resolved_stats
    missing = [c for c in cols_needed if c not in team_log.columns]
    if missing:
        raise ValueError(f"TEAM_CACHE missing columns needed for form snapshot: {missing}")

    tg = team_log[cols_needed].copy()
    tg["Team"] = tg["Team"].map(norm_team)
    tg["date"] = tg["match_url"].apply(match_url_to_date)
    tg = tg.dropna(subset=["date"]).sort_values(["Team", "date"])

    for s in resolved_stats:
        tg[s] = pd.to_numeric(tg[s], errors="coerce")

    # shifted rolling mean (pregame form)
    for w in FORM_WINDOWS:
        for s in resolved_stats:
            col = f"form_{s}_{w}"
            tg[col] = tg.groupby("Team")[s].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    form_cols = [f"form_{s}_{w}" for w in FORM_WINDOWS for s in resolved_stats]

    # last available form per team
    snap = (
        tg.sort_values(["Team", "date"])
          .groupby("Team", as_index=False)[form_cols]
          .last()
          .rename(columns={"Team": "team"})
    )
    return snap


def add_fixture_features(fixtures: pd.DataFrame, elos: dict, form_snap: pd.DataFrame) -> pd.DataFrame:
    """
    Attach current Elo + latest form snapshot to future fixtures.
    Produces: elo_diff and diff_form_{resolved_stat} (when single window)
    """
    fx = fixtures.copy()

    # Elo diff
    fx["elo_home_pre"] = fx["home_team"].map(lambda t: elos.get(t, float(ELO_BASE)))
    fx["elo_away_pre"] = fx["away_team"].map(lambda t: elos.get(t, float(ELO_BASE)))
    fx["elo_diff"] = fx["elo_home_pre"] - fx["elo_away_pre"]

    # Form snapshot join (home/away)
    home = form_snap.rename(columns={"team": "home_team"})
    away = form_snap.rename(columns={"team": "away_team"})

    fx = fx.merge(home, on="home_team", how="left").merge(away, on="away_team", how="left", suffixes=("_home", "_away"))

    # Build diffs for each resolved stat/window
    resolved_stats = resolve_model_stats(load_team_rows(), MODEL_STATS)  # or pass team_log if you prefer
    for w in FORM_WINDOWS:
        for s in resolved_stats:
            col = f"form_{s}_{w}"
            fx[f"diff_form_{s}_{w}"] = fx[f"{col}_home"] - fx[f"{col}_away"]

    # Convenience aliases when only one window
    if len(FORM_WINDOWS) == 1:
        w = FORM_WINDOWS[0]
        for s in resolved_stats:
            fx[f"diff_form_{s}"] = fx[f"diff_form_{s}_{w}"]

    return fx


def add_form_features(matches: pd.DataFrame, team_log: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling averages of chosen stats (MODEL_STATS) at team-game level,
    then build match-level differentials home - away for each stat/window.
    Uses ONLY past games because of shift(1).
    """
    resolved_stats = resolve_model_stats(team_log, MODEL_STATS)

    cols_needed = ["match_url", "season", "Team", "is_home"] + resolved_stats
    missing = [c for c in cols_needed if c not in team_log.columns]
    if missing:
        raise ValueError(f"TEAM_CACHE missing columns needed after resolve: {missing}")

    tg = team_log[cols_needed].copy()
    tg["Team"] = tg["Team"].map(norm_team)
    tg["date"] = tg["match_url"].apply(match_url_to_date)
    tg = tg.dropna(subset=["date"]).sort_values(["Team", "date"])

    print("Using MODEL_STATS resolved as:", dict(zip(MODEL_STATS, resolved_stats)))

    for s in resolved_stats:
        tg[s] = pd.to_numeric(tg[s], errors="coerce")

    # rolling means, shifted to avoid leakage
    for w in FORM_WINDOWS:
        for s in resolved_stats:
            col = f"form_{s}_{w}"
            tg[col] = tg.groupby("Team")[s].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    # IMPORTANT: form_cols must be built from resolved_stats (not MODEL_STATS)
    form_cols = [f"form_{s}_{w}" for w in FORM_WINDOWS for s in resolved_stats]

    home = tg.loc[tg["is_home"].astype(int) == 1, ["match_url", "Team"] + form_cols].rename(
        columns={"Team": "home_team"})
    away = tg.loc[tg["is_home"].astype(int) == 0, ["match_url", "Team"] + form_cols].rename(
        columns={"Team": "away_team"})

    m = (
        matches
        .merge(home, on=["match_url", "home_team"], how="left")
        .merge(away, on=["match_url", "away_team"], how="left", suffixes=("_home", "_away"))
    )

    for w in FORM_WINDOWS:
        for s in resolved_stats:
            m[f"diff_form_{s}_{w}"] = m[f"form_{s}_{w}_home"] - m[f"form_{s}_{w}_away"]

    # Convenience aliases when only one window
    if len(FORM_WINDOWS) == 1:
        w = FORM_WINDOWS[0]
        for s in resolved_stats:
            m[f"diff_form_{s}"] = m[f"diff_form_{s}_{w}"]

    return m


# =========================
# VENUE / REST / TRAVEL (optional)
# =========================
def venue_to_state(v: str):
    if pd.isna(v):
        return np.nan
    v = str(v).strip()
    if v in VENUE_STATE_MAP:
        return VENUE_STATE_MAP[v]
    lv = v.lower()
    if "mcg" in lv or "marvel" in lv:
        return "VIC"
    if "gabba" in lv:
        return "QLD"
    if "optus" in lv:
        return "WA"
    if "adelaide oval" in lv:
        return "SA"
    if "scg" in lv:
        return "NSW"
    if "tio" in lv or "traeger" in lv:
        return "NT"
    if "norwood" in lv or "barossa" in lv or "adelaide hills" in lv or "hands oval" in lv:
        return "SA"
    if "mars" in lv:
        return "VIC"
    if "accor" in lv:
        return "NSW"
    if "cazaly" in lv:
        return "QLD"
    if "ninja stadium" in lv:
        return "TAS"
    return np.nan


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").copy()

    home = d[["date", "home_team"]].rename(columns={"home_team": "team"})
    away = d[["date", "away_team"]].rename(columns={"away_team": "team"})
    team_games = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])

    team_games["prev_date"] = team_games.groupby("team")["date"].shift(1)
    team_games["rest_days"] = (team_games["date"] - team_games["prev_date"]).dt.days
    team_games.loc[(team_games["rest_days"] < 3) | (team_games["rest_days"] > 30), "rest_days"] = np.nan
    team_games["rest_days"] = team_games["rest_days"].clip(lower=3, upper=21)

    d = d.merge(
        team_games[["date", "team", "rest_days"]].rename(columns={"team": "home_team", "rest_days": "rest_home"}),
        on=["date", "home_team"],
        how="left"
    ).merge(
        team_games[["date", "team", "rest_days"]].rename(columns={"team": "away_team", "rest_days": "rest_away"}),
        on=["date", "away_team"],
        how="left"
    )

    d["rest_home"] = d["rest_home"].fillna(7)
    d["rest_away"] = d["rest_away"].fillna(7)
    d["rest_diff"] = d["rest_home"] - d["rest_away"]
    return d


def add_travel_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Venue"] = d["Venue"].astype(str).str.strip()
    d["venue_state"] = d["Venue"].apply(venue_to_state)
    d["home_state"] = d["home_team"].map(TEAM_STATE_MAP)
    d["away_state"] = d["away_team"].map(TEAM_STATE_MAP)

    d["home_interstate"] = (
        (d["home_state"].notna()) & (d["venue_state"].notna()) & (d["home_state"] != d["venue_state"])
    ).astype(int)
    d["away_interstate"] = (
        (d["away_state"].notna()) & (d["venue_state"].notna()) & (d["away_state"] != d["venue_state"])
    ).astype(int)

    d["travel_diff"] = d["away_interstate"] - d["home_interstate"]

    d["neutral_venue"] = (
        d["venue_state"].notna() &
        d["home_state"].notna() &
        d["away_state"].notna() &
        (d["venue_state"] != d["home_state"]) &
        (d["venue_state"] != d["away_state"])
    ).astype(int)

    d["both_interstate"] = ((d["home_interstate"] == 1) & (d["away_interstate"] == 1)).astype(int)
    return d


# =========================
# VENUE HOME/AWAY FEATURES
# =========================
def compute_venue_home_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical home win rate at each venue.
    Uses expanding mean with shift(1) to avoid leakage.
    Cold-start: use global home win rate when venue has < VENUE_MIN_GAMES.
    """
    d = df.sort_values("date").copy()

    # Ensure Venue column exists and is clean
    if "Venue" not in d.columns:
        d["venue_home_win_rate"] = GLOBAL_HOME_WIN_RATE
        return d

    d["Venue"] = d["Venue"].astype(str).str.strip()

    # Compute expanding mean of home_win by venue, shifted by 1 to avoid leakage
    d["_venue_cumsum"] = d.groupby("Venue")["home_win"].cumsum().shift(1)
    d["_venue_count"] = d.groupby("Venue").cumcount()  # 0-indexed count before current game

    # Calculate rate (avoiding division by zero)
    d["_venue_rate"] = np.where(
        d["_venue_count"] >= VENUE_MIN_GAMES,
        d["_venue_cumsum"] / d["_venue_count"],
        np.nan
    )

    # Fill cold-start with global rate
    d["venue_home_win_rate"] = d["_venue_rate"].fillna(GLOBAL_HOME_WIN_RATE)

    # Clean up temp columns
    d = d.drop(columns=["_venue_cumsum", "_venue_count", "_venue_rate"])

    return d


def compute_team_venue_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-venue combo, compute cumulative win rate (shifted).
    Returns difference: team_venue_win_rate_diff = home_team's rate - away_team's rate at this venue.
    Cold-start: use team's overall win rate when < 3 games at venue.
    """
    d = df.sort_values("date").copy()

    if "Venue" not in d.columns:
        d["team_venue_win_rate_diff"] = 0.0
        return d

    d["Venue"] = d["Venue"].astype(str).str.strip()

    # Build team-game level data from matches
    # Home team perspective
    home_games = d[["date", "Venue", "home_team", "home_win"]].copy()
    home_games = home_games.rename(columns={"home_team": "team", "home_win": "win"})

    # Away team perspective
    away_games = d[["date", "Venue", "away_team", "home_win"]].copy()
    away_games["win"] = 1 - away_games["home_win"]  # away team wins when home team loses
    away_games = away_games.rename(columns={"away_team": "team"})
    away_games = away_games.drop(columns=["home_win"])

    # Combine all team-venue games
    all_games = pd.concat([home_games, away_games], ignore_index=True)
    all_games = all_games.sort_values(["team", "Venue", "date"]).reset_index(drop=True)

    # Compute cumulative win rate per team-venue (shifted)
    all_games["_tv_cumsum"] = all_games.groupby(["team", "Venue"])["win"].cumsum().shift(1)
    all_games["_tv_count"] = all_games.groupby(["team", "Venue"]).cumcount()

    # Team-venue rate (require at least 3 games)
    all_games["_tv_rate"] = np.where(
        all_games["_tv_count"] >= 3,
        all_games["_tv_cumsum"] / all_games["_tv_count"],
        np.nan
    )

    # Compute overall team win rate as fallback (shifted)
    all_games["_team_cumsum"] = all_games.groupby("team")["win"].cumsum().shift(1)
    all_games["_team_count"] = all_games.groupby("team").cumcount()
    all_games["_team_rate"] = np.where(
        all_games["_team_count"] > 0,
        all_games["_team_cumsum"] / all_games["_team_count"],
        0.5  # ultimate fallback
    )

    # Fill team-venue rate with team overall rate if cold-start
    all_games["team_venue_rate"] = all_games["_tv_rate"].fillna(all_games["_team_rate"])

    # Get latest team-venue rate for each team-venue combo up to each date
    # Create a lookup: for each match, get home team's rate at venue and away team's rate at venue

    # Build lookup tables by taking the last known rate for each team-venue before each game
    # We'll merge back to original df using date + venue + team

    # For home team: look up their rate at this venue
    home_lookup = home_games[["date", "Venue", "team"]].copy()
    home_lookup = home_lookup.merge(
        all_games[["date", "Venue", "team", "team_venue_rate"]],
        on=["date", "Venue", "team"],
        how="left"
    )
    home_lookup = home_lookup.rename(columns={"team": "home_team", "team_venue_rate": "home_venue_rate"})

    # For away team: look up their rate at this venue
    away_lookup = away_games[["date", "Venue", "team"]].copy()
    away_lookup = away_lookup.merge(
        all_games[["date", "Venue", "team", "team_venue_rate"]],
        on=["date", "Venue", "team"],
        how="left"
    )
    away_lookup = away_lookup.rename(columns={"team": "away_team", "team_venue_rate": "away_venue_rate"})

    # Merge back to main df
    d = d.merge(home_lookup[["date", "Venue", "home_team", "home_venue_rate"]],
                on=["date", "Venue", "home_team"], how="left")
    d = d.merge(away_lookup[["date", "Venue", "away_team", "away_venue_rate"]],
                on=["date", "Venue", "away_team"], how="left")

    # Fill any remaining NaN with 0.5
    d["home_venue_rate"] = d["home_venue_rate"].fillna(0.5)
    d["away_venue_rate"] = d["away_venue_rate"].fillna(0.5)

    # Compute difference
    d["team_venue_win_rate_diff"] = d["home_venue_rate"] - d["away_venue_rate"]

    # Clean up temp columns
    d = d.drop(columns=["home_venue_rate", "away_venue_rate"], errors="ignore")

    return d


def compute_home_away_records(df: pd.DataFrame, window: int = HOME_AWAY_WINDOW) -> pd.DataFrame:
    """
    Track each team's rolling home win rate (home games only) and away win rate (away games only).
    Shift by 1 game to avoid leakage.
    Returns difference: home_away_record_diff = home_team's home record - away_team's away record.
    """
    d = df.sort_values("date").copy()

    # Build team-level data
    # Home games
    home_games = d[["date", "home_team", "home_win"]].copy()
    home_games = home_games.rename(columns={"home_team": "team", "home_win": "win"})
    home_games["is_home_game"] = 1

    # Away games
    away_games = d[["date", "away_team", "home_win"]].copy()
    away_games["win"] = 1 - away_games["home_win"]
    away_games = away_games.rename(columns={"away_team": "team"})
    away_games = away_games.drop(columns=["home_win"])
    away_games["is_home_game"] = 0

    # Process home games: rolling home win rate
    home_games = home_games.sort_values(["team", "date"]).reset_index(drop=True)
    home_games["home_record"] = home_games.groupby("team")["win"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    # Process away games: rolling away win rate
    away_games = away_games.sort_values(["team", "date"]).reset_index(drop=True)
    away_games["away_record"] = away_games.groupby("team")["win"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    # Create lookup for home team's home record
    home_lookup = home_games[["date", "team", "home_record"]].copy()
    home_lookup = home_lookup.rename(columns={"team": "home_team", "home_record": "home_team_home_record"})

    # Create lookup for away team's away record
    away_lookup = away_games[["date", "team", "away_record"]].copy()
    away_lookup = away_lookup.rename(columns={"team": "away_team", "away_record": "away_team_away_record"})

    # Merge back
    d = d.merge(home_lookup, on=["date", "home_team"], how="left")
    d = d.merge(away_lookup, on=["date", "away_team"], how="left")

    # Fill NaN with 0.5 (cold-start / first games)
    d["home_team_home_record"] = d["home_team_home_record"].fillna(0.5)
    d["away_team_away_record"] = d["away_team_away_record"].fillna(0.5)

    # Compute difference
    d["home_away_record_diff"] = d["home_team_home_record"] - d["away_team_away_record"]

    # Clean up temp columns
    d = d.drop(columns=["home_team_home_record", "away_team_away_record"], errors="ignore")

    return d


def add_venue_home_away_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master wrapper that adds all venue-based home/away features:
    - venue_home_win_rate
    - team_venue_win_rate_diff
    - home_away_record_diff

    Handles missing Venue gracefully (fills with median via pipeline imputation).
    """
    d = df.copy()

    # Add venue home win rate
    d = compute_venue_home_win_rate(d)

    # Add team venue win rate difference
    d = compute_team_venue_stats(d)

    # Add home/away record difference
    d = compute_home_away_records(d)

    return d


def add_fixture_venue_features(fixtures: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute latest venue stats from completed games and merge onto future fixtures.
    Uses snapshot values (latest known) for each feature.
    """
    fx = fixtures.copy()

    if "Venue" not in fx.columns:
        fx["venue_home_win_rate"] = GLOBAL_HOME_WIN_RATE
        fx["team_venue_win_rate_diff"] = 0.0
        fx["home_away_record_diff"] = 0.0
        return fx

    fx["Venue"] = fx["Venue"].astype(str).str.strip()
    hist = historical_df.sort_values("date").copy()
    hist["Venue"] = hist["Venue"].astype(str).str.strip() if "Venue" in hist.columns else ""

    # 1) Venue home win rate: compute final rate per venue
    venue_stats = hist.groupby("Venue").agg(
        venue_wins=("home_win", "sum"),
        venue_games=("home_win", "count")
    ).reset_index()
    venue_stats["venue_home_win_rate"] = np.where(
        venue_stats["venue_games"] >= VENUE_MIN_GAMES,
        venue_stats["venue_wins"] / venue_stats["venue_games"],
        GLOBAL_HOME_WIN_RATE
    )

    fx = fx.merge(venue_stats[["Venue", "venue_home_win_rate"]], on="Venue", how="left")
    fx["venue_home_win_rate"] = fx["venue_home_win_rate"].fillna(GLOBAL_HOME_WIN_RATE)

    # 2) Team venue win rate: compute per team-venue
    # Home perspective
    home_tv = hist.groupby(["Venue", "home_team"]).agg(
        wins=("home_win", "sum"),
        games=("home_win", "count")
    ).reset_index()
    home_tv["team_venue_rate"] = np.where(
        home_tv["games"] >= 3,
        home_tv["wins"] / home_tv["games"],
        np.nan
    )
    home_tv = home_tv.rename(columns={"home_team": "team"})

    # Away perspective
    away_tv = hist.copy()
    away_tv["away_win"] = 1 - away_tv["home_win"]
    away_tv = away_tv.groupby(["Venue", "away_team"]).agg(
        wins=("away_win", "sum"),
        games=("away_win", "count")
    ).reset_index()
    away_tv["team_venue_rate"] = np.where(
        away_tv["games"] >= 3,
        away_tv["wins"] / away_tv["games"],
        np.nan
    )
    away_tv = away_tv.rename(columns={"away_team": "team"})

    # Combine
    all_tv = pd.concat([home_tv[["Venue", "team", "team_venue_rate", "games"]],
                        away_tv[["Venue", "team", "team_venue_rate", "games"]]], ignore_index=True)
    all_tv = all_tv.groupby(["Venue", "team"]).agg(
        team_venue_rate=("team_venue_rate", "mean"),
        games=("games", "sum")
    ).reset_index()

    # Compute overall team win rates as fallback
    home_overall = hist.groupby("home_team").agg(wins=("home_win", "sum"), games=("home_win", "count")).reset_index()
    home_overall = home_overall.rename(columns={"home_team": "team"})
    away_overall = hist.copy()
    away_overall["away_win"] = 1 - away_overall["home_win"]
    away_overall = away_overall.groupby("away_team").agg(wins=("away_win", "sum"), games=("away_win", "count")).reset_index()
    away_overall = away_overall.rename(columns={"away_team": "team"})

    team_overall = pd.concat([home_overall, away_overall]).groupby("team").agg(
        wins=("wins", "sum"), games=("games", "sum")
    ).reset_index()
    team_overall["team_overall_rate"] = team_overall["wins"] / team_overall["games"]

    # Merge team-venue rates for home and away teams
    fx = fx.merge(
        all_tv[["Venue", "team", "team_venue_rate"]].rename(columns={"team": "home_team", "team_venue_rate": "home_tv_rate"}),
        on=["Venue", "home_team"], how="left"
    )
    fx = fx.merge(
        all_tv[["Venue", "team", "team_venue_rate"]].rename(columns={"team": "away_team", "team_venue_rate": "away_tv_rate"}),
        on=["Venue", "away_team"], how="left"
    )

    # Fallback to overall rate
    fx = fx.merge(team_overall[["team", "team_overall_rate"]].rename(columns={"team": "home_team", "team_overall_rate": "home_overall"}),
                  on="home_team", how="left")
    fx = fx.merge(team_overall[["team", "team_overall_rate"]].rename(columns={"team": "away_team", "team_overall_rate": "away_overall"}),
                  on="away_team", how="left")

    fx["home_tv_rate"] = fx["home_tv_rate"].fillna(fx["home_overall"]).fillna(0.5)
    fx["away_tv_rate"] = fx["away_tv_rate"].fillna(fx["away_overall"]).fillna(0.5)
    fx["team_venue_win_rate_diff"] = fx["home_tv_rate"] - fx["away_tv_rate"]

    # Clean up
    fx = fx.drop(columns=["home_tv_rate", "away_tv_rate", "home_overall", "away_overall"], errors="ignore")

    # 3) Home/away records: compute final rolling values per team
    # Home record per team (last HOME_AWAY_WINDOW home games)
    home_games = hist[["date", "home_team", "home_win"]].copy()
    home_games = home_games.sort_values("date")
    home_record = home_games.groupby("home_team").tail(HOME_AWAY_WINDOW).groupby("home_team")["home_win"].mean().reset_index()
    home_record = home_record.rename(columns={"home_team": "team", "home_win": "home_record"})

    # Away record per team (last HOME_AWAY_WINDOW away games)
    away_games = hist[["date", "away_team", "home_win"]].copy()
    away_games["away_win"] = 1 - away_games["home_win"]
    away_games = away_games.sort_values("date")
    away_record = away_games.groupby("away_team").tail(HOME_AWAY_WINDOW).groupby("away_team")["away_win"].mean().reset_index()
    away_record = away_record.rename(columns={"away_team": "team", "away_win": "away_record"})

    fx = fx.merge(home_record.rename(columns={"team": "home_team", "home_record": "ht_home_rec"}), on="home_team", how="left")
    fx = fx.merge(away_record.rename(columns={"team": "away_team", "away_record": "at_away_rec"}), on="away_team", how="left")

    fx["ht_home_rec"] = fx["ht_home_rec"].fillna(0.5)
    fx["at_away_rec"] = fx["at_away_rec"].fillna(0.5)
    fx["home_away_record_diff"] = fx["ht_home_rec"] - fx["at_away_rec"]

    fx = fx.drop(columns=["ht_home_rec", "at_away_rec"], errors="ignore")

    return fx


# =========================
# EXTERNAL DATA FEATURES
# =========================
def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features derived from bookmaker spread/line data.

    The spread represents the market's expected margin - this captures
    information about team news, injuries, and sharp money that our
    model doesn't have.

    Features:
    - spread_home: The handicap for home team (negative = home favored)
    - spread_vs_elo: Difference between spread-implied margin and Elo-implied margin
    - spread_move: Movement from open to close (sharp money direction)
    """
    d = df.copy()

    # Check if spread data exists
    spread_cols = ["Home Line Close", "Home Line Open"]
    if not all(c in d.columns for c in spread_cols):
        print("  Spread data not available, skipping spread features")
        d["spread_home"] = 0.0
        d["spread_vs_elo"] = 0.0
        d["spread_move"] = 0.0
        return d

    # Home line is typically negative when home is favored
    d["spread_home"] = pd.to_numeric(d["Home Line Close"], errors="coerce")
    d["spread_home_open"] = pd.to_numeric(d["Home Line Open"], errors="coerce")

    # Spread movement (positive = line moved toward home)
    d["spread_move"] = d["spread_home"] - d["spread_home_open"]

    # Compare spread to Elo prediction
    # Elo diff of ~65 points ≈ 1 goal expected margin
    # So elo_diff / 10 gives rough expected margin
    if "elo_diff" in d.columns:
        elo_implied_margin = d["elo_diff"] / 10
        d["spread_vs_elo"] = (-d["spread_home"]) - elo_implied_margin  # Negative spread means home favored
    else:
        d["spread_vs_elo"] = 0.0

    # Fill NaN
    d["spread_home"] = d["spread_home"].fillna(0)
    d["spread_vs_elo"] = d["spread_vs_elo"].fillna(0)
    d["spread_move"] = d["spread_move"].fillna(0)

    print(f"  Added spread features: spread_home, spread_vs_elo, spread_move")

    return d


def add_total_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features from over/under total score market.

    Low totals might indicate expected bad weather or defensive matchup.
    Total movement might indicate late news affecting game expectations.

    Features:
    - total_score_line: Expected total score
    - total_vs_avg: Difference from season average total
    - total_move: Movement from open to close
    """
    d = df.copy()

    total_cols = ["Total Score Close", "Total Score Open"]
    if not all(c in d.columns for c in total_cols):
        print("  Total score data not available, skipping total features")
        d["total_score_line"] = 0.0
        d["total_vs_avg"] = 0.0
        d["total_move"] = 0.0
        return d

    d["total_score_line"] = pd.to_numeric(d["Total Score Close"], errors="coerce")
    d["total_score_open"] = pd.to_numeric(d["Total Score Open"], errors="coerce")

    # Average total (rolling historical average)
    avg_total = d["total_score_line"].expanding().mean().shift(1).fillna(160)  # ~160 is typical AFL total
    d["total_vs_avg"] = d["total_score_line"] - avg_total

    # Total movement
    d["total_move"] = d["total_score_line"] - d["total_score_open"]

    # Fill NaN
    d["total_score_line"] = d["total_score_line"].fillna(160)
    d["total_vs_avg"] = d["total_vs_avg"].fillna(0)
    d["total_move"] = d["total_move"].fillna(0)

    print(f"  Added total features: total_score_line, total_vs_avg, total_move")

    return d


def add_kickoff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features based on kick-off time.

    Some teams perform differently in day vs night games,
    Friday night vs Saturday afternoon, etc.

    Features:
    - is_night_game: 1 if kickoff after 5pm local
    - is_friday_night: Premium timeslot
    - is_sunday: Often lower-quality matchups
    - day_of_week: 0=Monday, 6=Sunday
    """
    d = df.copy()

    # Try to get kickoff time
    if "Kick Off (local)" in d.columns:
        kickoff = pd.to_datetime(d["Kick Off (local)"], errors="coerce")
        d["kickoff_hour"] = kickoff.dt.hour.fillna(14)  # Default to afternoon
    elif "date" in d.columns:
        d["kickoff_hour"] = 14  # Default if no time available
    else:
        d["kickoff_hour"] = 14

    # Day of week from date
    if "date" in d.columns:
        d["day_of_week"] = pd.to_datetime(d["date"]).dt.dayofweek
    else:
        d["day_of_week"] = 5  # Default to Saturday

    # Derived features
    d["is_night_game"] = (d["kickoff_hour"] >= 17).astype(int)
    d["is_friday_night"] = ((d["day_of_week"] == 4) & (d["is_night_game"] == 1)).astype(int)
    d["is_sunday"] = (d["day_of_week"] == 6).astype(int)

    print(f"  Added kickoff features: is_night_game, is_friday_night, is_sunday, day_of_week")

    return d


def fetch_weather_data(venues: list, dates: list, cache_file: str = WEATHER_CACHE_FILE) -> pd.DataFrame:
    """
    Fetch historical weather data for venues and dates.

    Uses Open-Meteo API (free, no API key required) for historical weather.

    Returns DataFrame with: date, venue, temperature, precipitation, wind_speed
    """
    import urllib.request
    import json

    # Venue coordinates (approximate stadium locations)
    venue_coords = {
        "MCG": (-37.82, 144.98),
        "Marvel Stadium": (-37.82, 144.95),
        "GMHBA Stadium": (-38.16, 144.35),
        "Kardinia Park": (-38.16, 144.35),
        "Gabba": (-27.48, 153.04),
        "SCG": (-33.89, 151.22),
        "Adelaide Oval": (-34.92, 138.60),
        "Optus Stadium": (-31.95, 115.89),
        "Sydney Showground": (-33.84, 151.07),
        "ENGIE Stadium": (-33.84, 151.07),
        "Blundstone Arena": (-42.87, 147.37),
        "TIO Stadium": (-12.43, 130.84),
        "Manuka Oval": (-35.32, 149.13),
        "People First Stadium": (-27.77, 153.09),
        "Mars Stadium": (-37.56, 143.86),
        "UTAS Stadium": (-41.43, 147.14),
        "Cazaly's Stadium": (-16.93, 145.77),
        "Ninja Stadium": (-42.87, 147.32),
    }

    # Check cache first
    if os.path.exists(cache_file):
        try:
            cache = pd.read_csv(cache_file)
            cache["date"] = pd.to_datetime(cache["date"])
            print(f"  Loaded {len(cache)} cached weather records from {cache_file}")
            return cache
        except Exception as e:
            print(f"  Warning: Could not load weather cache: {e}")

    # Build list of unique venue-date pairs to fetch
    weather_records = []

    # Convert dates to date objects
    unique_dates = pd.to_datetime(pd.Series(dates)).dt.date.unique()
    unique_venues = [v for v in pd.Series(venues).unique() if v in venue_coords]

    print(f"  Fetching weather for {len(unique_venues)} venues, {len(unique_dates)} dates...")

    # Group dates by venue to minimize API calls
    for venue in unique_venues:
        if venue not in venue_coords:
            continue

        lat, lon = venue_coords[venue]

        # Get date range for this venue
        min_date = min(unique_dates)
        max_date = max(unique_dates)

        # Open-Meteo historical weather API
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={min_date}&end_date={max_date}"
            f"&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max"
            f"&timezone=auto"
        )

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())

            if "daily" in data:
                daily = data["daily"]
                for i, date_str in enumerate(daily.get("time", [])):
                    weather_records.append({
                        "date": pd.to_datetime(date_str),
                        "Venue": venue,
                        "temperature": daily.get("temperature_2m_max", [None])[i],
                        "precipitation": daily.get("precipitation_sum", [None])[i],
                        "wind_speed": daily.get("wind_speed_10m_max", [None])[i],
                    })

            print(f"    {venue}: fetched {len(daily.get('time', []))} days")

        except Exception as e:
            print(f"    {venue}: API error - {e}")
            continue

    if weather_records:
        weather_df = pd.DataFrame(weather_records)

        # Save to cache
        try:
            weather_df.to_csv(cache_file, index=False)
            print(f"  Saved {len(weather_df)} weather records to {cache_file}")
        except Exception as e:
            print(f"  Warning: Could not save weather cache: {e}")

        return weather_df

    print("  Weather data: No records fetched.")
    return pd.DataFrame(columns=["date", "Venue", "temperature", "precipitation", "wind_speed"])


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather-based features.

    Weather affects AFL significantly:
    - Rain: Lower scoring, more contested ball
    - Wind: Harder to kick accurately
    - Extreme heat: Fatigue factor

    Features:
    - precipitation: Rain amount (mm)
    - wind_speed: Wind speed (km/h)
    - is_wet_game: Binary flag for rain > 2mm
    - is_windy: Binary flag for wind > 25km/h
    """
    d = df.copy()

    # Try to get weather data
    if "Venue" in d.columns and "date" in d.columns:
        venues = d["Venue"].unique().tolist()
        dates = d["date"].unique().tolist()
        weather = fetch_weather_data(venues, dates)

        if len(weather) > 0:
            d = d.merge(weather, on=["date", "Venue"], how="left")
            d["is_wet_game"] = (d["precipitation"] > 2).astype(int)
            d["is_windy"] = (d["wind_speed"] > 25).astype(int)
            print(f"  Added weather features: precipitation, wind_speed, is_wet_game, is_windy")
        else:
            # No weather data available - use neutral defaults
            d["precipitation"] = 0.0
            d["wind_speed"] = 15.0
            d["is_wet_game"] = 0
            d["is_windy"] = 0
            print("  Weather data not available, using defaults")
    else:
        d["precipitation"] = 0.0
        d["wind_speed"] = 15.0
        d["is_wet_game"] = 0
        d["is_windy"] = 0

    return d


def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to add all external data features.
    """
    d = df.copy()

    print("\nAdding external data features:")

    if USE_SPREAD_FEATURES:
        d = add_spread_features(d)

    if USE_TOTAL_FEATURES:
        d = add_total_features(d)

    if USE_KICKOFF_FEATURES:
        d = add_kickoff_features(d)

    if USE_WEATHER_FEATURES:
        d = add_weather_features(d)

    return d


# =========================
# HEAD-TO-HEAD FEATURES
# =========================
def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical head-to-head records between teams.

    Features:
    - h2h_home_win_rate: Home team's historical win rate vs this away team
    - h2h_games: Number of H2H games played (for confidence)
    - h2h_home_streak: Current streak (positive = home team winning streak)
    - h2h_recent_home_wins: Home team wins in last 5 H2H meetings

    All shifted to avoid leakage.
    """
    d = df.sort_values("date").copy()

    # Create matchup key (sorted so A vs B = B vs A)
    d["matchup"] = d.apply(lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1)

    # Track H2H history
    h2h_home_win_rate = []
    h2h_games = []
    h2h_recent_home_wins = []

    # Dictionary to store H2H history: {(team1, team2): [(date, home_team, home_win), ...]}
    h2h_history = {}

    for idx, row in d.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        matchup = row["matchup"]

        # Get prior H2H games
        prior_games = h2h_history.get(matchup, [])

        if len(prior_games) >= H2H_MIN_GAMES:
            # Calculate home team's win rate in this matchup
            # Note: "home" here means the current home team, not who was home historically
            home_wins = sum(1 for g in prior_games if (g["home_team"] == home and g["home_win"] == 1) or
                                                       (g["away_team"] == home and g["home_win"] == 0))
            h2h_rate = home_wins / len(prior_games)

            # Recent form (last 5 H2H)
            recent = prior_games[-5:] if len(prior_games) >= 5 else prior_games
            recent_home_wins = sum(1 for g in recent if (g["home_team"] == home and g["home_win"] == 1) or
                                                         (g["away_team"] == home and g["home_win"] == 0))
        else:
            h2h_rate = 0.5  # Cold start
            recent_home_wins = 0

        h2h_home_win_rate.append(h2h_rate)
        h2h_games.append(len(prior_games))
        h2h_recent_home_wins.append(recent_home_wins)

        # Update history
        if matchup not in h2h_history:
            h2h_history[matchup] = []
        h2h_history[matchup].append({
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "home_win": row["home_win"]
        })

    d["h2h_home_win_rate"] = h2h_home_win_rate
    d["h2h_games"] = h2h_games
    d["h2h_recent_home_wins"] = h2h_recent_home_wins

    # Normalize recent wins to rate (0-1)
    d["h2h_recent_rate"] = d["h2h_recent_home_wins"] / 5.0

    # H2H advantage: how much better is home team's H2H record than 50%
    d["h2h_advantage"] = d["h2h_home_win_rate"] - 0.5

    # Drop temp columns
    d = d.drop(columns=["matchup"], errors="ignore")

    print(f"  Added H2H features: h2h_home_win_rate, h2h_games, h2h_advantage, h2h_recent_rate")

    return d


# =========================
# LADDER POSITION FEATURES
# =========================
def compute_ladder_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ladder position and finals race features.

    Features:
    - home_ladder_pos: Home team's current ladder position (1-18)
    - away_ladder_pos: Away team's current ladder position (1-18)
    - ladder_pos_diff: home_ladder_pos - away_ladder_pos (negative = home higher)
    - home_in_eight: Is home team currently in top 8?
    - away_in_eight: Is away team currently in top 8?
    - home_finals_race: Is home team within 2 games of 8th? (must-win territory)
    - away_finals_race: Is away team within 2 games of 8th?
    - finals_race_diff: Home must-win pressure minus away must-win pressure

    All computed from games prior to current game (shifted).
    """
    d = df.sort_values("date").copy()

    # Track cumulative season records
    # {season: {team: {"wins": n, "losses": n, "percentage": pct}}}
    season_records = {}

    home_ladder_pos = []
    away_ladder_pos = []
    home_wins_season = []
    away_wins_season = []
    home_in_eight = []
    away_in_eight = []
    home_finals_race = []
    away_finals_race = []

    for idx, row in d.iterrows():
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]
        home_score = row.get("home_score", 0)
        away_score = row.get("away_score", 0)

        # Initialize season if needed
        if season not in season_records:
            season_records[season] = {}

        records = season_records[season]

        # Initialize teams if needed
        for team in [home, away]:
            if team not in records:
                records[team] = {"wins": 0, "losses": 0, "pf": 0, "pa": 0}

        # Calculate current ladder (before this game)
        ladder = []
        for team, rec in records.items():
            games = rec["wins"] + rec["losses"]
            if games > 0:
                pct = (rec["pf"] / rec["pa"]) if rec["pa"] > 0 else 1.0
            else:
                pct = 1.0
            ladder.append({
                "team": team,
                "wins": rec["wins"],
                "losses": rec["losses"],
                "games": games,
                "pct": pct,
                "points": rec["wins"] * 4  # AFL uses 4 points per win
            })

        # Sort by points, then percentage
        ladder.sort(key=lambda x: (-x["points"], -x["pct"]))

        # Get positions
        team_positions = {t["team"]: i + 1 for i, t in enumerate(ladder)}
        home_pos = team_positions.get(home, 9)  # Default to 9th (outside 8) if new team
        away_pos = team_positions.get(away, 9)

        # Get wins
        home_w = records[home]["wins"]
        away_w = records[away]["wins"]

        # Check if in top 8
        home_in_8 = home_pos <= FINALS_CUTOFF
        away_in_8 = away_pos <= FINALS_CUTOFF

        # Finals race: within 2 wins of 8th place
        # Find 8th place wins
        if len(ladder) >= FINALS_CUTOFF:
            eighth_place_wins = ladder[FINALS_CUTOFF - 1]["wins"]
        else:
            eighth_place_wins = 0

        home_fr = abs(home_w - eighth_place_wins) <= 2 and home_pos >= 5 and home_pos <= 12
        away_fr = abs(away_w - eighth_place_wins) <= 2 and away_pos >= 5 and away_pos <= 12

        # Store values
        home_ladder_pos.append(home_pos)
        away_ladder_pos.append(away_pos)
        home_wins_season.append(home_w)
        away_wins_season.append(away_w)
        home_in_eight.append(int(home_in_8))
        away_in_eight.append(int(away_in_8))
        home_finals_race.append(int(home_fr))
        away_finals_race.append(int(away_fr))

        # Update records AFTER extracting features (avoid leakage)
        home_win = row["home_win"]
        records[home]["wins"] += home_win
        records[home]["losses"] += 1 - home_win
        records[home]["pf"] += home_score
        records[home]["pa"] += away_score

        records[away]["wins"] += 1 - home_win
        records[away]["losses"] += home_win
        records[away]["pf"] += away_score
        records[away]["pa"] += home_score

    d["home_ladder_pos"] = home_ladder_pos
    d["away_ladder_pos"] = away_ladder_pos
    d["ladder_pos_diff"] = np.array(home_ladder_pos) - np.array(away_ladder_pos)
    d["home_wins_season"] = home_wins_season
    d["away_wins_season"] = away_wins_season
    d["season_wins_diff"] = np.array(home_wins_season) - np.array(away_wins_season)
    d["home_in_eight"] = home_in_eight
    d["away_in_eight"] = away_in_eight
    d["home_finals_race"] = home_finals_race
    d["away_finals_race"] = away_finals_race

    # Combined features
    d["both_in_eight"] = (np.array(home_in_eight) & np.array(away_in_eight)).astype(int)
    d["neither_in_eight"] = (~np.array(home_in_eight).astype(bool) & ~np.array(away_in_eight).astype(bool)).astype(int)
    d["finals_race_diff"] = np.array(home_finals_race) - np.array(away_finals_race)

    print(f"  Added ladder features: ladder_pos_diff, season_wins_diff, home_in_eight, finals_race_diff, etc.")

    return d


def add_h2h_ladder_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to add H2H and ladder features.
    """
    d = df.copy()

    print("\nAdding H2H and ladder features:")

    if USE_H2H_FEATURES:
        d = compute_h2h_features(d)

    if USE_LADDER_FEATURES:
        d = compute_ladder_features(d)

    return d


# =========================
# MODEL FIT / PREDICT
# =========================
def fit_model(train_df: pd.DataFrame, feats: list[str]) -> Pipeline:
    X = train_df[feats]
    y = train_df["home_win"].astype(int)

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=500,
            C=0.2,
            solver="lbfgs"
        ))

    ])
    model.fit(X, y)
    return model


def predict(model: Pipeline, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return model.predict_proba(df[feats])[:, 1]


def evaluate_model_vs_market_holdout(matches_raw: pd.DataFrame, train_end_season: int, test_season: int):
    mw = merge_odds_meta(matches_raw, ODDS_FILE)

    if USE_VENUE_REST_TRAVEL:
        mw = add_rest_features(mw)
        mw = add_travel_features(mw)

    # Filter to regular season (odds file flags finals)
    mw["Play Off Game?"] = mw.get("Play Off Game?", "N")
    regular_all = mw[mw["Play Off Game?"].fillna("N") != "Y"].copy()

    # -----------------------------
    # Build features FIRST (Elo + form) on full dataset
    # -----------------------------
    both = regular_all.sort_values("date").reset_index(drop=True)
    both = compute_elo_diff(both, k=ELO_K, home_adv=ELO_HOME_ADV)

    team_log = load_team_rows()
    both = add_form_features(both, team_log)

    # Add venue home/away features
    if USE_VENUE_HOME_AWAY:
        both = add_venue_home_away_features(both)

    # Add external data features (spread, total, kickoff)
    if USE_EXTERNAL_DATA:
        both = add_external_features(both)

    # Add H2H and ladder features
    if USE_H2H_FEATURES or USE_LADDER_FEATURES:
        both = add_h2h_ladder_features(both)

    # -----------------------------
    # Re-split AFTER features exist
    # -----------------------------
    train2 = both[both["season"] <= train_end_season].copy()
    test2 = both[both["season"] == test_season].copy()
    val2 = both[both["season"] == train_end_season].copy()  # 2024 slice for calibration

    print(f"\nHoldout evaluation: train<= {train_end_season}, test= {test_season}")
    print("Rows (regular season):", len(test2))

    # -----------------------------
    # Build feature lists
    # -----------------------------
    resolved_stats = resolve_model_stats(team_log, MODEL_STATS)

    # Baseline: Elo + form only
    baseline_feats = ["elo_diff"]
    for s in resolved_stats:
        baseline_feats.append(f"diff_form_{s}")
    if USE_VENUE_REST_TRAVEL:
        baseline_feats += ["rest_diff", "travel_diff", "neutral_venue", "both_interstate"]

    # Enhanced: baseline + external data features
    feats = baseline_feats.copy()

    if USE_VENUE_HOME_AWAY:
        feats += ["venue_home_win_rate", "team_venue_win_rate_diff", "home_away_record_diff"]

    if USE_EXTERNAL_DATA:
        if USE_SPREAD_FEATURES:
            feats += ["spread_home", "spread_vs_elo", "spread_move"]
        if USE_TOTAL_FEATURES:
            feats += ["total_vs_avg", "total_move"]
        if USE_KICKOFF_FEATURES:
            feats += ["is_night_game", "is_friday_night"]
        if USE_WEATHER_FEATURES:
            feats += ["precipitation", "wind_speed", "is_wet_game", "is_windy"]

    if USE_H2H_FEATURES:
        feats += ["h2h_home_win_rate", "h2h_advantage", "h2h_recent_rate"]

    if USE_LADDER_FEATURES:
        feats += ["ladder_pos_diff", "season_wins_diff", "home_in_eight", "away_in_eight", "finals_race_diff"]

    print(f"\nBaseline features ({len(baseline_feats)}): {baseline_feats}")
    print(f"Enhanced features ({len(feats)}): {feats}")

    # -----------------------------
    # Fit models on TRAIN and predict on TRAIN/VAL/TEST
    # -----------------------------
    # Fit baseline model (Elo + form)
    baseline_model = fit_model(train2, baseline_feats)
    train2["p_baseline"] = predict(baseline_model, train2, baseline_feats)
    test2["p_baseline"] = predict(baseline_model, test2, baseline_feats)
    val2["p_baseline"] = predict(baseline_model, val2, baseline_feats)

    # Fit enhanced model (+ external data)
    model = fit_model(train2, feats)

    # predict raw probs everywhere we care about
    train2["p_home_win"] = predict(model, train2, feats)
    val2["p_home_win"] = predict(model, val2, feats)
    test2["p_home_win"] = predict(model, test2, feats)

    # -----------------------------
    # Platt calibration fit on 2024 (val2)
    # -----------------------------
    # Calibrate baseline model
    a_base, b_base = fit_platt_calibrator(val2["p_baseline"].values, val2["home_win"].values)
    train2["p_baseline_cal"] = apply_platt(train2["p_baseline"].values, a_base, b_base)
    val2["p_baseline_cal"] = apply_platt(val2["p_baseline"].values, a_base, b_base)
    test2["p_baseline_cal"] = apply_platt(test2["p_baseline"].values, a_base, b_base)

    # Calibrate venue-enhanced model
    a, b = fit_platt_calibrator(val2["p_home_win"].values, val2["home_win"].values)
    print(f"\nPlatt calibrator - baseline (fit on {train_end_season}): a={a_base:.4f}, b={b_base:.4f}")
    print(f"Platt calibrator - venue-enhanced (fit on {train_end_season}): a={a:.4f}, b={b:.4f}")

    # apply calibrated probs to train/test (and val for reporting)
    train2["p_home_win_cal"] = apply_platt(train2["p_home_win"].values, a, b)
    val2["p_home_win_cal"] = apply_platt(val2["p_home_win"].values, a, b)
    test2["p_home_win_cal"] = apply_platt(test2["p_home_win"].values, a, b)

    # Print feature coefficients
    print("\n=== Feature Coefficients (Baseline Model) ===")
    clf_base = baseline_model.named_steps["clf"]
    coefs_base = clf_base.coef_[0]
    for feat_name, coef in zip(baseline_feats, coefs_base):
        print(f"  {feat_name}: {coef:.4f}")

    print("\n=== Feature Coefficients (Enhanced Model) ===")
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]
    for feat_name, coef in zip(feats, coefs):
        print(f"  {feat_name}: {coef:.4f}")

    # -----------------------------
    # Market columns (prefer no-vig)
    # -----------------------------
    market_open = "p_home_open_nv" if "p_home_open_nv" in test2.columns else "p_home_open"
    market_close = "p_home_close_nv" if "p_home_close_nv" in test2.columns else "p_home_close"

    # ============================================================



    # -----------------------------
    # Tune hybrid on 2022-2024 (calibrated probs now exist)
    # -----------------------------
    TUNE_START_SEASON = 2022
    TUNE_END_SEASON = 2024

    tune_df = train2[(train2["season"] >= TUNE_START_SEASON) & (train2["season"] <= TUNE_END_SEASON)].copy()
    tune_df = tune_df.dropna(subset=[market_open, "p_home_win_cal"]).copy()

    print("tune_df rows:", len(tune_df))
    print("tune_df has p_home_win_cal?", "p_home_win_cal" in tune_df.columns)

    best_hyb = tune_hybrid_rule(
        tune_df,
        model_col="p_home_win_cal",
        market_col=market_open,
        delta_grid=[0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
        band_grid=[(0.47, 0.53), (0.45, 0.55), (0.40, 0.60), (0.35, 0.65), (0.30, 0.70)],
        max_delta_grid=[None],  # <--- forces no cap
    )

    print(f"\nBest hybrid params (tuned on {TUNE_START_SEASON}-{TUNE_END_SEASON}): {best_hyb}")

    # ============================================================
    # DIAGNOSE: why do tuned overrides = 0 in 2025?
    # NOTE: run this AFTER best_hyb exists and AFTER test2 has p_home_win_cal
    # ============================================================
    pm = pd.to_numeric(test2["p_home_win_cal"], errors="coerce").to_numpy(dtype=float)
    pk = pd.to_numeric(test2[market_open], errors="coerce").to_numpy(dtype=float)

    delta = pm - pk
    abs_delta = np.abs(delta)

    disagree = (pm >= 0.5) != (pk >= 0.5)
    in_band = (pk >= float(best_hyb["band"][0])) & (pk <= float(best_hyb["band"][1]))
    ge_dt = abs_delta >= float(best_hyb["delta_thresh"])

    md = best_hyb.get("max_delta", None)
    if md is None or (isinstance(md, float) and np.isnan(md)):
        le_md = np.ones_like(ge_dt, dtype=bool)
    else:
        le_md = abs_delta <= float(md)

    finite = np.isfinite(pm) & np.isfinite(pk)
    all_ok = finite & disagree & in_band & ge_dt & le_md

    mask = (finite & disagree & in_band & ge_dt)  # everything except md
    bad = test2.loc[mask, ["date", "home_team", "away_team", market_open, "p_home_win_cal"]].copy()
    bad["abs_delta"] = (bad["p_home_win_cal"] - bad[market_open]).abs()
    print(bad.sort_values("abs_delta", ascending=False).head(20).to_string(index=False))

    print("\n--- Tuned hybrid gate counts on 2025 ---")
    print("Total games:", int(len(test2)))
    print("Finite pm/pk:", int(finite.sum()))
    print("Disagree side:", int((finite & disagree).sum()))
    print("In market band:", int((finite & in_band).sum()))
    print("|delta| >= dt:", int((finite & ge_dt).sum()))
    print("|delta| <= md:", int((finite & le_md).sum()))
    print("ALL gates met (override):", int(all_ok.sum()))

    print("\n--- Gate intersections (2025) ---")
    print("Disagree & in_band:", int((finite & disagree & in_band).sum()))
    print("Disagree & |delta|>=dt:", int((finite & disagree & ge_dt).sum()))
    print("In_band & |delta|>=dt:", int((finite & in_band & ge_dt).sum()))
    print("Disagree & in_band & |delta|>=dt:", int((finite & disagree & in_band & ge_dt).sum()))
    print("...and <= md:", int((finite & disagree & in_band & ge_dt & le_md).sum()))

    # apply tuned hybrid to TEST (2025)
    test2 = add_hybrid_tip(
        test2,
        model_col="p_home_win_cal",
        market_col=market_open,
        delta_thresh=best_hyb["delta_thresh"],
        market_band=best_hyb["band"],
        max_delta=best_hyb.get("max_delta", None),
        out_col="p_hybrid_tuned",
    )

    # --- tuned override diagnostics on test set (2025) ---
    tuned_override_col = "hybrid_tuned_override"
    tuned_delta_col = "hybrid_tuned_delta"

    tuned_overrides = int(test2[tuned_override_col].sum())
    print("\nTuned-hybrid override count (2025):", tuned_overrides)

    if tuned_overrides > 0:
        tips_bin = (test2["p_hybrid_tuned"].astype(float) >= 0.5).astype(int)
        hit = float(
            (tips_bin[test2[tuned_override_col] == 1] ==
             test2.loc[test2[tuned_override_col] == 1, "home_win"].astype(int)).mean()
        )
        print("Tuned-hybrid override hit rate (2025):", hit)

        cols = ["date", "home_team", "away_team", "home_win",
                market_open, "p_home_win_cal", "p_hybrid_tuned", tuned_delta_col]
        print("\n=== Tuned override games (2025) ===")
        print(test2.loc[test2[tuned_override_col] == 1, cols].sort_values("date").to_string(index=False))
    else:
        print("Tuned-hybrid override hit rate (2025): nan")

    # -----------------------------
    # Manual hybrid benchmark
    # -----------------------------
    test2 = add_hybrid_tip(
        test2,
        model_col="p_home_win_cal",
        market_col=market_open,
        delta_thresh=0.15,
        market_band=(0.35, 0.65),
        max_delta=None,
        out_col="p_hybrid_manual",
    )

    # -----------------------------
    # Blend baseline (use CALIBRATED model)
    # -----------------------------
    # -----------------------------
    # Tune blend on 2022-2024
    # -----------------------------
    blend_tune_df = train2[(train2["season"] >= TUNE_START_SEASON) &
                           (train2["season"] <= TUNE_END_SEASON)].copy()
    blend_tune_df = blend_tune_df.dropna(subset=[market_open, "p_home_win_cal"]).copy()

    # Fine-grained blend weight search
    w_grid = np.linspace(0.0, 1.0, 101)  # 0.00, 0.01, 0.02, ..., 1.00

    best_blend = tune_blend_weight(
        blend_tune_df,
        model_col="p_home_win_cal",
        market_col=market_open,
        w_grid=w_grid
    )

    print(f"\nBest blend weight (tuned on {TUNE_START_SEASON}-{TUNE_END_SEASON}): {best_blend}")

    # Show blend weight sensitivity
    print("\n--- Blend Weight Sensitivity (on tuning set) ---")
    blend_results = []
    for w_test in [0.0, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        p_test = w_test * blend_tune_df[market_open] + (1 - w_test) * blend_tune_df["p_home_win_cal"]
        tips_test = (p_test >= 0.5).astype(int)
        correct_test = int((tips_test == blend_tune_df["home_win"].astype(int)).sum())
        blend_results.append({"w": w_test, "correct": correct_test, "acc": correct_test / len(blend_tune_df)})
    print(pd.DataFrame(blend_results).to_string(index=False))

    # Compare tuned blend vs pure market on the tuning set
    blend_tune_df = blend_tune_df.copy()
    blend_tune_df["p_blend_tune"] = best_blend["w"] * blend_tune_df[market_open] + (1 - best_blend["w"]) * \
                                    blend_tune_df["p_home_win_cal"]

    print("\nTuning-set tipping (2022-2024):")
    print("Market:", evaluate_tipping(blend_tune_df, market_open, "Market"))
    print("Blend :", evaluate_tipping(blend_tune_df, "p_blend_tune", f"Blend (w={best_blend['w']:.2f})"))
    print("Model :", evaluate_tipping(blend_tune_df, "p_home_win_cal", "Model"))

    w = best_blend["w"]
    test2["p_blend"] = w * test2[market_open] + (1 - w) * test2["p_home_win_cal"]

    # -----------------------------
    # Output tables (probability metrics)
    # -----------------------------
    rows = [
        metrics_block(test2, market_close, "Market (Close, no-vig)"),
        metrics_block(test2, market_open, "Market (Open, no-vig)"),
        metrics_block(test2, "p_blend", f"Blend (w={w})"),
        metrics_block(test2, "p_baseline_cal", "Model (Elo+form baseline)"),
        metrics_block(test2, "p_home_win_cal", "Model (+ external data)"),
    ]
    out = pd.DataFrame(rows).sort_values("log_loss")

    print(f"\n=== Model vs Market ({test_season} regular season) ===")
    print(out.to_string(index=False))

    # -----------------------------
    # Tipping accuracy (print once)
    # -----------------------------
    tips = pd.DataFrame([
        evaluate_tipping(test2, market_open, "Tip: Market favourite (Open, no-vig)"),
        evaluate_tipping(test2, "p_baseline_cal", "Tip: Elo+form baseline"),
        evaluate_tipping(test2, "p_home_win_cal", "Tip: + external data"),
        evaluate_tipping(test2, "p_blend", f"Tip: Blend (w={w})"),
        evaluate_tipping(test2, "p_hybrid_manual", "Tip: Hybrid (manual rule)"),
        evaluate_tipping(test2, "p_hybrid_tuned", f"Tip: Hybrid TUNED on {TUNE_START_SEASON}-{TUNE_END_SEASON}"),
    ])

    print(f"\n=== Tipping accuracy ({test_season} regular season) ===")
    print(tips.to_string(index=False))

    # --- Paired compare: tuned hybrid vs market open favourite (2025) ---
    y = test2["home_win"].astype(int).to_numpy()
    a = (test2["p_hybrid_tuned"].astype(float).to_numpy() >= 0.5).astype(int)
    b = (test2[market_open].astype(float).to_numpy() >= 0.5).astype(int)

    a_ok = (a == y)
    b_ok = (b == y)

    both_ok = int((a_ok & b_ok).sum())
    both_bad = int((~a_ok & ~b_ok).sum())
    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())

    print(f"\n=== Paired tipping compare: Hybrid tuned (2022-2024) vs Market open favourite ===")
    print(f"Both correct: {both_ok}")
    print(f"Both wrong:   {both_bad}")
    print(f"Hybrid tuned only correct: {a_only}")
    print(f"Market only correct:       {b_only}")
    print(f"Net advantage (Hybrid - Market): {a_only - b_only}")

    # Paired compare: enhanced vs baseline model
    y_cmp = test2["home_win"].astype(int).to_numpy()
    enhanced_tips = (test2["p_home_win_cal"].astype(float).to_numpy() >= 0.5).astype(int)
    base_tips = (test2["p_baseline_cal"].astype(float).to_numpy() >= 0.5).astype(int)
    mkt_tips = (test2[market_open].astype(float).to_numpy() >= 0.5).astype(int)

    enhanced_ok = (enhanced_tips == y_cmp)
    base_ok = (base_tips == y_cmp)
    mkt_ok = (mkt_tips == y_cmp)

    # Enhanced vs Baseline
    eb_both_ok = int((enhanced_ok & base_ok).sum())
    eb_both_bad = int((~enhanced_ok & ~base_ok).sum())
    eb_enh_only = int((enhanced_ok & ~base_ok).sum())
    eb_base_only = int((~enhanced_ok & base_ok).sum())

    print(f"\n=== Paired tipping compare: Enhanced (+ external) vs Baseline (Elo+form) ===")
    print(f"Both correct: {eb_both_ok}")
    print(f"Both wrong:   {eb_both_bad}")
    print(f"Enhanced only correct: {eb_enh_only}")
    print(f"Baseline only correct: {eb_base_only}")
    print(f"Net advantage (Enhanced - Baseline): {eb_enh_only - eb_base_only}")
    p_val = mcnemar_exact(eb_enh_only, eb_base_only)
    print(f"McNemar p-value: {p_val:.4f}")

    # Enhanced vs Market
    em_both_ok = int((enhanced_ok & mkt_ok).sum())
    em_both_bad = int((~enhanced_ok & ~mkt_ok).sum())
    em_enh_only = int((enhanced_ok & ~mkt_ok).sum())
    em_mkt_only = int((~enhanced_ok & mkt_ok).sum())

    print(f"\n=== Paired tipping compare: Enhanced (+ external) vs Market (Open, no-vig) ===")
    print(f"Both correct: {em_both_ok}")
    print(f"Both wrong:   {em_both_bad}")
    print(f"Enhanced only correct: {em_enh_only}")
    print(f"Market only correct: {em_mkt_only}")
    print(f"Net advantage (Enhanced - Market): {em_enh_only - em_mkt_only}")
    p_val_em = mcnemar_exact(em_enh_only, em_mkt_only)
    print(f"McNemar p-value: {p_val_em:.4f}")

    # =========================================================================
    # CLOSING LINE VALUE (CLV) ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CLOSING LINE VALUE (CLV) ANALYSIS")
    print("=" * 70)
    print("CLV = Model probability - Closing market probability")
    print("Positive CLV = Model was sharper than closing line")
    print("=" * 70)

    # Calculate CLV
    test2["clv"] = test2["p_home_win_cal"] - test2[market_close]
    test2["clv_baseline"] = test2["p_baseline_cal"] - test2[market_close]

    # Also calculate CLV vs opening line (less efficient, easier to beat)
    test2["clv_vs_open"] = test2["p_home_win_cal"] - test2[market_open]

    # Model's predicted side (True = home, False = away)
    test2["model_side_home"] = test2["p_home_win_cal"] >= 0.5
    test2["market_close_side_home"] = test2[market_close] >= 0.5
    test2["market_open_side_home"] = test2[market_open] >= 0.5

    # Signed CLV (positive when model is "more right" about winning side)
    # If model picks home and home wins, positive CLV is good
    # If model picks away and away wins, negative CLV (model said lower home prob) is good
    test2["signed_clv"] = np.where(
        test2["home_win"] == 1,
        test2["clv"],  # Home won: higher home prob = better
        -test2["clv"]  # Away won: lower home prob = better
    )

    # Overall CLV stats
    print(f"\n--- Overall CLV Statistics ({test_season}) ---")
    print(f"Games analyzed: {len(test2)}")
    print(f"Mean CLV vs Close: {test2['clv'].mean():.4f} ({test2['clv'].mean()*100:.2f}%)")
    print(f"Mean CLV vs Open:  {test2['clv_vs_open'].mean():.4f} ({test2['clv_vs_open'].mean()*100:.2f}%)")
    print(f"Mean Signed CLV:   {test2['signed_clv'].mean():.4f} ({test2['signed_clv'].mean()*100:.2f}%)")
    print(f"Baseline CLV:      {test2['clv_baseline'].mean():.4f} ({test2['clv_baseline'].mean()*100:.2f}%)")

    # CLV by model confidence level
    print(f"\n--- CLV by Model Confidence Level ---")
    test2["model_confidence"] = (test2["p_home_win_cal"] - 0.5).abs()
    confidence_bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
    confidence_labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-80%", "80%+"]
    test2["confidence_bin"] = pd.cut(test2["model_confidence"], bins=confidence_bins, labels=confidence_labels)

    conf_analysis = test2.groupby("confidence_bin", observed=True).agg(
        n=("clv", "count"),
        mean_clv=("clv", "mean"),
        mean_signed_clv=("signed_clv", "mean"),
        model_acc=("home_win", lambda x: ((test2.loc[x.index, "p_home_win_cal"] >= 0.5) == x).mean()),
        market_acc=("home_win", lambda x: ((test2.loc[x.index, market_close] >= 0.5) == x).mean()),
    ).reset_index()
    conf_analysis["mean_clv_pct"] = conf_analysis["mean_clv"] * 100
    conf_analysis["mean_signed_clv_pct"] = conf_analysis["mean_signed_clv"] * 100
    print(conf_analysis.to_string(index=False))

    # CLV by agreement/disagreement with market
    print(f"\n--- CLV by Model vs Market Agreement ---")
    test2["agrees_with_close"] = test2["model_side_home"] == test2["market_close_side_home"]
    test2["agrees_with_open"] = test2["model_side_home"] == test2["market_open_side_home"]

    agree_analysis = test2.groupby("agrees_with_close", observed=True).agg(
        n=("clv", "count"),
        mean_clv=("clv", "mean"),
        mean_signed_clv=("signed_clv", "mean"),
        model_acc=("home_win", lambda x: ((test2.loc[x.index, "p_home_win_cal"] >= 0.5) == x).mean()),
        market_acc=("home_win", lambda x: ((test2.loc[x.index, market_close] >= 0.5) == x).mean()),
    ).reset_index()
    agree_analysis["agrees_with_close"] = agree_analysis["agrees_with_close"].map({True: "Agrees", False: "Disagrees"})
    agree_analysis["mean_clv_pct"] = agree_analysis["mean_clv"] * 100
    print(agree_analysis.to_string(index=False))

    # CLV when model disagrees AND is confident
    print(f"\n--- CLV on High-Confidence Disagreements ---")
    disagree_confident = test2[
        (~test2["agrees_with_close"]) &
        (test2["model_confidence"] >= 0.10)
    ]
    if len(disagree_confident) > 0:
        print(f"Games where model disagrees with market AND confidence >= 60%: {len(disagree_confident)}")
        print(f"Mean CLV: {disagree_confident['clv'].mean():.4f} ({disagree_confident['clv'].mean()*100:.2f}%)")
        print(f"Mean Signed CLV: {disagree_confident['signed_clv'].mean():.4f}")
        print(f"Model accuracy: {((disagree_confident['p_home_win_cal'] >= 0.5) == disagree_confident['home_win']).mean():.3f}")
        print(f"Market accuracy: {((disagree_confident[market_close] >= 0.5) == disagree_confident['home_win']).mean():.3f}")
    else:
        print("No high-confidence disagreements found.")

    # CLV by delta from market (how much model differs)
    print(f"\n--- CLV by Model-Market Delta Size ---")
    test2["abs_delta_close"] = (test2["p_home_win_cal"] - test2[market_close]).abs()
    delta_bins = [0, 0.05, 0.10, 0.15, 0.20, 1.0]
    delta_labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]
    test2["delta_bin"] = pd.cut(test2["abs_delta_close"], bins=delta_bins, labels=delta_labels)

    delta_analysis = test2.groupby("delta_bin", observed=True).agg(
        n=("clv", "count"),
        mean_clv=("clv", "mean"),
        mean_signed_clv=("signed_clv", "mean"),
        model_acc=("home_win", lambda x: ((test2.loc[x.index, "p_home_win_cal"] >= 0.5) == x).mean()),
        market_acc=("home_win", lambda x: ((test2.loc[x.index, market_close] >= 0.5) == x).mean()),
    ).reset_index()
    delta_analysis["mean_clv_pct"] = delta_analysis["mean_clv"] * 100
    print(delta_analysis.to_string(index=False))

    # Line movement analysis
    print(f"\n--- Line Movement Analysis ---")
    test2["line_move"] = test2[market_close] - test2[market_open]  # Positive = moved toward home
    test2["model_agrees_with_move"] = (
        ((test2["p_home_win_cal"] > test2[market_open]) & (test2["line_move"] > 0)) |
        ((test2["p_home_win_cal"] < test2[market_open]) & (test2["line_move"] < 0))
    )

    move_analysis = test2.groupby("model_agrees_with_move", observed=True).agg(
        n=("clv", "count"),
        mean_signed_clv=("signed_clv", "mean"),
        model_acc=("home_win", lambda x: ((test2.loc[x.index, "p_home_win_cal"] >= 0.5) == x).mean()),
    ).reset_index()
    move_analysis["model_agrees_with_move"] = move_analysis["model_agrees_with_move"].map(
        {True: "Model agrees with line move", False: "Model opposes line move"}
    )
    print(move_analysis.to_string(index=False))

    # Best and worst CLV games
    print(f"\n--- Top 10 Positive CLV Games (Model sharper than close) ---")
    top_clv = test2.nlargest(10, "signed_clv")[
        ["date", "home_team", "away_team", "home_win", "p_home_win_cal", market_close, "clv", "signed_clv"]
    ].copy()
    top_clv["clv_pct"] = top_clv["clv"] * 100
    top_clv["result"] = top_clv["home_win"].map({1: "Home W", 0: "Away W"})
    print(top_clv[["date", "home_team", "away_team", "result", "p_home_win_cal", market_close, "clv_pct"]].to_string(index=False))

    print(f"\n--- Top 10 Negative CLV Games (Market sharper than model) ---")
    bottom_clv = test2.nsmallest(10, "signed_clv")[
        ["date", "home_team", "away_team", "home_win", "p_home_win_cal", market_close, "clv", "signed_clv"]
    ].copy()
    bottom_clv["clv_pct"] = bottom_clv["clv"] * 100
    bottom_clv["result"] = bottom_clv["home_win"].map({1: "Home W", 0: "Away W"})
    print(bottom_clv[["date", "home_team", "away_team", "result", "p_home_win_cal", market_close, "clv_pct"]].to_string(index=False))

    # Summary recommendation
    print(f"\n--- CLV Summary & Recommendations ---")
    overall_clv = test2["signed_clv"].mean()
    if overall_clv > 0.01:
        print(f"POSITIVE signed CLV ({overall_clv:.4f}): Model may have genuine edge")
    elif overall_clv < -0.01:
        print(f"NEGATIVE signed CLV ({overall_clv:.4f}): Market is sharper than model")
    else:
        print(f"NEUTRAL signed CLV ({overall_clv:.4f}): Model roughly matches market efficiency")

    # Find any promising subsets
    promising = []
    for label, subset in [
        ("High confidence (65%+)", test2[test2["model_confidence"] >= 0.15]),
        ("Disagrees with market", test2[~test2["agrees_with_close"]]),
        ("Large delta (15%+)", test2[test2["abs_delta_close"] >= 0.15]),
        ("Agrees with line move", test2[test2["model_agrees_with_move"]]),
    ]:
        if len(subset) >= 10:
            subset_clv = subset["signed_clv"].mean()
            subset_acc = ((subset["p_home_win_cal"] >= 0.5) == subset["home_win"]).mean()
            mkt_acc = ((subset[market_close] >= 0.5) == subset["home_win"]).mean()
            if subset_clv > 0.02 or subset_acc > mkt_acc:
                promising.append({
                    "subset": label,
                    "n": len(subset),
                    "signed_clv": subset_clv,
                    "model_acc": subset_acc,
                    "market_acc": mkt_acc,
                    "edge": subset_acc - mkt_acc
                })

    if promising:
        print("\nPotentially promising subsets:")
        for p in promising:
            print(f"  {p['subset']}: n={p['n']}, CLV={p['signed_clv']:.3f}, Model={p['model_acc']:.1%}, Market={p['market_acc']:.1%}, Edge={p['edge']:+.1%}")
    else:
        print("\nNo subsets found with clear model edge over market.")

    # =========================================================================
    # GAME TYPE ANALYSIS - Where does the model perform best?
    # =========================================================================
    print("\n" + "=" * 70)
    print("GAME TYPE ANALYSIS - Finding Model Edge")
    print("=" * 70)

    def analyze_subset(df, subset_mask, label):
        """Analyze model vs market performance on a subset of games."""
        subset = df[subset_mask]
        if len(subset) < 5:
            return None

        model_tips = (subset["p_home_win_cal"] >= 0.5).astype(int)
        market_tips = (subset[market_open] >= 0.5).astype(int)
        blend_tips = (subset["p_blend"] >= 0.5).astype(int)
        actual = subset["home_win"].astype(int)

        model_correct = int((model_tips == actual).sum())
        market_correct = int((market_tips == actual).sum())
        blend_correct = int((blend_tips == actual).sum())

        return {
            "subset": label,
            "n": len(subset),
            "model_acc": model_correct / len(subset),
            "market_acc": market_correct / len(subset),
            "blend_acc": blend_correct / len(subset),
            "model_correct": model_correct,
            "market_correct": market_correct,
            "blend_correct": blend_correct,
            "edge_vs_market": model_correct - market_correct,
            "blend_edge": blend_correct - market_correct,
        }

    game_type_results = []

    # 1. By day of week
    print("\n--- By Day of Week ---")
    for dow, dow_name in [(4, "Friday"), (5, "Saturday"), (6, "Sunday"), (-1, "Other")]:
        if dow == -1:
            mask = ~test2["day_of_week"].isin([4, 5, 6])
        else:
            mask = test2["day_of_week"] == dow
        result = analyze_subset(test2, mask, dow_name)
        if result:
            game_type_results.append(result)

    dow_df = pd.DataFrame([r for r in game_type_results if r])
    if len(dow_df) > 0:
        print(dow_df[["subset", "n", "model_acc", "market_acc", "blend_acc", "edge_vs_market", "blend_edge"]].to_string(index=False))

    # 2. By time of day
    print("\n--- By Time of Day ---")
    game_type_results = []
    for label, mask in [
        ("Day game", test2["is_night_game"] == 0),
        ("Night game", test2["is_night_game"] == 1),
        ("Friday night", test2["is_friday_night"] == 1),
    ]:
        result = analyze_subset(test2, mask, label)
        if result:
            game_type_results.append(result)

    time_df = pd.DataFrame([r for r in game_type_results if r])
    if len(time_df) > 0:
        print(time_df[["subset", "n", "model_acc", "market_acc", "blend_acc", "edge_vs_market", "blend_edge"]].to_string(index=False))

    # 3. By market confidence (how close is the market?)
    print("\n--- By Market Confidence ---")
    game_type_results = []
    test2["market_confidence"] = (test2[market_open] - 0.5).abs()

    for label, lo, hi in [
        ("Toss-up (45-55%)", 0.0, 0.05),
        ("Lean (55-65%)", 0.05, 0.15),
        ("Clear fav (65-75%)", 0.15, 0.25),
        ("Strong fav (75%+)", 0.25, 0.5),
    ]:
        mask = (test2["market_confidence"] >= lo) & (test2["market_confidence"] < hi)
        result = analyze_subset(test2, mask, label)
        if result:
            game_type_results.append(result)

    conf_df = pd.DataFrame([r for r in game_type_results if r])
    if len(conf_df) > 0:
        print(conf_df[["subset", "n", "model_acc", "market_acc", "blend_acc", "edge_vs_market", "blend_edge"]].to_string(index=False))

    # 4. By season phase
    print("\n--- By Season Phase ---")
    game_type_results = []
    test2["month"] = pd.to_datetime(test2["date"]).dt.month

    for label, months in [
        ("Early season (Mar-Apr)", [3, 4]),
        ("Mid season (May-Jun)", [5, 6]),
        ("Late season (Jul-Aug)", [7, 8]),
    ]:
        mask = test2["month"].isin(months)
        result = analyze_subset(test2, mask, label)
        if result:
            game_type_results.append(result)

    phase_df = pd.DataFrame([r for r in game_type_results if r])
    if len(phase_df) > 0:
        print(phase_df[["subset", "n", "model_acc", "market_acc", "blend_acc", "edge_vs_market", "blend_edge"]].to_string(index=False))

    # 5. By expected total score (proxy for weather/game style)
    print("\n--- By Expected Total Score ---")
    game_type_results = []
    if "total_score_line" in test2.columns:
        total_median = test2["total_score_line"].median()
        for label, mask in [
            ("Low total (<median)", test2["total_score_line"] < total_median),
            ("High total (>=median)", test2["total_score_line"] >= total_median),
            ("Very low (<150)", test2["total_score_line"] < 150),
            ("Very high (>175)", test2["total_score_line"] > 175),
        ]:
            result = analyze_subset(test2, mask, label)
            if result:
                game_type_results.append(result)

        total_df = pd.DataFrame([r for r in game_type_results if r])
        if len(total_df) > 0:
            print(total_df[["subset", "n", "model_acc", "market_acc", "blend_acc", "edge_vs_market", "blend_edge"]].to_string(index=False))

    # 6. By venue state
    print("\n--- By Venue State ---")
    game_type_results = []
    if "venue_state" in test2.columns:
        for state in test2["venue_state"].dropna().unique():
            mask = test2["venue_state"] == state
            result = analyze_subset(test2, mask, f"Venue: {state}")
            if result and result["n"] >= 10:
                game_type_results.append(result)

        state_df = pd.DataFrame([r for r in game_type_results if r])
        if len(state_df) > 0:
            state_df = state_df.sort_values("blend_edge", ascending=False)
            print(state_df[["subset", "n", "model_acc", "market_acc", "blend_acc", "edge_vs_market", "blend_edge"]].to_string(index=False))

    # 7. By spread movement direction
    print("\n--- By Line Movement Direction ---")
    game_type_results = []
    if "spread_move" in test2.columns:
        for label, mask in [
            ("Line moved to home", test2["spread_move"] < -1),
            ("Line stable", (test2["spread_move"] >= -1) & (test2["spread_move"] <= 1)),
            ("Line moved to away", test2["spread_move"] > 1),
        ]:
            result = analyze_subset(test2, mask, label)
            if result:
                game_type_results.append(result)

        move_df = pd.DataFrame([r for r in game_type_results if r])
        if len(move_df) > 0:
            print(move_df[["subset", "n", "model_acc", "market_acc", "blend_acc", "edge_vs_market", "blend_edge"]].to_string(index=False))

    # 8. Summary: Best subsets for model/blend
    print("\n--- SUMMARY: Best Spots for Model Edge ---")
    all_results = []

    # Collect all subset analyses
    subset_definitions = [
        ("Friday games", test2["day_of_week"] == 4),
        ("Saturday games", test2["day_of_week"] == 5),
        ("Sunday games", test2["day_of_week"] == 6),
        ("Day games", test2["is_night_game"] == 0),
        ("Night games", test2["is_night_game"] == 1),
        ("Friday night", test2["is_friday_night"] == 1),
        ("Toss-up games (45-55%)", test2["market_confidence"] < 0.05),
        ("Clear favorite (65%+)", test2["market_confidence"] >= 0.15),
        ("Early season", test2["month"].isin([3, 4])),
        ("Mid season", test2["month"].isin([5, 6])),
        ("Late season", test2["month"].isin([7, 8])),
        ("Model agrees with move", test2["model_agrees_with_move"] == True),
        ("Model opposes move", test2["model_agrees_with_move"] == False),
    ]

    if "total_score_line" in test2.columns:
        subset_definitions.extend([
            ("Low total games", test2["total_score_line"] < 155),
            ("High total games", test2["total_score_line"] > 170),
        ])

    for label, mask in subset_definitions:
        result = analyze_subset(test2, mask, label)
        if result and result["n"] >= 10:
            all_results.append(result)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.sort_values("blend_edge", ascending=False)

        print("\nTop spots where BLEND beats market:")
        top_blend = summary_df[summary_df["blend_edge"] > 0].head(5)
        if len(top_blend) > 0:
            print(top_blend[["subset", "n", "blend_acc", "market_acc", "blend_edge"]].to_string(index=False))
        else:
            print("  No subsets where blend beats market.")

        print("\nTop spots where MODEL beats market:")
        top_model = summary_df.sort_values("edge_vs_market", ascending=False)
        top_model = top_model[top_model["edge_vs_market"] > 0].head(5)
        if len(top_model) > 0:
            print(top_model[["subset", "n", "model_acc", "market_acc", "edge_vs_market"]].to_string(index=False))
        else:
            print("  No subsets where model beats market.")

        print("\nWorst spots (avoid these):")
        worst = summary_df.sort_values("blend_edge", ascending=True).head(3)
        print(worst[["subset", "n", "blend_acc", "market_acc", "blend_edge"]].to_string(index=False))

    # Save
    test2.to_csv(OUT_WITH_ODDS, index=False)
    print(f"\nSaved: {os.path.abspath(OUT_WITH_ODDS)}")

    return test2, out



# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("Loading cached AFL data...")
    team_log = load_team_rows()
    matches = build_match_level(team_log)
    # --- Sanity: confirm files are parsing into real columns ---
    print("\nSANITY: team_log columns (first 15):", list(team_log.columns)[:15])
    print("SANITY: team_log rows:", len(team_log))
    # --- Sanity: confirm odds file headers are correct ---
    odds_dbg = read_odds_file(ODDS_FILE)
    print("SANITY: odds columns (first 15):", list(odds_dbg.columns[:15]))
    print("SANITY: odds rows:", len(odds_dbg))

    # --- Sanity: odds merge hit-rate (are we matching games?) ---
    # --- Sanity: odds merge hit-rate (are we matching games?) ---
    merged_dbg = merge_odds_meta(matches, ODDS_FILE)

    print("\nSANITY: merged columns contain odds?",
          "p_home_open_nv" in merged_dbg.columns,
          "p_home_open" in merged_dbg.columns)

    # True merge hit-rate: how many games got a non-null market prob
    if "p_home_open_nv" in merged_dbg.columns:
        hit = 1 - float(merged_dbg["p_home_open_nv"].isna().mean())
    else:
        hit = 1 - float(merged_dbg["p_home_open"].isna().mean())

    print("SANITY: odds merge hit-rate:", hit)
    print("SANITY: example merged rows with odds:")
    print(
        merged_dbg[["date", "home_team", "away_team", "p_home_open_nv"]]
        .dropna()
        .head(5)
        .to_string(index=False)
    )

    # Optional debug for venue/rest/travel when enabled
    if USE_VENUE_REST_TRAVEL:
        mw_dbg = merge_odds_meta(matches, ODDS_FILE)
        mw_dbg = add_rest_features(mw_dbg)
        mw_dbg = add_travel_features(mw_dbg)

        print("Venue missing rate:", float(mw_dbg["Venue"].isna().mean()))
        print("venue_state missing rate:", float(mw_dbg["venue_state"].isna().mean()) if "venue_state" in mw_dbg.columns else np.nan)
        if "venue_state" in mw_dbg.columns and mw_dbg["venue_state"].isna().any():
            print(mw_dbg.loc[mw_dbg["venue_state"].isna(), "Venue"].value_counts().head(50))
        print("rest_diff stats:", mw_dbg["rest_diff"].describe() if "rest_diff" in mw_dbg.columns else "N/A")
        print("travel_diff value_counts:\n", mw_dbg["travel_diff"].value_counts(dropna=False).head(10) if "travel_diff" in mw_dbg.columns else "N/A")

    # Holdout eval (your existing function)
    evaluate_model_vs_market_holdout(matches, train_end_season=TRAIN_END_SEASON, test_season=TEST_SEASON)

    # -----------------------------
    # 2026 fixture predictions (walk-forward ready)
    # -----------------------------
    print("\nBuilding 2026 fixture predictions...")

    # =========================
    # 1) BUILD HISTORICAL FEATURES (for training)
    # =========================

    hist = matches.sort_values("date").reset_index(drop=True)
    # ↑ order games chronologically (important for Elo)

    hist = compute_elo_diff(hist, k=ELO_K, home_adv=ELO_HOME_ADV, base=ELO_BASE)
    # ↑ adds column: elo_diff

    hist = add_form_features(hist, team_log)
    # ↑ adds rolling form features: diff_form_IF, diff_form_CL, diff_form_DI, etc

    # Add venue home/away features
    if USE_VENUE_HOME_AWAY:
        hist = add_venue_home_away_features(hist)

    # =========================
    # 2) DEFINE TRAIN + CALIBRATION SETS
    # =========================

    train_hist = hist[hist["season"] <= TRAIN_THROUGH_SEASON].copy()
    # ↑ data to train model

    calib = hist[hist["season"] == CALIB_SEASON].copy()
    # ↑ data to fit Platt calibrator

    # =========================
    # 3) DEFINE FEATURE LIST
    # =========================

    feats = ["elo_diff"]

    resolved_stats = resolve_model_stats(team_log, MODEL_STATS)
    for s in resolved_stats:
        feats.append(f"diff_form_{s}")

    # Add venue features to feature list
    if USE_VENUE_HOME_AWAY:
        feats += ["venue_home_win_rate", "team_venue_win_rate_diff", "home_away_record_diff"]
    # ↑ final feature list used by model

    # =========================
    # 4) FIT MODEL
    # =========================

    model = fit_model(train_hist, feats)
    # ↑ logistic regression trained here

    # =========================
    # 5) FIT CALIBRATOR (Platt scaling)
    # =========================

    calib["p_home_win"] = predict(model, calib, feats)
    # ↑ raw model probability (uncalibrated)

    a, b = fit_platt_calibrator(calib["p_home_win"].values, calib["home_win"].values)
    # ↑ learn calibration params

    # =========================
    # 6) LOAD FIXTURES
    # =========================

    fx = load_fixtures_csv(FIXTURES_FILE)
    fx = fx[fx["season"] == PREDICT_SEASON].copy()
    # ↑ now fx = future matches only

    # =========================
    # 7) BUILD FIXTURE FEATURES (Elo + latest form)
    # =========================

    elos = compute_current_elos(train_hist, k=ELO_K, home_adv=ELO_HOME_ADV, base=ELO_BASE)
    # ↑ final Elo ratings after last played match

    snap = latest_form_snapshot(team_log)
    # ↑ last known rolling form values for each team

    fx_feat = add_fixture_features(fx, elos, snap)
    # ↑ creates elo_diff + diff_form_* for fixtures

    # Add venue features for fixtures
    if USE_VENUE_HOME_AWAY:
        fx_feat = add_fixture_venue_features(fx_feat, train_hist)

    # =========================
    # 8) PREDICT MODEL PROBABILITY
    # =========================

    fx_feat["p_home_win"] = predict(model, fx_feat, feats)
    # ↑ RAW model output (not calibrated yet)

    fx_feat["p_home_win_cal"] = apply_platt(fx_feat["p_home_win"].values, a, b)
    # ↑ THIS IS YOUR FINAL MODEL PROBABILITY
    #    (this answers: "where do I predict p_home_win_cal?")

    # =========================
    # 9) (OPTIONAL) COMPUTE MARKET PROBABILITY FROM FIXTURE ODDS
    # =========================

    if "home_odds" in fx_feat.columns and "away_odds" in fx_feat.columns:
        home_odds = pd.to_numeric(fx_feat["home_odds"], errors="coerce")
        away_odds = pd.to_numeric(fx_feat["away_odds"], errors="coerce")

        p_home_raw = 1.0 / home_odds
        p_away_raw = 1.0 / away_odds
        denom = p_home_raw + p_away_raw

        fx_feat["p_home_open_nv"] = np.where(denom > 0, p_home_raw / denom, np.nan)
    else:
        fx_feat["p_home_open_nv"] = np.nan

    # =========================
    # 10) BLEND MODEL + MARKET
    # =========================

    W_BLEND = 0.05  # tuned weight

    fx_feat["p_blend"] = np.where(
        fx_feat["p_home_open_nv"].notna(),
        W_BLEND * fx_feat["p_home_open_nv"] + (1 - W_BLEND) * fx_feat["p_home_win_cal"],
        fx_feat["p_home_win_cal"]
    )
    # ↑ if odds exist → blend
    # ↑ if not → fall back to model

    # =========================
    # 11) GENERATE TIP
    # =========================

    fx_feat["tip_team"] = np.where(
        fx_feat["p_blend"] >= 0.5,
        fx_feat["home_team"],
        fx_feat["away_team"]
    )

    # =========================
    # 12) OUTPUT
    # =========================

    out_cols = [
        "round", "date", "Venue", "home_team", "away_team",
        "home_odds", "away_odds",
        "p_home_open_nv",
        "p_home_win_cal",
        "p_blend",
        "tip_team"
    ]

    out_cols = [c for c in out_cols if c in fx_feat.columns]

    fx_feat[out_cols].sort_values(["round", "date"]).to_csv(OUT_FIXTURE_PREDS, index=False)

    print(f"Saved fixture preds: {os.path.abspath(OUT_FIXTURE_PREDS)}")
