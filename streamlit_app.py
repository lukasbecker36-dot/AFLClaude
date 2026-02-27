import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from afl_pipeline import (
    load_team_rows,
    load_fixtures_csv,
    build_historical_feature_frame,
    make_feature_list,
    train_and_calibrate,
    compute_current_elos,
    latest_form_snapshot,
    add_fixture_features,
    predict_proba,
    apply_platt,
)

# ── Constants ─────────────────────────────────────────────────────────────────
TEAM_CACHE      = os.path.join(PROJECT_ROOT, "afl_team_rows.csv")
FIXTURES_FILE   = os.path.join(PROJECT_ROOT, "fixtures_2026.csv")
MODEL_STATS     = ["I50", "CL", "DI"]
FORM_WINDOWS    = [5]
ELO_K           = 25
ELO_HOME_ADV    = 65
ELO_BASE        = 1500
TRAIN_THROUGH   = 2025
CALIB_SEASON    = 2025
PREDICT_SEASON  = 2026
DEFAULT_BLEND_W = 0.84


# ── No-vig helper ─────────────────────────────────────────────────────────────
def compute_novig(home_odds, away_odds):
    """Returns (p_home_nv, p_away_nv) or (None, None) if invalid."""
    try:
        h, a = float(home_odds), float(away_odds)
        if h <= 1.0 or a <= 1.0:
            return None, None
        ph, pa = 1.0 / h, 1.0 / a
        d = ph + pa
        return ph / d, pa / d
    except (TypeError, ValueError, ZeroDivisionError):
        return None, None


# ── Cached pipeline layers ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data():
    team_log = load_team_rows(TEAM_CACHE)
    fixtures = load_fixtures_csv(FIXTURES_FILE)
    return team_log, fixtures


@st.cache_data(show_spinner="Training model (first load only)...")
def build_model(team_log):
    hist = build_historical_feature_frame(
        team_log, MODEL_STATS, FORM_WINDOWS, ELO_K, ELO_HOME_ADV, ELO_BASE
    )
    feats = make_feature_list(team_log, MODEL_STATS, FORM_WINDOWS)
    model, (a, b) = train_and_calibrate(hist, feats, TRAIN_THROUGH, CALIB_SEASON)
    train_hist = hist[hist["season"] <= TRAIN_THROUGH].copy()
    return model, a, b, feats, train_hist


@st.cache_data(show_spinner="Computing team ratings...")
def build_elo_and_form(train_hist, team_log):
    elos = compute_current_elos(train_hist, k=ELO_K, home_adv=ELO_HOME_ADV, base=ELO_BASE)
    snap = latest_form_snapshot(team_log, model_stats=MODEL_STATS, form_windows=FORM_WINDOWS)
    return elos, snap


# ── Prediction (uncached — runs on each button click) ─────────────────────────
def run_prediction(round_fx, edited_odds, model, platt_a, platt_b, feats, elos, snap, team_log):
    fx = add_fixture_features(
        fixtures=round_fx,
        elos=elos,
        form_snap=snap,
        team_log=team_log,
        model_stats=MODEL_STATS,
        form_windows=FORM_WINDOWS,
        elo_base=ELO_BASE,
    )
    # Realign index after merge in add_fixture_features
    fx = fx.reset_index(drop=True)

    fx["p_home_cal"] = apply_platt(predict_proba(model, fx, feats), platt_a, platt_b)
    fx["p_market_open_nv"] = np.nan

    warnings = []
    for i, row in edited_odds.iterrows():
        h_val = row.get("Home Odds")
        a_val = row.get("Away Odds")
        h_ok = pd.notna(h_val) and str(h_val) != ""
        a_ok = pd.notna(a_val) and str(a_val) != ""

        if h_ok and a_ok:
            p_nv, _ = compute_novig(h_val, a_val)
            if p_nv is not None:
                fx.loc[i, "p_market_open_nv"] = p_nv
            else:
                match_label = f"{round_fx.at[i, 'home_team']} vs {round_fx.at[i, 'away_team']}"
                warnings.append(f"{match_label}: odds must be > 1.0 — using model only.")
        elif h_ok or a_ok:
            match_label = f"{round_fx.at[i, 'home_team']} vs {round_fx.at[i, 'away_team']}"
            warnings.append(f"{match_label}: only one side entered — using model only.")

    fx["p_blend"] = np.where(
        fx["p_market_open_nv"].notna(),
        DEFAULT_BLEND_W * fx["p_market_open_nv"] + (1 - DEFAULT_BLEND_W) * fx["p_home_cal"],
        fx["p_home_cal"],
    )
    fx["tip_team"] = np.where(fx["p_blend"] >= 0.5, fx["home_team"], fx["away_team"])
    fx["tip_conf"] = (fx["p_blend"] - 0.5).abs() * 2

    return fx, warnings


# ── App layout ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AFL 2026 Tips", layout="wide")
st.title("AFL 2026 — Weekly Tips")

# Load + train (cached after first run)
try:
    team_log, all_fixtures = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

model, platt_a, platt_b, feats, train_hist = build_model(team_log)
elos, snap = build_elo_and_form(train_hist, team_log)

# Filter to predict season
fx_2026 = all_fixtures[all_fixtures["season"] == PREDICT_SEASON].copy()
available_rounds = sorted(fx_2026["round"].unique().tolist())

if not available_rounds:
    st.error("No 2026 fixtures found in fixtures_2026.csv.")
    st.stop()

# Auto-detect next upcoming round
today = pd.Timestamp(date.today())

def _default_round(fx, rounds):
    for r in rounds:
        if fx.loc[fx["round"] == r, "date"].min() >= today:
            return r
    return rounds[-1]

default_idx = available_rounds.index(_default_round(fx_2026, available_rounds))

selected_round = st.selectbox(
    "Select Round",
    options=available_rounds,
    index=default_idx,
    format_func=lambda r: f"Round {r}",
)

# Fixtures for the selected round (reset index for alignment with data_editor)
round_fx = (
    fx_2026[fx_2026["round"] == selected_round]
    .copy()
    .reset_index(drop=True)
)

# Build the odds entry table
odds_df = pd.DataFrame({
    "Match":      round_fx["home_team"] + " vs " + round_fx["away_team"],
    "Date":       round_fx["date"].dt.strftime("%a %d %b"),
    "Venue":      round_fx["Venue"],
    "Home Odds":  [np.nan] * len(round_fx),
    "Away Odds":  [np.nan] * len(round_fx),
})

st.subheader(f"Round {selected_round} — Enter Decimal Odds")
st.caption("Leave blank to use model-only predictions. Both sides required per match. Odds must be > 1.0 (e.g. 1.85).")

edited_odds = st.data_editor(
    odds_df,
    column_config={
        "Match":     st.column_config.TextColumn("Match", disabled=True),
        "Date":      st.column_config.TextColumn("Date", disabled=True),
        "Venue":     st.column_config.TextColumn("Venue", disabled=True),
        "Home Odds": st.column_config.NumberColumn(
            "Home Odds", min_value=1.01, step=0.05, format="%.2f"
        ),
        "Away Odds": st.column_config.NumberColumn(
            "Away Odds", min_value=1.01, step=0.05, format="%.2f"
        ),
    },
    hide_index=True,
    use_container_width=True,
    key=f"odds_editor_r{selected_round}",
)

if st.button("Generate Tips", type="primary"):
    with st.spinner("Running predictions..."):
        fx_result, warnings = run_prediction(
            round_fx, edited_odds, model, platt_a, platt_b, feats, elos, snap, team_log
        )

    for w in warnings:
        st.warning(w)

    # Summary metrics
    n_games       = len(fx_result)
    n_with_market = int(fx_result["p_market_open_nv"].notna().sum())
    n_model_only  = n_games - n_with_market

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Games", n_games)
    col2.metric("With Market Odds", n_with_market)
    col3.metric("Model Only", n_model_only)

    # Build display table
    def _pct(val):
        return f"{val:.1%}" if pd.notna(val) else "—"

    rows = []
    for idx in fx_result.index:
        row = fx_result.loc[idx]
        h_odds = edited_odds.at[idx, "Home Odds"]
        a_odds = edited_odds.at[idx, "Away Odds"]
        rows.append({
            "Match":           f"{row['home_team']} vs {row['away_team']}",
            "Date":            row["date"].strftime("%a %d %b") if pd.notna(row["date"]) else "",
            "Venue":           row.get("Venue", ""),
            "Model% (Home)":   _pct(row["p_home_cal"]),
            "Market% (Home)":  _pct(row["p_market_open_nv"]),
            "Blend% (Home)":   _pct(row["p_blend"]),
            "Tip":             row["tip_team"],
            "Confidence":      f"{row['tip_conf']:.0%}",
            "Home Odds":       f"{h_odds:.2f}" if pd.notna(h_odds) else "—",
            "Away Odds":       f"{a_odds:.2f}" if pd.notna(a_odds) else "—",
        })

    display_df = pd.DataFrame(rows)

    st.subheader(f"Round {selected_round} Tips")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # Download CSV
    out_cols = [
        "round", "date", "Venue", "home_team", "away_team",
        "p_home_cal", "p_market_open_nv", "p_blend", "tip_team", "tip_conf",
    ]
    csv_df = fx_result[[c for c in out_cols if c in fx_result.columns]].copy()
    st.download_button(
        label="Download CSV",
        data=csv_df.to_csv(index=False),
        file_name=f"afl_tips_2026_r{selected_round}.csv",
        mime="text/csv",
    )
