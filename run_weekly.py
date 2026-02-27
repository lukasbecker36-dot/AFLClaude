# run_weekly.py
import os
import argparse
import numpy as np
import pandas as pd

from src.afl_pipeline import (
    load_team_rows,
    load_fixtures_csv,
    build_historical_feature_frame,
    make_feature_list,
    train_and_calibrate,
    predict_proba,
    apply_platt,
    compute_current_elos,
    latest_form_snapshot,
    add_fixture_features,
    merge_odds_meta,
    evaluate_tipping,
    metrics_block,
)

# ====== CONFIG DEFAULTS ======
TEAM_CACHE = "afl_team_rows.csv"
FIXTURES_FILE = "fixtures_2026.csv"
ODDS_FILE = "aflodds.csv"  # optional, can be missing for early-week runs

MODEL_STATS = ["I50", "CL", "DI"]
FORM_WINDOWS = [5]

ELO_K = 25
ELO_HOME_ADV = 65
ELO_BASE = 1500

# production blend
DEFAULT_BLEND_W = 0.84  # your tuned-ish weight; override via CLI if you want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team_cache", default=TEAM_CACHE)
    ap.add_argument("--fixtures", default=FIXTURES_FILE)
    ap.add_argument("--odds", default=ODDS_FILE)

    ap.add_argument("--predict_season", type=int, default=2026)
    ap.add_argument("--predict_round", type=int, default=None, help="If set, output only this round")

    ap.add_argument("--train_through_season", type=int, default=2025)
    ap.add_argument("--calib_season", type=int, default=2025)

    ap.add_argument("--blend_w", type=float, default=DEFAULT_BLEND_W)
    ap.add_argument("--out", default="outputs/afl_fixture_preds.csv")

    ap.add_argument("--do_holdout_eval", action="store_true",
                    help="Optional: also run a simple market/model/blend eval on last season if odds exist")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 1) Load team rows (updated weekly by scraper)
    team_log = load_team_rows(args.team_cache)

    # 2) Build full historical match feature frame
    hist = build_historical_feature_frame(
        team_log=team_log,
        model_stats=MODEL_STATS,
        form_windows=FORM_WINDOWS,
        elo_k=ELO_K,
        elo_home_adv=ELO_HOME_ADV,
        elo_base=ELO_BASE,
    )

    feats = make_feature_list(team_log, MODEL_STATS, FORM_WINDOWS)

    # 3) Train + Platt
    model, (a, b) = train_and_calibrate(
        hist=hist,
        feats=feats,
        train_through_season=args.train_through_season,
        calib_season=args.calib_season,
    )

    # 4) Load fixtures + filter
    fx = load_fixtures_csv(args.fixtures)
    fx = fx[fx["season"] == args.predict_season].copy()
    if args.predict_round is not None:
        fx = fx[fx["round"].astype(int) == int(args.predict_round)].copy()

    # 5) Current Elo + latest form snapshot as-of latest played game in training set
    train_hist = hist[hist["season"] <= args.train_through_season].copy()
    elos = compute_current_elos(train_hist, k=ELO_K, home_adv=ELO_HOME_ADV, base=ELO_BASE)
    snap = latest_form_snapshot(team_log, model_stats=MODEL_STATS, form_windows=FORM_WINDOWS)

    fx_feat = add_fixture_features(
        fixtures=fx,
        elos=elos,
        form_snap=snap,
        team_log=team_log,
        model_stats=MODEL_STATS,
        form_windows=FORM_WINDOWS,
        elo_base=ELO_BASE,
    )

    # 6) Predict raw + calibrated
    fx_feat["p_home_raw"] = predict_proba(model, fx_feat, feats)
    fx_feat["p_home_cal"] = apply_platt(fx_feat["p_home_raw"].values, a, b)

    # 7) Optional: merge odds for fixtures and compute blend if odds exist for those games
    fx_feat["p_market_open_nv"] = np.nan
    if args.odds and os.path.exists(args.odds):
        # re-use your match-style merger by creating a minimal "matches-like" df
        # containing date/home_team/away_team. merge_odds_meta will attach p_home_open_nv.
        tmp = fx_feat[["date", "home_team", "away_team"]].copy()
        tmp["match_url"] = ""      # dummy
        tmp["season"] = args.predict_season
        tmp["home_score"] = np.nan
        tmp["away_score"] = np.nan
        tmp["home_win"] = 0        # dummy

        tmp2 = merge_odds_meta(tmp, args.odds)
        if "p_home_open_nv" in tmp2.columns:
            fx_feat["p_market_open_nv"] = tmp2["p_home_open_nv"].values

    # blend: only where market exists; else fallback to model cal
    w = float(args.blend_w)
    fx_feat["p_blend"] = np.where(
        fx_feat["p_market_open_nv"].notna(),
        w * fx_feat["p_market_open_nv"].astype(float) + (1 - w) * fx_feat["p_home_cal"].astype(float),
        fx_feat["p_home_cal"].astype(float)
    )

    # 8) Tips
    fx_feat["tip_team"] = np.where(fx_feat["p_blend"] >= 0.5, fx_feat["home_team"], fx_feat["away_team"])
    fx_feat["tip_conf"] = (fx_feat["p_blend"] - 0.5).abs() * 2  # 0..1

    out_cols = [
        "round", "date", "Venue", "home_team", "away_team",
        "p_home_cal", "p_market_open_nv", "p_blend",
        "tip_team", "tip_conf",
    ]
    fx_out = fx_feat[out_cols].sort_values(["round", "date"]).reset_index(drop=True)
    fx_out.to_csv(args.out, index=False)
    print(f"Saved: {os.path.abspath(args.out)}")

    # 9) Optional: quick holdout eval if odds exist (kept minimal, not your huge diagnostic block)
    if args.do_holdout_eval and args.odds and os.path.exists(args.odds):
        # Example: evaluate last season regular season only, using existing odds flags if present
        from src.afl_pipeline import merge_odds_meta

        mw = merge_odds_meta(hist, args.odds)
        mw["Play Off Game?"] = mw.get("Play Off Game?", "N")
        reg = mw[mw["Play Off Game?"].fillna("N") != "Y"].copy()
        reg = reg[reg["season"] == args.calib_season].copy()

        if len(reg) > 0 and "p_home_open_nv" in reg.columns:
            reg["p_raw"] = predict_proba(model, reg, feats)
            reg["p_cal"] = apply_platt(reg["p_raw"].values, a, b)
            reg["p_blend"] = w * reg["p_home_open_nv"].astype(float) + (1 - w) * reg["p_cal"].astype(float)

            print("\nHoldout (last season) tipping:")
            print(evaluate_tipping(reg.dropna(subset=["p_home_open_nv"]), "p_home_open_nv", "Market open nv"))
            print(evaluate_tipping(reg, "p_cal", "Model cal"))
            print(evaluate_tipping(reg.dropna(subset=["p_home_open_nv"]), "p_blend", f"Blend w={w:.2f}"))

            print("\nHoldout (last season) prob metrics:")
            rows = [
                metrics_block(reg.dropna(subset=["p_home_open_nv"]), "p_home_open_nv", "Market open nv"),
                metrics_block(reg, "p_cal", "Model cal"),
                metrics_block(reg.dropna(subset=["p_home_open_nv"]), "p_blend", f"Blend w={w:.2f}"),
            ]
            print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
