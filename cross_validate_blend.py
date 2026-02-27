"""
Time-series cross-validation for blend weight tuning.
Tests if the optimal blend weight is stable across years.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Import from main model
from afl_win_model import (
    load_team_rows, build_match_level, safe_read_csv, read_odds_file,
    norm_team, fit_platt_calibrator, apply_platt, logit, inv_logit,
    compute_elo_diff, add_form_features,
    ODDS_FILE, MODEL_STATS, resolve_model_stats, FORM_WINDOWS,
    ELO_K, ELO_HOME_ADV, ELO_BASE, USE_MARGIN_ELO, ELO_MARGIN_SCALE, ELO_SEASON_REGRESS,
)

def tune_blend_weight(df, model_col, market_col, w_grid=None):
    if w_grid is None:
        w_grid = np.linspace(0.0, 1.0, 101)

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


def run_fold(matches_with_odds, train_end, test_season, calib_season, tune_window=2):
    """Run one fold of cross-validation."""

    # Split data
    train = matches_with_odds[matches_with_odds["season"] <= train_end].copy()
    test = matches_with_odds[matches_with_odds["season"] == test_season].copy()
    calib = matches_with_odds[matches_with_odds["season"] == calib_season].copy()

    # Already filtered to regular season at load time

    market_col = "p_home_open_nv"

    # Drop games without odds
    train = train.dropna(subset=[market_col]).copy()
    test = test.dropna(subset=[market_col]).copy()
    calib = calib.dropna(subset=[market_col]).copy()

    if len(test) == 0:
        return None

    # Features - baseline only for simplicity
    feature_cols = ["elo_diff", "diff_form_IF", "diff_form_CL", "diff_form_DI"]

    # Check columns exist
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return None

    X_train = train[feature_cols].values
    y_train = train["home_win"].astype(int).values
    X_test = test[feature_cols].values
    y_test = test["home_win"].astype(int).values
    X_calib = calib[feature_cols].values
    y_calib = calib["home_win"].astype(int).values

    # Train model
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, C=1.0))
    ])
    pipe.fit(X_train, y_train)

    # Raw probabilities
    p_train_raw = pipe.predict_proba(X_train)[:, 1]
    p_calib_raw = pipe.predict_proba(X_calib)[:, 1]
    p_test_raw = pipe.predict_proba(X_test)[:, 1]

    # Platt calibration
    a, b = fit_platt_calibrator(p_calib_raw, y_calib)
    p_test_cal = apply_platt(p_test_raw, a, b)

    # Add to test dataframe
    test = test.copy()
    test["p_model"] = p_test_cal

    # Tune blend weight on train data (use last N years of train for tuning)
    tune_start = train_end - tune_window + 1  # e.g., if train_end=2024, tune_window=3, tune on 2022-2024
    tune_df = train[train["season"] >= tune_start].copy()

    # Get calibrated probs for tune set
    X_tune = tune_df[feature_cols].values
    p_tune_raw = pipe.predict_proba(X_tune)[:, 1]
    p_tune_cal = apply_platt(p_tune_raw, a, b)
    tune_df["p_model"] = p_tune_cal

    # Tune blend weight
    w_grid = np.linspace(0.0, 1.0, 101)
    best = tune_blend_weight(tune_df, "p_model", market_col, w_grid)

    # Apply best weight to test set
    w = best["w"]
    test["p_blend"] = w * test[market_col] + (1 - w) * test["p_model"]

    # Evaluate on test
    market_tips = (test[market_col] >= 0.5).astype(int)
    model_tips = (test["p_model"] >= 0.5).astype(int)
    blend_tips = (test["p_blend"] >= 0.5).astype(int)

    market_correct = int((market_tips == y_test).sum())
    model_correct = int((model_tips == y_test).sum())
    blend_correct = int((blend_tips == y_test).sum())

    result = {
        "test_season": test_season,
        "train_end": train_end,
        "calib_season": calib_season,
        "tune_seasons": f"{tune_start}-{train_end}",
        "n_test": len(test),
        "best_w": best["w"],
        "tune_acc": best["acc"],
        "market_correct": market_correct,
        "model_correct": model_correct,
        "blend_correct": blend_correct,
        "market_acc": market_correct / len(test),
        "model_acc": model_correct / len(test),
        "blend_acc": blend_correct / len(test),
        "blend_vs_market": blend_correct - market_correct,
        "model_vs_market": model_correct - market_correct,
    }

    # Also evaluate fixed blend weights
    fixed_weights = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    for fw in fixed_weights:
        p_fixed = fw * test[market_col] + (1 - fw) * test["p_model"]
        fixed_tips = (p_fixed >= 0.5).astype(int)
        fixed_correct = int((fixed_tips == y_test).sum())
        result[f"w{int(fw*100):02d}_correct"] = fixed_correct

    return result


def main():
    print("Loading data...")
    team_log = load_team_rows()
    matches = build_match_level(team_log)

    # Add Elo and form features
    matches = compute_elo_diff(matches)
    matches = add_form_features(matches, team_log)

    # Load and merge odds
    odds = read_odds_file(ODDS_FILE)
    odds["Home Team"] = odds["Home Team"].apply(norm_team)
    odds["Away Team"] = odds["Away Team"].apply(norm_team)
    odds["Date"] = pd.to_datetime(odds["Date"], dayfirst=True)

    # Calculate no-vig probabilities
    h = 1 / odds["Home Odds"].astype(float)
    a = 1 / odds["Away Odds"].astype(float)
    odds["p_home_open_nv"] = h / (h + a)

    # Merge
    matches["date"] = pd.to_datetime(matches["date"])
    merged = matches.merge(
        odds[["Date", "Home Team", "Away Team", "p_home_open_nv", "Play Off Game?"]],
        left_on=["date", "home_team", "away_team"],
        right_on=["Date", "Home Team", "Away Team"],
        how="left"
    )

    # Filter to regular season (non-finals)
    merged["Play Off Game?"] = merged["Play Off Game?"].fillna("N")
    merged = merged[merged["Play Off Game?"] != "Y"].copy()

    print(f"Merged {len(merged)} matches, {merged['p_home_open_nv'].notna().sum()} with odds")

    # Time-series CV folds (3-year tuning windows)
    folds = [
        {"train_end": 2022, "test_season": 2023, "calib_season": 2022, "tune_window": 2},
        {"train_end": 2023, "test_season": 2024, "calib_season": 2023, "tune_window": 3},
        {"train_end": 2024, "test_season": 2025, "calib_season": 2024, "tune_window": 3},
    ]

    results = []
    for fold in folds:
        print(f"\nRunning fold: train<={fold['train_end']}, test={fold['test_season']}...")
        res = run_fold(merged, **fold)
        if res:
            results.append(res)
            print(f"  Best w: {res['best_w']:.2f}")
            print(f"  Test: Market={res['market_correct']}, Model={res['model_correct']}, Blend={res['blend_correct']}")
            print(f"  Blend vs Market: {res['blend_vs_market']:+d}")

    # Summary
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)

    df = pd.DataFrame(results)
    print(df[["test_season", "tune_seasons", "best_w", "n_test",
              "market_correct", "model_correct", "blend_correct",
              "blend_vs_market"]].to_string(index=False))

    print(f"\nOptimal blend weights across folds: {df['best_w'].tolist()}")
    print(f"Mean optimal w: {df['best_w'].mean():.2f}")
    print(f"Std of optimal w: {df['best_w'].std():.2f}")

    total_blend = df["blend_correct"].sum()
    total_market = df["market_correct"].sum()
    total_model = df["model_correct"].sum()
    total_n = df["n_test"].sum()

    print(f"\nAggregated across all test seasons:")
    print(f"  Market: {total_market}/{total_n} ({100*total_market/total_n:.1f}%)")
    print(f"  Model:  {total_model}/{total_n} ({100*total_model/total_n:.1f}%)")
    print(f"  Blend:  {total_blend}/{total_n} ({100*total_blend/total_n:.1f}%)")
    print(f"  Blend vs Market: {total_blend - total_market:+d}")

    # Test fixed blend weights
    print("\n" + "="*80)
    print("FIXED BLEND WEIGHT COMPARISON (across all test years)")
    print("="*80)

    fixed_weights = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    fixed_results = []

    for w in fixed_weights:
        total_correct = 0
        by_year = {}
        for res in results:
            key = f"w{int(w*100):02d}_correct"
            correct = res.get(key, 0)
            total_correct += correct
            by_year[res["test_season"]] = correct

        fixed_results.append({
            "w": w,
            "2023": by_year.get(2023, 0),
            "2024": by_year.get(2024, 0),
            "2025": by_year.get(2025, 0),
            "total": total_correct,
            "acc": total_correct / total_n if total_n > 0 else 0,
            "vs_market": total_correct - total_market,
        })

    fixed_df = pd.DataFrame(fixed_results)
    print(fixed_df.to_string(index=False))

    # Find best fixed weight
    best_fixed = fixed_df.loc[fixed_df["total"].idxmax()]
    print(f"\nBest fixed weight: w={best_fixed['w']:.2f} with {int(best_fixed['total'])} correct ({best_fixed['vs_market']:+.0f} vs market)")

    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*80)
    print(f"  Pure Market (w=1.0):     {total_market} correct")
    print(f"  Pure Model (w=0.0):      {total_model} correct ({total_model - total_market:+d} vs market)")
    print(f"  Tuned blend (variable):  {total_blend} correct ({total_blend - total_market:+d} vs market)")
    print(f"  Best fixed (w={best_fixed['w']:.2f}):      {int(best_fixed['total'])} correct ({best_fixed['vs_market']:+.0f} vs market)")


if __name__ == "__main__":
    main()
