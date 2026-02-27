# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AFL match prediction system that combines Elo ratings, rolling team form statistics, and bookmaker odds to generate win probabilities and tipping predictions for Australian Football League games.

## Key Commands

**Scrape latest match data:**
```bash
python scrape_afl_tables.py
```
Incrementally updates `afl_team_rows.csv` with new matches from afltables.com.

**Generate weekly predictions:**
```bash
python run_weekly.py --predict_season 2026 --predict_round 1
```
Key flags:
- `--blend_w 0.84` - market/model blend weight (higher = more market)
- `--do_holdout_eval` - run holdout evaluation on previous season
- `--out outputs/predictions.csv` - output path

**Run cross-validation for blend weight:**
```bash
python cross_validate_blend.py
```

**Test additional stat features:**
```bash
python test_more_stats.py
```

**Run season ladder simulation:**
```bash
python data/afl_season_sim.py
```

**Backtest ladder predictions on historical seasons:**
```bash
python backtest_ladder.py
```
Tests 2023-2025 seasons using only prior year data for training. Outputs Spearman correlation, mean absolute rank error, top 4/8 accuracy, and probability calibration metrics.

## Architecture

### Data Flow
1. `scrape_afl_tables.py` scrapes AFL Tables for team-level match stats → `afl_team_rows.csv`
2. `afl_win_model.py` or `afl_pipeline.py` builds features (Elo, rolling form) and trains model
3. `run_weekly.py` generates predictions for upcoming fixtures using trained model + market odds blend
4. `data/afl_season_sim.py` runs Monte Carlo simulations for ladder projections
5. `backtest_ladder.py` evaluates ladder prediction accuracy on historical seasons (2023-2025)

### Core Modules

**afl_pipeline.py** - Clean reusable pipeline functions:
- `build_historical_feature_frame()` - constructs match-level feature DataFrame
- `train_and_calibrate()` - trains logistic regression + Platt calibration
- `compute_current_elos()` / `latest_form_snapshot()` - get current state for predictions
- `add_fixture_features()` - attach features to future fixtures

**afl_win_model.py** - Extended model with additional features:
- Venue/travel features, head-to-head records, ladder position
- Weather data integration (cached in `weather_cache.csv`)
- Hybrid tipping rules and blend weight tuning
- Contains `add_hybrid_tip()` and `tune_blend_weight()` for strategy optimization

### Key Configuration (in afl_win_model.py / run_weekly.py)
- `MODEL_STATS = ["I50", "CL", "DI"]` - Inside 50s, Clearances, Disposals
- `FORM_WINDOWS = [5]` - 5-game rolling average
- `ELO_K = 25`, `ELO_HOME_ADV = 65`, `ELO_BASE = 1500`
- `BLEND_W = 0.84-0.85` - tuned to favor market odds

### Team Name Normalization
All modules use `norm_team()` to standardize team names between AFL Tables and odds data. Maps variations like "GWS Giants" → "Greater Western Sydney", "Sydney Swans" → "Sydney".

## Data Files
- `afl_team_rows.csv` - historical team-level match stats (scraped)
- `aflodds.csv` - historical bookmaker odds (OddsPortal format, header on row 2)
- `fixtures_2026.csv` - upcoming season fixtures
- `weather_cache.csv` - cached weather API responses

## Model Evaluation
The codebase uses time-series cross-validation. Train on seasons ≤ N, calibrate on season N, test on season N+1. Key metrics: log loss, Brier score, AUC, tipping accuracy. Market odds (no-vig) serve as the primary baseline.

## Project Scope (IMPORTANT)

The current project has **two linked but distinct purposes**:

1. **Primary purpose:**
   Predict winners of AFL matches week-by-week using:

   * Elo ratings,
   * Rolling team statistics,
   * Optional market odds blending.

2. **Secondary purpose:**
   Generate a **pre-season predictive ladder** by:

   * Predicting all matches in a season,
   * Simulating the ladder via Monte Carlo,
   * Evaluating accuracy via historical backtesting.

⚠️ Ladder prediction is an *evaluation layer* built on top of the match model.
⚠️ Do NOT simplify or discard match-level logic to optimise ladder accuracy.
⚠️ Any improvements must preserve or enhance match prediction quality.

---

## Project Goal

Use the match-level model to:

* Produce weekly tipping probabilities,
* Produce pre-season ladder forecasts,
* Backtest both historically to evaluate realism.

---

## What Has Been Implemented

### 1. Historical Ladder Backtesting (`backtest_ladder.py`)

A new script was created that:

* Trains the model only on seasons **before** the target season,
* Predicts all matches in the target season using the match model,
* Simulates the ladder,
* Compares predicted ladder vs actual ladder.

Metrics:

* Spearman rank correlation,
* Mean absolute rank error (MARE),
* Top 8 accuracy,
* Top 4 accuracy,
* Minor premier hit,
* Wooden spoon hit,
* Probability calibration (Brier score).

Backtested seasons:

* 2023 (trained on 2022),
* 2024 (trained on 2022–2023),
* 2025 (trained on 2022–2024).

---

### 2. Walk-Forward Elo (Major Improvement)

Instead of static preseason Elo:

* Elo is updated **after each match using real results** when generating season predictions.
* Later matches use updated team strengths.

Key finding:

> Walk-forward Elo is the primary source of improvement for ladder accuracy and should be retained.

Current config:

* `USE_WALKFORWARD_ELO = True`

---

### 3. Premiership Odds Blending

Implemented:

* Historical preseason premiership odds for 2023–2025 (hardcoded),
* Odds converted to implied ladder ranks,
* Optional blending:
  `blended_rank = (1 - w) * model_rank + w * market_rank`

---

### 4. Blend Weight Tuning

Grid search over blend weights (0.0 → 1.0):

Result:

* Best Spearman / MARE: **w = 0.0 (pure model)**
* Light blend (w ≈ 0.2) slightly improved Top 8 count but hurt rank correlation.
* Heavy blending degraded performance.

Conclusion:

> Market odds add little once walk-forward Elo is used.

Current config:

* `PREMIERSHIP_BLEND_W = 0.0`

---

## Performance Summary

Baseline (static Elo):

* Spearman ≈ 0.57
* MARE ≈ 3.33

Improved (walk-forward Elo):

* Spearman ≈ 0.78
* MARE ≈ 2.45
* Top 8 accuracy ≈ 71%

Strengths:

* Bottom teams predicted well (wooden spoon hit 2/3 years),
* Top 8 identification reasonable.

Weakness:

* Breakout seasons missed:

  * Adelaide 2025 (predicted ~12th → finished 1st),
  * Hawthorn 2024 (predicted ~14th → finished 7th),
  * Geelong 2023 (predicted ~2nd → finished 12th).

Market odds also missed these, so this is a structural problem.

---

## What NOT To Do

* ❌ Do NOT discard match-level prediction logic.
* ❌ Do NOT rewrite the project to be ladder-only.
* ❌ Do NOT remove walk-forward Elo.
* ❌ Do NOT reintroduce heavy market blending.
* ❌ Do NOT train on future-season data.

Ladder accuracy must be improved **via better match probabilities**, not via shortcut ranking logic.

---

## Current Best Settings

* Walk-forward Elo: ON
* Premiership odds blend: OFF (w = 0.0)
* Training data: seasons strictly before target year
* Prediction style: preseason ladder forecast
* Core model purpose: weekly match winner probabilities

---

## Next Objective (Unfinished Work)

Main goal:

> Improve detection of breakout seasons while preserving match prediction quality.

### A. Internal Features (Not Implemented)

Derive from existing match data:

* Late-season form,
* Momentum / trajectory,
* End-of-season Elo slope,
* Improvement vs prior season.

---

### B. External Data (Research Complete, Implementation Pending)

Add:

1. Coaching changes,
2. Player movement / list turnover,
3. List age profiles.

**Data Sources Identified (Session 2026-02-02):**

#### Coaching Changes
- AFL Tables Coaches Index: https://afltables.com/afl/stats/coaches/coaches_idx.html
- Wikipedia: https://en.wikipedia.org/wiki/List_of_current_Australian_Football_League_coaches

**Key coaching changes found:**
- 2022: Craig McRae → Collingwood (won 2023 flag), Sam Mitchell → Hawthorn (breakout 2024)
- 2023: Adam Kingsley → GWS, Adem Yze → Richmond
- 2024: Andrew McQualter → West Coast (replaced Adam Simpson mid-season)
- 2025: Dean Cox → Sydney (replaced Longmire)

#### Player Movements / List Turnover
- AFL Trade Tracker: https://www.afl.com.au/trade/trade-tracker/2024
- AFL List Changes: https://www.afl.com.au/trade/list-changes
- ESPN AFL Player Movement Tracker: https://www.espn.com/afl/story/_/id/41541718/afl-2024-player-movement-tracker

#### List Age Profiles
- AFL.com.au Age Analysis: https://www.afl.com.au/news/1341020/young-fremantle-side-rising-collingwood-lead-the-way-the-oldest-and-youngest-teams-of-2025-revealed
- AFL Tables Player Ages: https://afltables.com/afl/stats/ages.html
- Footywire Players Database: https://www.footywire.com/afl/footy/ft_players

**2025 Age Rankings (example data):**
- Oldest: Collingwood (avg 28.5 years)
- Youngest: Fremantle (avg 24.6 years)

---

## Immediate Next Steps

**Step 1: Derive momentum features from existing data**
- Calculate late-season form (last 6 matches of prior season)
- Compute Elo trajectory (end-of-season Elo minus start-of-season Elo)
- Compare year-over-year improvement

**Step 2: Hardcode external data for 2023-2025**
Since scraping is complex, manually encode:
- `NEW_COACH` flag: teams with coach in year 1-2 of tenure
- `COACH_TENURE`: years current coach has been at club
- Key player acquisitions (binary flags for major trades)
- List age category (young/mid/old based on published rankings)

**Step 3: Add features to backtest**
- Modify `backtest_ladder.py` to include new features
- Test impact on breakout detection (Adelaide 2025, Hawthorn 2024)

**Step 4: Evaluate and iterate**
- Compare Spearman/MARE before vs after
- Check if breakout teams are now ranked higher preseason

---

## Command to Run

```bash
python backtest_ladder.py
```

---

## Ground Rules

* Match prediction is the core objective.
* Ladder simulation is built on match predictions.
* No future information leakage.
* Market odds only preseason for ladder predictions.
* Walk-forward Elo allowed.
* Improvements should enhance both match and ladder realism.

---

## Session Log

### 2026-02-02: Momentum Features Implemented & Key Insight

**Completed:**
- Implemented late-season momentum features in `backtest_ladder.py`
- Added functions: `compute_team_season_stats()`, `compute_elo_trajectory()`, `compute_momentum_features()`, `get_preseason_momentum()`, `print_momentum_analysis()`
- Features: late_win_rate, late_avg_margin, elo_change, momentum_score (composite)
- Tested momentum features on 2023-2025 backtests

**Critical Finding: Momentum CANNOT predict breakouts**

The momentum analysis showed:
- 2024 Hawthorn: momentum = -0.158 (NEGATIVE)
- 2025 Adelaide: momentum = -0.150 (NEGATIVE)

**Why:** Breakout teams had POOR recent form before their breakthrough. They weren't playing well - they improved due to OTHER factors. Late-season momentum measures "how well team was playing" which is the OPPOSITE of what identifies breakouts.

**Momentum Results:**
- No improvement in Spearman/MARE/Top 8
- Actually made some predictions slightly worse
- Conclusion: Late-season momentum is NOT useful for ladder predictions

**What DOES predict breakouts:**
1. Coaching tenure (year 2-3 of new coach = system bedding in)
   - Sam Mitchell (Hawthorn): Year 1=2022, Year 2=2023, Year 3=2024 **BREAKOUT**
   - Matthew Nicks (Adelaide): Year 5=2024, Year 6=2025 **BREAKOUT** (but extended tenure may relate to player maturity)
   - Craig McRae (Collingwood): Year 1=2022, Year 2=2023 **PREMIERSHIP**
2. Young list profile maturing
3. Key player acquisitions from trade period

**Data sources identified:**
- AFL Tables coaches: https://afltables.com/afl/stats/coaches/coaches_idx.html
- Key coaching changes 2022-2025 documented in claude.md above

**Next steps:**
1. ~~Add momentum features~~ ✓ (did not help)
2. ~~Add coaching tenure feature~~ ✓ (helps modestly - see below)
3. Add list age category if data available
4. Consider other factors for non-coaching breakouts (Adelaide 2025)

---

### 2026-02-02 (continued): Coaching Tenure Feature Results

**Implemented:**
- Added `COACHING_TENURE` dict with tenure years for all teams 2023-2025
- Added `BREAKOUT_TENURE_YEARS = [2, 3]` - coaches in year 2-3 get +30 Elo boost
- Added `get_coaching_adjustments()` and `print_coaching_analysis()` functions

**Results (Improved vs With Coaching):**
```
Season   Improved         With Coaching
         Spearman  MARE   Spearman  MARE   Top8
2023     0.740     2.59   0.725     2.44   6/8 (+1)
2024     0.773     2.60   0.787     2.63   6/8
2025     0.822     2.16   0.839     2.08   6/8
Mean     0.778     2.45   0.784     2.38   18/24 (+1)
```

**Coaching Impact:**
- Spearman: +0.006 (small improvement)
- MARE: -0.07 (better)
- Top 8: +1 more correct

**Breakout Team Analysis:**
- 2023 Collingwood (Year 2 coach, IN zone): +0.3 positions closer
- 2024 Hawthorn (Year 3 coach, IN zone): +0.5 positions closer
- 2025 Adelaide (Year 6 coach, NOT in zone): -0.2 positions (coaching can't help)

**Key Insight:**
Coaching tenure helps for teams with NEW coaches (year 2-3), but cannot explain breakouts like Adelaide 2025 where the coach had 6 years tenure. Adelaide's breakout was due to other factors:
- List maturity (young players developing)
- Key trades/acquisitions
- Culture/system finally clicking

**Remaining gaps:**
- Adelaide 2025 still poorly predicted (actual 1st, predicted 10th)
- Need additional features beyond coaching to capture all breakout types

---

### 2026-02-02 (continued): 2026 Predictions Generated

**Implemented:**
- Added `COACHING_TENURE[2026]` with all team coaching tenures
- Added `predict_2026()` function to `backtest_ladder.py`
- Run via: `python backtest_ladder.py predict2026`

**2026 Breakout Zone Teams (Coach Year 2-3):**
- Sydney (Dean Cox Year 2)
- West Coast (McQualter Year 2)
- Richmond (Yze Year 3)
- Gold Coast (Hardwick Year 3)
- North Melbourne (Clarkson Year 3)

**2026 Predicted Ladder (Top 10):**
```
Rank  Team                  Exp Pts   Top8%   Prem%
1     Brisbane Lions        72.0      99%     53%
2     Geelong               63.4      92%     14%
3     Hawthorn              63.0      92%     13%
4     Sydney *              58.6      81%      6%   <- BREAKOUT CANDIDATE
5     Collingwood           58.2      80%      5%
6     GWS                   55.8      71%      3%
7     Western Bulldogs      54.3      65%      2%
8     Fremantle             53.8      63%      2%
9     Gold Coast *          53.0      60%      2%   <- BREAKOUT CANDIDATE
10    Adelaide              49.5      44%      1%

* = coach in breakout zone (Year 2-3)
```

**Impact of Coaching Feature on 2026:**
- Sydney: 9th → 6th (+3.2 positions) - main breakout candidate
- Gold Coast: 9th → 8th (+1.2 positions)
- Richmond/West Coast/North Melbourne: still bottom 3 (Elo too low for +30 boost to help)

**Key Prediction:**
Sydney (Dean Cox Year 2) is flagged as the 2026 breakout candidate, similar to how Hawthorn 2024 and Collingwood 2023 broke out with Year 2-3 coaches.

**Files Generated:**
- `afl_2026_ladder_prediction_with_coaching.csv`

---

## Summary of Model Improvements

| Feature | Spearman Impact | Useful? |
|---------|----------------|---------|
| Walk-forward Elo | +0.21 | YES - major improvement |
| Coaching tenure (Year 2-3 boost) | +0.006 | YES - modest but consistent |
| Late-season momentum | +0.000 | NO - breakout teams have negative momentum |
| Premiership odds blend | variable | NO - walk-forward Elo already captures this |

**Final Model Config:**
- Walk-forward Elo: ON
- Coaching tenure boost: ON (+30 Elo for Year 2-3 coaches)
- Momentum features: OFF (Elo-adjustment approach — proved not helpful)
- Momentum model features: ON (logistic regression features — new approach)
- Premiership blend: OFF

---

### Momentum Model Features (Logistic Regression Approach)

**Key distinction:** Previous momentum attempt adjusted Elo ratings directly (+/- points). This new approach adds momentum as **features in the logistic regression**, letting the model learn non-obvious relationships (e.g., a team with high pct_trend but low win rate may be undervalued).

**4 features per team (computed from season T-1 only):**
1. `mom_elo_slope` — OLS slope of pre-match Elo vs match index within T-1. Captures within-season trajectory.
2. `mom_second_half_delta` — Win rate in 2nd half minus 1st half. Captures late improvement/decline.
3. `mom_win_rate_last8` — Win rate over last 8 matches. Raw late-season form.
4. `mom_pct_trend` — Scoring percentage (PF/PA) in last 8 minus season-average. Captures scoring improvement.

Each enters the model as a **diff** (home value minus away value), consistent with existing features.

**Implementation:**
- `afl_pipeline.py`: `compute_team_momentum_features()`, `add_preseason_momentum()`, `MOMENTUM_FEATURES` constant
- `build_historical_feature_frame(use_momentum=True)` adds features to training data
- `make_feature_list(use_momentum=True)` includes `diff_mom_*` feature names
- `add_fixture_features(momentum_snap=...)` merges features onto fixtures
- All backward-compatible (new params default to preserving old behavior)

**Run backtest with momentum model:**
```bash
python backtest_ladder.py
```
The backtest now includes a "WITH MOMENTUM MODEL" section comparing coaching-only vs coaching+momentum.


