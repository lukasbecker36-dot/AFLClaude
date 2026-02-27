# AFL Match Prediction & Ladder Forecasting

A Python-based system for predicting AFL (Australian Football League) match outcomes and generating season ladder forecasts. Combines Elo ratings, rolling team statistics, coaching tenure analysis, momentum/trajectory model features, list age priors, and optional bookmaker odds blending.

## Project Status

### What's Working

**Match-level prediction pipeline** — fully operational:
- Scrapes team-level match stats from afltables.com (`scrape_afl_tables.py`)
- Scrapes team average ages by season from AFL Tables (`scrape_list_ages.py`)
- Builds features: Elo ratings, 5-game rolling form (Inside 50s, Clearances, Disposals), list age priors
- Momentum/trajectory features derived from prior season (Elo slope, 2nd-half win rate delta, last-8 win rate, scoring percentage trend)
- Trains logistic regression with Platt calibration using time-series cross-validation
- Generates weekly tipping predictions with optional market odds blending (`run_weekly.py`)
- Historical data through 2025 season stored in `afl_team_rows.csv`

**Ladder backtesting** — fully operational:
- Backtests predicted ladders against actual results for 2023, 2024, 2025
- Walk-forward Elo (updates ratings after each match using real results)
- Monte Carlo ladder simulation (20,000 iterations)
- Coaching tenure feature (+30 Elo boost for coaches in year 2-3 of tenure)
- Momentum model features (4 per-team trajectory signals fed as logistic regression features)
- List age priors (team average age as a model feature, young lists may have upside)
- Comprehensive evaluation: Spearman correlation, mean absolute rank error, top 4/8 accuracy, minor premier/wooden spoon hits, Brier score, match-level log loss

**2026 predictions** — generated:
- Pre-season ladder forecast using walk-forward Elo + coaching tenure + momentum features + list age priors
- Hardcoded 2026 team ages (scraped data covers through 2025)
- Breakout candidates identified based on coaching tenure (Sydney, West Coast, Richmond, Gold Coast, North Melbourne)

### Current Performance

| Metric | Static Elo (Baseline) | Walk-forward Elo | + Coaching | + Momentum Model | + List Age Priors |
|--------|----------------------|-------------------|------------|------------------|-------------------|
| Spearman | ~0.57 | ~0.78 | ~0.78 | ~0.78 | ~0.77 |
| MARE | ~3.33 | ~2.45 | ~2.38 | ~2.46 | ~2.53 |
| Top 8 | — | ~71% | 75% (18/24) | 79% (19/24) | 75% (18/24) |

Walk-forward Elo is the single biggest improvement. Coaching tenure adds a modest gain to MARE. Momentum model features improve Top 8 accuracy (notably 2025: 6/8 → 7/8) at the cost of slightly higher MARE. List age priors show neutral performance on current 3-season backtest; may benefit from additional features (pct_prime_age, etc.).

### What Didn't Work

- **Momentum as Elo adjustments** — the earlier approach of directly adjusting Elo by a momentum score was counterproductive because breakout teams (Hawthorn 2024, Adelaide 2025) had *negative* momentum. The new approach adds momentum as logistic regression features instead, letting the model learn non-obvious relationships.
- **Premiership odds blending** (for ladder) — walk-forward Elo already captures what market odds would add; best blend weight is 0.0 (pure model)

### Known Weaknesses

Breakout seasons remain the hardest problem:
- Adelaide 2025: predicted ~10th, finished 1st
- Hawthorn 2024: predicted ~13th, finished 7th
- Geelong 2023: predicted ~2nd, finished 12th

Market odds also missed these, suggesting this is a structural limitation of any pre-season model.

### Unfinished / Next Steps

1. **External data**:
   - List age priors: **implemented** (source: AFL Tables Player Ages — `python scrape_list_ages.py`)
     - Current: average age only (neutral performance on 3-season backtest)
     - Future: add pct_prime_age, median_age, age distribution features
   - Player movement / list turnover data (research done, implementation pending)
2. **Adelaide 2025-type breakouts** remain unsolved — coaching tenure helps new-coach breakouts (year 2-3), but not long-tenure coaches with maturing young lists. List age features may help identify these patterns with more data/refinement.

## Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy requests beautifulsoup4 matplotlib pytest
```

### Commands

**Scrape latest match data:**
```bash
python scrape_afl_tables.py
```

**Scrape team list ages:**
```bash
python scrape_list_ages.py
```

**Generate weekly predictions:**
```bash
python run_weekly.py --predict_season 2026 --predict_round 1
```

**Run ladder backtests (2023-2025):**
```bash
python backtest_ladder.py
```

**Generate 2026 ladder prediction:**
```bash
python backtest_ladder.py predict2026
```

**Run season ladder simulation:**
```bash
python data/afl_season_sim.py
```

**Cross-validate blend weight:**
```bash
python cross_validate_blend.py
```

**Run tests:**
```bash
pytest tests/ -v
```

## Architecture

```
scrape_afl_tables.py     Scrapes afltables.com -> afl_team_rows.csv
scrape_list_ages.py      Scrapes AFL Tables ages -> data/team_season_list_priors.csv
        |
  afl_pipeline.py        Core pipeline: Elo, form, momentum, list priors, model training, calibration
        |
   +---------+-----------+
   |                     |
run_weekly.py     backtest_ladder.py
Weekly tips       Ladder backtesting & 2026 predictions
   |                     |
   |              data/afl_season_sim.py
   |              Monte Carlo ladder simulation
   |
afl_win_model.py         Extended model (venue/travel, weather, hybrid tipping)
cross_validate_blend.py   Blend weight tuning
```

### Core Modules

| File | Purpose |
|------|---------|
| `afl_pipeline.py` | Reusable pipeline: feature building, Elo, form stats, momentum features, list priors, model training, Platt calibration |
| `backtest_ladder.py` | Historical ladder backtesting, coaching tenure features, momentum model, list age priors, walk-forward Elo, 2026 predictions |
| `run_weekly.py` | Weekly prediction generation with market odds blending |
| `scrape_afl_tables.py` | Incremental scraper for afltables.com match stats |
| `scrape_list_ages.py` | Scraper for AFL Tables player ages page (team average ages by season) |
| `afl_win_model.py` | Extended model with venue/travel features, weather integration, hybrid tipping |
| `data/afl_season_sim.py` | Monte Carlo ladder simulation, rank evolution plots |
| `cross_validate_blend.py` | Time-series cross-validation for model/market blend weight |

### Data Files

| File | Description |
|------|-------------|
| `afl_team_rows.csv` | Historical team-level match stats (scraped from afltables.com) |
| `aflodds.csv` | Historical bookmaker odds (OddsPortal format, header on row 2) |
| `fixtures_2026.csv` | 2026 season fixtures |
| `weather_cache.csv` | Cached weather API responses |
| `data/team_season_list_priors.csv` | Team average ages by season (scraped from AFL Tables) |

### Tests

| File | Description |
|------|-------------|
| `tests/test_list_ages.py` | Unit tests for list age priors feature (join correctness, missing priors handling, scraper output validation) |

## Model Details

- **Algorithm**: Logistic regression with L2 regularisation (C=0.2), Platt-calibrated probabilities
- **Core features**: Elo difference + rolling 5-game form differences for Inside 50s, Clearances, Disposals
- **Momentum features** (optional, computed from prior season):
  - `mom_elo_slope` — OLS slope of pre-match Elo vs match index (within-season trajectory)
  - `mom_second_half_delta` — win rate 2nd half minus 1st half (late improvement/decline)
  - `mom_win_rate_last8` — win rate over last 8 matches (raw late-season form)
  - `mom_pct_trend` — scoring percentage last 8 minus season average (scoring improvement)
- **List age priors** (optional, scraped from AFL Tables):
  - `list_avg_age` — team average age for the season (young lists may have upside, older lists may decline)
  - Extensible for future features: pct_prime_age, median_age, etc.
- All features enter the model as **diffs** (home value minus away value)
- **Elo config**: K=25, home advantage=65, base=1500
- **Weekly blend**: 84% market odds / 16% model (for match tipping)
- **Ladder blend**: 0% market (pure model with walk-forward Elo performs best)
- **Coaching boost**: +30 Elo for teams with coaches in year 2-3 of tenure
- **Simulation**: 20,000 Monte Carlo iterations with 0.9 shrinkage toward 0.5

## Configuration

Key settings are defined as constants at the top of each script. The most important:

| Setting | Value | Location |
|---------|-------|----------|
| `USE_WALKFORWARD_ELO` | `True` | `backtest_ladder.py` |
| `USE_COACHING_FEATURES` | `True` | `backtest_ladder.py` |
| `USE_MOMENTUM_FEATURES` | `False` | `backtest_ladder.py` |
| `USE_MOMENTUM_MODEL_FEATS` | `True` | `backtest_ladder.py` |
| `USE_LIST_PRIORS` | `True` | `backtest_ladder.py` |
| `PREMIERSHIP_BLEND_W` | `0.0` | `backtest_ladder.py` |
| `DEFAULT_BLEND_W` | `0.84` | `run_weekly.py` |
| `N_SIMS` | `20000` | `backtest_ladder.py` |

Note: `USE_MOMENTUM_FEATURES` controls the older Elo-adjustment approach (off). `USE_MOMENTUM_MODEL_FEATS` controls the newer logistic regression feature approach (on). `USE_LIST_PRIORS` adds team average age as a model feature.
