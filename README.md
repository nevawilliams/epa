# Context-Adjusted EPA Model
**Los Angeles Chargers Football Analysis Fellowship**
**Author:** Neva Williams

## Overview
XGBoost multiclass Expected Points model trained on competitive NFL situations (WP 5-95%, 2014-2023).
Primary innovation: a defensive success rate feature (`def_sr`) that adjusts expected points
for opponent quality in real time using only raw play outcomes — no external EPA model involved.
The feature updates weekly with strict lag to prevent data leakage.

## Performance
| Dataset | nflfastR mlogloss | Our mlogloss | Improvement |
|---------|------------------|--------------|-------------|
| Val 2024 | 1.2872 | 1.2679 | +1.5% |
| Test 2025 | 1.2763 | 1.2553 | +1.6% |

Evaluated on identical plays, same labels, direct apples-to-apples comparison.

## Key Design Decisions
- **Training window:** 2014-2023 — 2014 is the first season NFL pass rate exceeded 58.9% and remained elevated, marking the start of the modern passing era
- **Competitive play filter:** WP 5-95% removes garbage time from both training and evaluation
- **Chronological split:** train 2014-2023, val 2024, test 2025 — simulates real deployment
- **Label construction:** next scoring event this half from possession team perspective, sign determined by scoring team identity
- **def_sr:** defensive success rate with a lag — no data leakage

## Features
| Feature | Description | In nflfastR EP? |
|---------|-------------|-----------------|
| half_seconds_remaining | Time remaining in half | Yes |
| yardline_100 | Field position | Yes |
| score_differential | Score margin | No |
| def_sr | Defensive success rate allowed | No |
| down1-4 | Down indicators | Yes |
| ydstogo | Yards to first down | Yes |
| posteam/defteam_timeouts | Timeouts remaining | Yes |
| home/dome/retractable/outdoors | Stadium and roof type | Yes |
| is_pass/run/punt/fg | Play type indicators | No |
| seasons_since_2015 | Continuous era adjustment | Yes (buckets) |

## Files
- `01_prep_data.R` — data preparation, label construction, defensive SR computation
- `02_train_model.R` — XGBoost training with early stopping and sanity checks
- `03_evaluate.R` — mlogloss evaluation, nflfastR comparison, feature importance
- `04_analysis.R` — 2nd and short analysis and team EPA comparisons

## Requirements
```r
install.packages(c("dplyr", "nflfastR", "xgboost", "zoo",
                   "tidyr", "purrr", "tibble"))
```

## Usage
Run scripts in order. Each script reads from and writes to the working directory.
```r
source("01_prep_data.R")   # ~60 min — generates train/val/test RDS files
source("02_train_model.R") # ~25 min — generates ep_model.json
source("03_evaluate.R")    # ~5 min  — prints evaluation metrics
source("04_analysis.R")    # ~5 min  — prints 2nd and short analysis
```

## Model Architecture
- **Algorithm:** XGBoost multiclass classification (multi:softprob, 7 classes)
- **Target:** Next scoring event this half from possession team perspective
- **Classes:** -7 (opp TD), -3 (opp FG), -2 (opp safety), 0 (no score), 2 (safety), 3 (FG), 7 (TD)
- **Expected Points:** weighted sum of class probabilities times point values
- **EPA:** EP after play minus EP before play, with scoring play overrides

## References
- Carter and Machol (1971) — original Expected Points framework
- Yurko, Ventura and Horowitz (2018) — nflWAR multinomial EP model
- Baldwin (2021) — nflfastR EP/WP model documentation

