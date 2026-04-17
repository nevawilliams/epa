
library(dplyr)
library(xgboost)
library(nflfastR)

# -----------------------------------------------------------------------------
# LOAD MODEL AND VALIDATION DATA
# -----------------------------------------------------------------------------
# Load the trained model and 2024 validation data from disk.
# We never retrain here - just load and evaluate.
# -----------------------------------------------------------------------------
ep_model   <- xgboost::xgb.load("ep_model.json")
val_data   <- readRDS("val_data.rds")
train_data <- readRDS("train_data.rds")

# -----------------------------------------------------------------------------
# COMPUTE EP FROM MODEL PREDICTIONS
# -----------------------------------------------------------------------------
# The model outputs probabilities for each of the 7 scoring outcomes.
# EP is the weighted sum of those probabilities multiplied by their
# point values. This is exactly how nflfastR computes EP internally.
#
# Score outcomes and their point values:
# Class 0 = opponent TD   = -7 points
# Class 1 = opponent FG   = -3 points
# Class 2 = opponent safety = -2 points
# Class 3 = no score      =  0 points
# Class 4 = own safety    =  2 points
# Class 5 = own FG        =  3 points
# Class 6 = own TD        =  7 points
# -----------------------------------------------------------------------------

# point values corresponding to each class
point_values <- c(-7, -3, -2, 0, 2, 3, 7)

# feature columns must match exactly what the model was trained on
feature_cols <- c(
  "half_seconds_remaining",
  "yardline_100",
  "home",
  "retractable",
  "dome",
  "outdoors",
  "ydstogo",
  "down1", "down2", "down3", "down4",
  "posteam_timeouts_remaining",
  "defteam_timeouts_remaining",
  "score_differential",
  "season"
)

# build prediction matrix from validation data
X_val <- as.matrix(val_data[, feature_cols])
dval  <- xgboost::xgb.DMatrix(data = X_val)

# get raw probabilities from model - returns one probability per class per play
probs <- predict(ep_model, dval, reshape = TRUE)

# compute EP as weighted sum of probabilities times point values
val_data$ep_new <- as.numeric(probs %*% point_values)

# -----------------------------------------------------------------------------
# COMPUTE EPA FROM NEW EP
# -----------------------------------------------------------------------------
# EPA = EP after play minus EP before play.
# We compute this by taking the difference between consecutive EP values
# within each game and half, then joining back to the play level.
# -----------------------------------------------------------------------------
val_data <- val_data |>
  dplyr::arrange(season) |>
  dplyr::mutate(
    ep_new_next = dplyr::lead(ep_new),
    epa_new = ep_new_next - ep_new
  )

# -----------------------------------------------------------------------------
# LOAD 2024 FULL PBP FOR COMPARISON
# -----------------------------------------------------------------------------
# We need nflfastR's standard EPA to compare against our new EPA.
# Load the full 2024 pbp which contains nflfastR's pre-calculated EPA.
# -----------------------------------------------------------------------------
cat("Loading 2024 pbp for comparison...\n")
pbp_2024 <- nflfastR::load_pbp(2024) |>
  tibble::as_tibble() |>
  dplyr::filter(
    !is.na(down),
    !is.na(epa),
    !is.na(score_differential),
    play_type %in% c("pass", "run"),
    qb_spike == 0,
    qb_kneel == 0
  )

# -----------------------------------------------------------------------------
# QUICK SANITY CHECKS
# -----------------------------------------------------------------------------
# Before comparing metrics, verify our EP values make intuitive sense.
# EP should increase as you get closer to the end zone and
# should be higher in competitive situations than garbage time.
# -----------------------------------------------------------------------------

cat("\n--- EP Sanity Check: Field Position ---\n")
val_data |>
  dplyr::mutate(yardline_bucket = cut(yardline_100, breaks = seq(0, 100, 10))) |>
  dplyr::group_by(yardline_bucket) |>
  dplyr::summarise(mean_ep = mean(ep_new, na.rm = TRUE)) |>
  print()

cat("\n--- EP Sanity Check: Score Differential ---\n")
val_data |>
  dplyr::mutate(score_bucket = cut(score_differential, 
                                   breaks = c(-Inf, -21, -7, 0, 7, 21, Inf))) |>
  dplyr::group_by(score_bucket) |>
  dplyr::summarise(mean_ep = mean(ep_new, na.rm = TRUE)) |>
  print()

cat("\nValidation complete. Run 04_test.R for final evaluation.\n")