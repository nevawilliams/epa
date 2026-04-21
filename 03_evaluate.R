library(xgboost)
library(dplyr)
library(nflfastR)

ep_model  <- xgboost::xgb.load("ep_model.json")
val_data  <- readRDS("val_data.rds")
test_data <- readRDS("test_data.rds")

feature_cols <- c(
  "half_seconds_remaining", "yardline_100",
  "home", "retractable", "dome", "outdoors",
  "ydstogo", "down1", "down2", "down3", "down4",
  "posteam_timeouts_remaining", "defteam_timeouts_remaining",
  "score_differential", "seasons_since_2015",
  "is_pass", "is_run", "is_punt", "is_fg",
  "def_sr"
)

point_values <- c(-7, -3, -2, 0, 2, 3, 7)
label_map    <- c("-7"=0, "-3"=1, "-2"=2, "0"=3, "2"=4, "3"=5, "7"=6)

mlogloss <- function(probs, labels) {
  idx <- cbind(1:nrow(probs), labels + 1L)
  -mean(log(pmax(probs[idx], 1e-15)))
}

score_split <- function(data, label, tag) {
  y     <- label_map[as.character(data[[label]])]
  dmat  <- xgboost::xgb.DMatrix(as.matrix(data[, feature_cols]))
  probs <- predict(ep_model, dmat, reshape = TRUE)
  ep    <- as.numeric(probs %*% point_values)
  loss  <- mlogloss(probs, y)
  
  cal <- data.frame(yardline_100 = data$yardline_100, ep = ep) |>
    dplyr::mutate(bucket = cut(yardline_100, breaks = seq(0, 100, 10),
                               include.lowest = TRUE)) |>
    dplyr::group_by(bucket) |>
    dplyr::summarise(mean_ep = round(mean(ep), 3), n = dplyr::n(),
                     .groups = "drop")
  
  cat("\n==", tag, "==\n")
  cat("  mlogloss :", round(loss, 4), "\n")
  cat("  n plays  :", nrow(data), "\n")
  cat("  EP range :", round(min(ep), 2), "to", round(max(ep), 2), "\n")
  cat("  Calibration by field position:\n")
  print(cal, n = Inf)
  
  invisible(list(probs = probs, ep = ep, loss = loss))
}

val_out  <- score_split(val_data,  "label", "VAL  2024")
test_out <- score_split(test_data, "label", "TEST 2025")

# nflfastR direct comparison on identical plays
pbp_2024 <- nflfastR::load_pbp(2024) |> tibble::as_tibble()
pbp_2025 <- nflfastR::load_pbp(2025) |> tibble::as_tibble()

compare_nflfastr <- function(pbp, model_data, tag) {
  joined <- pbp |>
    dplyr::select(game_id, play_id, td_prob, fg_prob, safety_prob,
                  opp_safety_prob, opp_fg_prob, opp_td_prob, no_score_prob) |>
    dplyr::inner_join(
      model_data |> dplyr::select(game_id, play_id, label),
      by = c("game_id", "play_id")
    ) |>
    dplyr::filter(!is.na(td_prob))
  
  prob_matrix <- joined |>
    dplyr::select(opp_td_prob, opp_fg_prob, opp_safety_prob,
                  no_score_prob, safety_prob, fg_prob, td_prob) |>
    as.matrix()
  
  y <- label_map[as.character(joined$label)]
  loss <- mlogloss(prob_matrix, y)
  cat("\nnflfastR mlogloss on", tag, ":", round(loss, 4), "\n")
  cat("n plays compared:", nrow(joined), "\n")
}

compare_nflfastr(pbp_2024, val_data,  "VAL  2024")
compare_nflfastr(pbp_2025, test_data, "TEST 2025")

# EP sanity check
cat("\nEP sanity check (1st & 10, neutral game state, avg defense):\n")
sanity <- data.frame(
  half_seconds_remaining     = 1800,
  yardline_100               = seq(5, 95, by = 10),
  home = 1, retractable = 0, dome = 0, outdoors = 1,
  ydstogo = 10, down1 = 1, down2 = 0, down3 = 0, down4 = 0,
  posteam_timeouts_remaining = 3, defteam_timeouts_remaining = 3,
  score_differential = 0, seasons_since_2015 = 8L,
  is_pass = 1, is_run = 0, is_punt = 0, is_fg = 0,
  def_sr = 0.5
)
p <- predict(ep_model,
             xgboost::xgb.DMatrix(as.matrix(sanity[, feature_cols])),
             reshape = TRUE)
sanity$ep <- as.numeric(p %*% point_values)
print(sanity[, c("yardline_100", "ep")])

# EP calibration vs nflfastR by field position
cat("\nEP vs nflfastR by field position (1st & 10, 2025):\n")
probs_test <- predict(ep_model,
                      xgboost::xgb.DMatrix(as.matrix(test_data[, feature_cols])),
                      reshape = TRUE)
test_data$ep_new <- as.numeric(probs_test %*% point_values)

pbp_2025 |>
  dplyr::filter(!is.na(down), !is.na(ep),
                play_type %in% c("pass", "run"),
                down == 1, ydstogo == 10,
                qb_spike == 0, qb_kneel == 0) |>
  dplyr::left_join(
    test_data |> dplyr::select(game_id, play_id, ep_new),
    by = c("game_id", "play_id")
  ) |>
  dplyr::filter(!is.na(ep_new)) |>
  dplyr::mutate(bucket = cut(yardline_100, breaks = seq(0, 100, 10),
                             include.lowest = TRUE)) |>
  dplyr::group_by(bucket) |>
  dplyr::summarise(
    nflfastr_ep = round(mean(ep,     na.rm = TRUE), 3),
    our_ep      = round(mean(ep_new, na.rm = TRUE), 3),
    diff        = round(mean(ep_new - ep, na.rm = TRUE), 3),
    n           = dplyr::n(),
    .groups     = "drop"
  ) |>
  print(n = Inf, width = 200)

# feature importance
cat("\nFeature importance:\n")
importance <- xgboost::xgb.importance(feature_names = feature_cols,
                                      model = ep_model)
print(importance)