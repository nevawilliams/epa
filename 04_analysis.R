options(width = 200)

library(dplyr)
library(xgboost)
library(nflfastR)

ep_model  <- xgboost::xgb.load("ep_model.json")
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

probs <- predict(
  ep_model,
  xgboost::xgb.DMatrix(as.matrix(test_data[, feature_cols])),
  reshape = TRUE
)
test_data$ep_new <- as.numeric(probs %*% point_values)

pbp_2025 <- nflfastR::load_pbp(2025) |>
  tibble::as_tibble() |>
  dplyr::filter(
    !is.na(down), !is.na(epa),
    play_type %in% c("pass", "run"),
    qb_spike == 0, qb_kneel == 0
  )

pbp_2025 <- pbp_2025 |>
  dplyr::left_join(
    test_data |> dplyr::select(game_id, play_id, ep_new),
    by = c("game_id", "play_id")
  ) |>
  dplyr::arrange(game_id, game_half, play_id) |>
  dplyr::group_by(game_id, game_half) |>
  dplyr::mutate(
    next_posteam = dplyr::lead(posteam),
    ep_next      = dplyr::lead(ep_new),
    ep_next_adj  = dplyr::case_when(
      is.na(ep_next)          ~ 0,
      next_posteam != posteam ~ -ep_next,
      TRUE                    ~ ep_next
    ),
    epa_new = ep_next_adj - ep_new,
    epa_new = dplyr::case_when(
      !is.na(td_team) & td_team == posteam ~  7 - ep_new,
      !is.na(td_team) & td_team != posteam ~ -7 - ep_new,
      field_goal_result == "made"          ~  3 - ep_new,
      safety == 1                          ~  2 - ep_new,
      TRUE                                 ~ epa_new
    ),
    epa_new = dplyr::if_else(
      is.na(dplyr::lead(play_id)), 0 - ep_new, epa_new
    )
  ) |>
  dplyr::ungroup()

pbp_competitive <- pbp_2025 |>
  dplyr::filter(game_seconds_remaining > 300, wp >= 0.05 & wp <= 0.95)

cat("Total 2025 plays:", nrow(pbp_2025), "\n")
cat("Competitive plays:", nrow(pbp_competitive), "\n")

cat("\nTeam EPA — new vs standard (competitive plays, 2025):\n")
pbp_competitive |>
  dplyr::filter(!is.na(epa_new)) |>
  dplyr::group_by(posteam) |>
  dplyr::summarise(
    n            = dplyr::n(),
    mean_epa_std = round(mean(epa,     na.rm = TRUE), 4),
    mean_epa_new = round(mean(epa_new, na.rm = TRUE), 4),
    .groups = "drop"
  ) |>
  dplyr::arrange(dplyr::desc(mean_epa_new)) |>
  print(n = 32, width = 200)

cat("\n2nd & Short (<=3 yards to go) EPA by play type:\n")
pbp_competitive |>
  dplyr::filter(down == 2, ydstogo <= 3, !is.na(epa_new)) |>
  dplyr::group_by(play_type) |>
  dplyr::summarise(
    n            = dplyr::n(),
    mean_epa_std = round(mean(epa,     na.rm = TRUE), 4),
    mean_epa_new = round(mean(epa_new, na.rm = TRUE), 4),
    .groups = "drop"
  ) |>
  print(width = 200)

cat("\n2nd & Short by field position zone:\n")
pbp_competitive |>
  dplyr::filter(down == 2, ydstogo <= 3, !is.na(epa_new)) |>
  dplyr::mutate(
    zone = dplyr::case_when(
      yardline_100 <= 20 ~ "Red Zone",
      yardline_100 <= 40 ~ "Scoring Position",
      yardline_100 <= 60 ~ "Midfield",
      TRUE               ~ "Own Territory"
    )
  ) |>
  dplyr::group_by(zone, play_type) |>
  dplyr::summarise(
    n            = dplyr::n(),
    mean_epa_std = round(mean(epa,     na.rm = TRUE), 4),
    mean_epa_new = round(mean(epa_new, na.rm = TRUE), 4),
    .groups = "drop"
  ) |>
  dplyr::arrange(zone, play_type) |>
  print(width = 200)

cat("\n2nd & Short by game state:\n")
pbp_competitive |>
  dplyr::filter(down == 2, ydstogo <= 3, !is.na(epa_new)) |>
  dplyr::mutate(
    game_state = dplyr::case_when(
      score_differential >  7 ~ "Leading",
      score_differential < -7 ~ "Trailing",
      TRUE                    ~ "Close"
    )
  ) |>
  dplyr::group_by(game_state, play_type) |>
  dplyr::summarise(
    n            = dplyr::n(),
    mean_epa_std = round(mean(epa,     na.rm = TRUE), 4),
    mean_epa_new = round(mean(epa_new, na.rm = TRUE), 4),
    .groups = "drop"
  ) |>
  dplyr::arrange(game_state, play_type) |>
  print(width = 200)

cat("\nLAC 2nd & Short breakdown:\n")
pbp_competitive |>
  dplyr::filter(posteam == "LAC", down == 2, ydstogo <= 3, !is.na(epa_new)) |>
  dplyr::group_by(play_type) |>
  dplyr::summarise(
    n            = dplyr::n(),
    mean_epa_std = round(mean(epa,     na.rm = TRUE), 4),
    mean_epa_new = round(mean(epa_new, na.rm = TRUE), 4),
    .groups = "drop"
  ) |>
  print(width = 200)


cat("\nRank changes vs standard EPA:\n")
pbp_competitive |>
  dplyr::filter(!is.na(epa_new)) |>
  dplyr::group_by(posteam) |>
  dplyr::summarise(
    mean_std = round(mean(epa,     na.rm = TRUE), 4),
    mean_new = round(mean(epa_new, na.rm = TRUE), 4),
    .groups = "drop"
  ) |>
  dplyr::mutate(
    rank_std  = rank(-mean_std),
    rank_new  = rank(-mean_new),
    rank_diff = rank_std - rank_new
  ) |>
  dplyr::arrange(dplyr::desc(abs(rank_diff))) |>
  print(n = 32, width = 200)

saveRDS(pbp_competitive, "results_competitive.rds")