library(dplyr)
library(nflfastR)
library(zoo)

compute_def_sr <- function(pbp_raw) {
  pbp_raw |>
    dplyr::filter(
      !is.na(down), !is.na(yards_gained), !is.na(defteam), !is.na(wp),
      wp >= 0.05 & wp <= 0.95,
      play_type %in% c("pass", "run"),
      qb_spike == 0, qb_kneel == 0
    ) |>
    dplyr::arrange(game_id, play_id) |>
    dplyr::mutate(
      success = dplyr::case_when(
        down == 1         ~ yards_gained >= 0.4 * ydstogo,
        down == 2         ~ yards_gained >= 0.6 * ydstogo,
        down %in% c(3, 4) ~ yards_gained >= ydstogo,
        TRUE              ~ FALSE
      )
    ) |>
    dplyr::group_by(defteam) |>
    dplyr::mutate(def_sr = dplyr::lag(cummean(as.numeric(success)))) |>
    dplyr::ungroup() |>
    dplyr::select(game_id, play_id, defteam, def_sr)
}

prep_season <- function(yr) {
  pbp <- nflfastR::load_pbp(yr) |> tibble::as_tibble()
  
  def_sr <- compute_def_sr(pbp)
  
  # half-level label: next score this half from current offense's perspective
  # sign determined by which team scores, not score change direction
  # correctly captures opponent FGs/TDs that happen on future drives
  pbp <- pbp |>
    dplyr::arrange(game_id, game_half, play_id) |>
    dplyr::group_by(game_id, game_half) |>
    dplyr::mutate(
      posteam_score_change = posteam_score_post - posteam_score,
      defteam_score_change = defteam_score_post - defteam_score,
      
      score_team = dplyr::case_when(
        posteam_score_change == 6 ~ posteam,
        posteam_score_change == 3 ~ posteam,
        posteam_score_change == 2 ~ posteam,
        defteam_score_change == 6 ~ defteam,
        defteam_score_change == 2 ~ defteam,
        TRUE                      ~ NA_character_
      ),
      
      score_pts = dplyr::case_when(
        posteam_score_change == 6 ~ 7,
        posteam_score_change == 3 ~ 3,
        posteam_score_change == 2 ~ 2,
        defteam_score_change == 6 ~ 7,
        defteam_score_change == 2 ~ 2,
        TRUE                      ~ NA_real_
      ),
      
      next_score_team = zoo::na.locf(score_team, fromLast = TRUE, na.rm = FALSE),
      next_score_pts  = zoo::na.locf(score_pts,  fromLast = TRUE, na.rm = FALSE),
    
      label = dplyr::case_when(
        is.na(next_score_team)     ~  0,
        next_score_team == posteam ~  next_score_pts,
        TRUE                       ~ -next_score_pts
      )
    ) |>
    dplyr::ungroup()
  
  pbp <- pbp |>
    dplyr::filter(
      !is.na(down), !is.na(score_differential), !is.na(posteam), !is.na(wp),
      wp >= 0.05 & wp <= 0.95,
      play_type %in% c("pass", "run", "punt", "field_goal"),
      qb_spike == 0, qb_kneel == 0
    ) |>
    dplyr::mutate(
      is_pass = as.integer(play_type == "pass"),
      is_run  = as.integer(play_type == "run"),
      is_punt = as.integer(play_type == "punt"),
      is_fg   = as.integer(play_type == "field_goal"),
      home        = as.integer(posteam == home_team),
      retractable = as.integer(roof %in% c("closed", "open")),
      dome        = as.integer(roof == "dome"),
      outdoors    = as.integer(roof == "outdoors"),
      down1 = as.integer(down == 1),
      down2 = as.integer(down == 2),
      down3 = as.integer(down == 3),
      down4 = as.integer(down == 4),
      seasons_since_2015 = yr - 2015L
    ) |>
    dplyr::left_join(def_sr, by = c("game_id", "play_id")) |>
    dplyr::mutate(def_sr = tidyr::replace_na(def_sr, 0.5))
  
  out <- pbp |>
    dplyr::select(
      game_id, play_id, game_half, posteam,
      half_seconds_remaining, yardline_100,
      home, retractable, dome, outdoors,
      ydstogo, down1, down2, down3, down4,
      posteam_timeouts_remaining, defteam_timeouts_remaining,
      score_differential, seasons_since_2015,
      is_pass, is_run, is_punt, is_fg,
      def_sr, label
    ) |>
    dplyr::filter(dplyr::if_all(
      dplyr::all_of(c(
        "half_seconds_remaining", "yardline_100",
        "home", "retractable", "dome", "outdoors",
        "ydstogo", "down1", "down2", "down3", "down4",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "score_differential", "seasons_since_2015", "label"
      )),
      ~ !is.na(.)
    ))
  
  rm(pbp); gc()
  return(out)
}

train_data <- purrr::map_dfr(2014:2023, function(yr) {
  cat("Loading:", yr, "\n")
  prep_season(yr)
})

cat("label distribution:\n")
print(prop.table(table(train_data$label)))
cat("def_sr completeness:", round(mean(!is.na(train_data$def_sr)) * 100, 1), "%\n")

val_data  <- prep_season(2024)
test_data <- prep_season(2025)

saveRDS(train_data, "train_data.rds")
saveRDS(val_data,   "val_data.rds")
saveRDS(test_data,  "test_data.rds")

cat("train:", nrow(train_data), "| val:", nrow(val_data), "| test:", nrow(test_data), "\n")