
library(dplyr)
library(xgboost)
library(nflfastR)

# -----------------------------------------------------------------------------
# DATA PREPARATION FUNCTION
# -----------------------------------------------------------------------------
# This function takes a single NFL season (year) and returns a clean, 
# model-ready dataset. We process one year at a time to avoid loading
# all seasons into memory simultaneously, which caused crashes earlier.
# -----------------------------------------------------------------------------

prep_season <- function(yr) {
  
  # Load the raw play-by-play data for the given season
  # This contains every play from every game including metadata,
  # game state variables, and nflfastR's pre-calculated EP/EPA values
  pbp <- nflfastR::load_pbp(yr) |>
    tibble::as_tibble()
  
  # -------------------------------------------------------------------------
  # FILTERING
  # We only want real offensive plays where EPA is meaningful.
  # Removing special teams, penalties, and clock-management plays
  # ensures we're training on situations where field position,
  # down, distance, and score actually drive decision-making.
  # -------------------------------------------------------------------------
  pbp <- pbp |>
    dplyr::filter(
      !is.na(down),              # removes kickoffs, punts, extra points
      !is.na(epa),               # removes plays where EPA isn't defined
      !is.na(score_differential),# removes plays where score isn't recorded
      play_type %in% c("pass", "run"), # keeps only real offensive plays
      qb_spike == 0,             # removes intentional spikes to stop clock
      qb_kneel == 0              # removes intentional kneels to run clock
    )
  
  
#-------------------------------------------------------------------------
    # FEATURE ENGINEERING VIA nflfastR's INTERNAL FUNCTION
    # make_model_mutations() creates several derived variables that the
    # original nflfastR EP model uses internally, including:
    #   - down1, down2, down3, down4: binary dummy variables for each down
    #     (e.g. down1 = 1 if it's 1st down, 0 otherwise)
    #   - home: binary indicator for whether the possession team is home
    #   - retractable, dome, outdoors: binary dummies for roof type
    #     (converted from the single 'roof' text column)
    #   - era0 through era4: binary dummies for NFL era groupings
    #     (we drop these in favor of continuous season variable below)
    # -------------------------------------------------------------------------
  pbp <- pbp |>
    nflfastR:::make_model_mutations() |>
    as.data.frame()  # convert from data.table to plain dataframe to avoid conflicts
  
   pbp <- pbp |>
    dplyr::arrange(game_id, game_half, play_id) |>
    dplyr::group_by(game_id, game_half) |>
    dplyr::mutate(
      # calculate score changes on each play
      posteam_score_change = posteam_score_post - posteam_score,
      defteam_score_change = defteam_score_post - defteam_score,
      
      # encode the scoring event that occurred on this play
      score_event = dplyr::case_when(
        posteam_score_change >= 6 ~ 7,  # own TD
        posteam_score_change == 3 ~ 3,  # own FG
        posteam_score_change == 2 ~ 2,  # own safety
        defteam_score_change >= 6 ~ -7, # opp TD
        defteam_score_change == 3 ~ -3, # opp FG
        defteam_score_change == 2 ~ -2, # opp safety
        TRUE ~ 0                         # no score on this play
      ),
      
      # for each play, look forward and find the next non-zero score event
      # this becomes our target - what actually happened scoring-wise
      # after this play in the same half
      label = dplyr::lead(score_event, default = 0)
    ) |>
    dplyr::ungroup()
  


  # -------------------------------------------------------------------------
  # VARIABLE SELECTION
  # We select only the columns needed for modeling.
  # This matches the original nflfastR EP model features (from ep_model_select)
  # PLUS our two additions: score_differential and season.
  #
  # WHY THESE SPECIFIC VARIABLES:
  # half_seconds_remaining  - time left in half drives urgency of possession
  # yardline_100            - distance from opponent end zone (100 = own endzone)
  # home                    - small home field advantage adjustment
  # retractable/dome/outdoors - weather/environment affects scoring rates
  # ydstogo                 - yards needed for first down
  # down1-down4             - which down it is as binary indicators
  # posteam_timeouts_remaining  - offense timeouts affect late game value
  # defteam_timeouts_remaining  - defense timeouts affect late game value
  #
  # OUR ADDITIONS:
  # score_differential      - THE KEY IMPROVEMENT: standard EPA ignores the
  #                           score entirely. A 5 yard gain means completely
  #                           different things when up 21 vs down 21.
  #                           This is the primary weakness we are fixing.
  # season                  - replaces nflfastR's era dummy variables.
  #                           Instead of hardcoded era buckets, we let the
  #                           model find its own year-over-year trends in
  #                           scoring environment as a continuous variable.
  #
  # label                   - our target variable (next scoring outcome)
  # -------------------------------------------------------------------------
  out <- pbp |>
    dplyr::select(
      # original nflfastR EP model features
      half_seconds_remaining,
      yardline_100,
      home,
      retractable,
      dome,
      outdoors,
      ydstogo,
      down1, down2, down3, down4,
      posteam_timeouts_remaining,
      defteam_timeouts_remaining,
      # our additions
      score_differential,
      season,
      # target variable
      label
    ) |>
    dplyr::filter(dplyr::if_all(dplyr::everything(), ~ !is.na(.)))
  
  # -------------------------------------------------------------------------
  # MEMORY MANAGEMENT
  # Explicitly delete the raw pbp object and call garbage collection
  # to free RAM before the next season loads. Without this, R holds
  # multiple seasons in memory simultaneously and crashes.
  # -------------------------------------------------------------------------
  rm(pbp)
  gc()
  
  return(out)
}

# -----------------------------------------------------------------------------
# LOAD ALL TRAINING SEASONS
# -----------------------------------------------------------------------------
# We train on 2015-2023 for two reasons:
# 1. Modern football - by 2015 the pass-friendly rule changes are fully
#    established and the game reflects what coaches are running today
# 2. We hold out 2024-2025 as validation and test sets to prove the model
#    works on data it has never seen, which is critical for credibility
# -----------------------------------------------------------------------------
train_data <- purrr::map_dfr(2015:2023, function(yr) {
  cat("Loading season:", yr, "\n")
  prep_season(yr)
})

# quick sanity check - should show ~200k+ rows and 16 columns
cat("Training data dimensions:", nrow(train_data), "rows,", ncol(train_data), "columns\n")



# -----------------------------------------------------------------------------
# VALIDATION SET (2024)
# -----------------------------------------------------------------------------
# Used during model development to check performance and tune hyperparameters.
# Because we look at this repeatedly during building, it indirectly influences
# model decisions - so it cannot be used as a final performance measure.
# -----------------------------------------------------------------------------
val_data <- purrr::map_dfr(2024, function(yr) {
  cat("Loading validation season:", yr, "\n")
  prep_season(yr)
})
saveRDS(val_data, "val_data.rds")

# -----------------------------------------------------------------------------
# TEST SET (2025)
# -----------------------------------------------------------------------------
# Touched exactly once when the model is completely finished.
# This is the most recent complete season - proving it works on 2025 data
# is the strongest possible pitch to a HC making decisions in 2026.
# -----------------------------------------------------------------------------
test_data <- purrr::map_dfr(2025, function(yr) {
  cat("Loading test season:", yr, "\n")
  prep_season(yr)
})
saveRDS(test_data, "test_data.rds")

# -----------------------------------------------------------------------------
# SAVE TRAINING DATA
# -----------------------------------------------------------------------------
saveRDS(train_data, "train_data.rds")

cat("All datasets saved.\n")
cat("Train:", nrow(train_data), "rows\n")
cat("Validation:", nrow(val_data), "rows\n")
cat("Test:", nrow(test_data), "rows\n")



#Train: 308,054 rows — 9 seasons (2015-2023)
#Validation: 34,903 rows — 2024 season
#Test: 34,503 rows — 2025 season
