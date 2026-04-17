library(nflfastR)
library(xgboost)
library(tidyverse)
pbp_data <- load_pbp(2010:2025)

pbp_data
plays <- pbp_data %>%
  filter(
    !is.na(epa),
    !is.na(down),
    !is.na(wp),
    !is.na(yardline_100)
  )

nrow(plays)

?nflfastR::calculate_expected_points

plays %>%
  select(
    season, home_team, posteam, roof,
    half_seconds_remaining, yardline_100,
    down, ydstogo,
    posteam_timeouts_remaining,
    defteam_timeouts_remaining
  ) %>%
  head(5) %>%
  nflfastR::calculate_expected_points()

plays %>%
  select(ep, epa) %>%
  summary()

plays <- plays %>%
  mutate(
    # garbage time indicator
    # game is competitive if wp is between 10% and 90%
    competitive = if_else(wp >= 0.10 & wp <= 0.90, 1, 0),
    
    # continuous weight based on how competitive the game is
    # plays closer to 50/50 get full weight
    # blowouts get downweighted
    garbage_weight = case_when(
      wp >= 0.10 & wp <= 0.90 ~ 1.0,
      wp >= 0.05 & wp <= 0.95 ~ 0.5,
      TRUE ~ 0.0
    )
  )

# check how many plays are garbage time
plays %>%
  count(competitive) %>%
  mutate(pct = n / sum(n))


# 24 percent ( alot )

# build defensive strength in competitive situations only
def_strength <- plays %>%
  filter(competitive == 1) %>%
  group_by(defteam, season) %>%
  summarize(
    def_epa_allowed = mean(epa, na.rm = TRUE),
    .groups = "drop"
  )

# merge back and add all context features
plays <- plays %>%
  left_join(def_strength, by = c("defteam", "season")) %>%
  mutate(
    home = if_else(posteam == home_team, 1, 0),
    home = if_else(location == "Neutral", 0, home)
  )

# check

glimpse(plays %>% select(competitive, garbage_weight, def_epa_allowed, home, shotgun, no_huddle, pass)) 


