library(xgboost)

train_data <- readRDS("train_data.rds")
val_data   <- readRDS("val_data.rds")

feature_cols <- c(
  "half_seconds_remaining", "yardline_100",
  "home", "retractable", "dome", "outdoors",
  "ydstogo", "down1", "down2", "down3", "down4",
  "posteam_timeouts_remaining", "defteam_timeouts_remaining",
  "score_differential", "season",
  "is_pass", "is_run", "is_punt", "is_fg",
  "rolling_def_sr"
)

label_map    <- c("-7"=0, "-3"=1, "-2"=2, "0"=3, "2"=4, "3"=5, "7"=6)
point_values <- c(-7, -3, -2, 0, 2, 3, 7)

y_train <- label_map[as.character(train_data$label)]
y_val   <- label_map[as.character(val_data$label)]

dtrain <- xgboost::xgb.DMatrix(
  data  = as.matrix(train_data[, feature_cols]),
  label = y_train
)
dval <- xgboost::xgb.DMatrix(
  data  = as.matrix(val_data[, feature_cols]),
  label = y_val
)

cat("Training matrix:", nrow(train_data), "rows,", length(feature_cols), "features\n")

params <- list(
  objective        = "multi:softprob",
  num_class        = 7,
  eval_metric      = "mlogloss",
  eta              = 0.025,
  max_depth        = 5,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  gamma            = 1
)

set.seed(42)
ep_model <- xgboost::xgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 300,
  watchlist             = list(train = dtrain, val = dval),
  early_stopping_rounds = 25,
  print_every_n         = 25,
  verbose               = 1
)

xgboost::xgb.save(ep_model, "ep_model.json")
cat("Best iteration:", ep_model$best_iteration, "\n")
cat("Best val mlogloss:", ep_model$best_score, "\n")

# sanity check
test_cases <- data.frame(
  half_seconds_remaining = 1800,
  yardline_100           = c(5, 25, 50, 75, 95),
  home = 1, retractable = 0, dome = 0, outdoors = 1,
  ydstogo = 10, down1 = 1, down2 = 0, down3 = 0, down4 = 0,
  posteam_timeouts_remaining = 3, defteam_timeouts_remaining = 3,
  score_differential = 0, season = 2023,
  is_pass = 1, is_run = 0, is_punt = 0, is_fg = 0,
  rolling_def_sr = 0.5
)
probs <- predict(ep_model,
                 xgboost::xgb.DMatrix(as.matrix(test_cases[, feature_cols])),
                 reshape = TRUE)
test_cases$ep <- as.numeric(probs %*% point_values)
cat("\nEP sanity check (vs league avg defense):\n")
print(test_cases[, c("yardline_100", "ep")])