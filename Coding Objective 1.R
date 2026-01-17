pkgs <- c("tidyverse","tableone","janitor","skimr","caret","xgboost",
          "pROC","ROSE","SHAPforxgboost","data.table","scales")

installed <- pkgs %in% rownames(installed.packages())
if(any(!installed)) install.packages(pkgs[!installed])
lapply(pkgs, library, character.only = TRUE)

set.seed(123)

df_raw <- read.csv(file.choose(), stringsAsFactors = FALSE)
head(df_raw)
tail(df_raw)

#data pre-processing
df <- df_raw %>%
  mutate(across(c(brand1, brand2, brand3), ~na_if(as.character(.x), ""))) %>%
  
  #standardize age, gender, comorbidity, and outcome formats
  mutate(
    age = as.numeric(age),
    gender = as.character(gender),
    comorb = as.character(comorb),
    bid = as.character(bid)
  ) %>%
  
  #vaccination status construction
  mutate(
    dose1_flag = if_else(!is.na(brand1) & brand1 != "", 1L, 0L),
    dose2_flag = if_else(!is.na(brand2) & brand2 != "", 1L, 0L),
    dose3_flag = if_else(!is.na(brand3) & brand3 != "", 1L, 0L),
    doses_received = dose1_flag + dose2_flag + dose3_flag,
    vaccination_status = case_when(
      doses_received == 0 ~ "unvaccinated",
      doses_received == 1 ~ "partially",
      doses_received == 2 ~ "fully",
      doses_received >= 3 ~ "boosted",
      TRUE ~ NA_character_
    )
  ) %>%
  
  #recode variables into analysis-ready categories
  mutate(
    gender = case_when(
      gender %in% c("0", 0) ~ "female",
      gender %in% c("1", 1) ~ "male",
      gender %in% c("female","Female","F") ~ "female",
      gender %in% c("male","Male","M") ~ "male",
      TRUE ~ as.character(gender)
    ),
    comorb = case_when(
      comorb %in% c("0", 0, "no", "No", "FALSE") ~ "no_comorb",
      comorb %in% c("1", 1, "yes", "Yes", "TRUE") ~ "yes_comorb",
      TRUE ~ as.character(comorb)
    ),
    bid = case_when(
      bid %in% c("0", 0, "not_dead", "alive") ~ "not_dead",
      bid %in% c("1", 1, "dead") ~ "dead",
      TRUE ~ as.character(bid)
    )
  ) %>%
  
  #convert variables to factors
  mutate(
    gender = factor(gender, levels = c("female","male")),
    comorb = factor(comorb, levels = c("no_comorb","yes_comorb")),
    vaccination_status = factor(vaccination_status,
                                levels = c("unvaccinated","partially","fully","boosted")),
    bid = factor(bid, levels = c("not_dead","dead"))
  )

#quick data summary after preprocessing
cat("Number of observations after preprocessing:", nrow(df), "\n")
print(table(df$bid))
print(table(df$vaccination_status))

library(dplyr)
library(caret)
library(xgboost)
library(pROC)
library(PRROC)

#data preparation for modelling
mod_df <- df %>%
  select(age, comorb, vaccination_status, bid) %>%
  filter(!is.na(bid))

mod_df <- mod_df %>%
  mutate(
    label = if_else(bid == "dead", 1L, 0L),
    comorb = factor(comorb),
    vaccination_status = factor(vaccination_status)
  )

#check class imbalance
table(mod_df$label)
prop.table(table(mod_df$label))


#train-test split
set.seed(123)
train_idx <- createDataPartition(mod_df$label, p = 0.8, list = FALSE)
train <- mod_df[train_idx, ]
test  <- mod_df[-train_idx, ]

#calculate scale_pos_weight to handle class imbalance
neg <- sum(train$label == 0)
pos <- sum(train$label == 1)
scale_pos_weight <- neg / pos
cat("scale_pos_weight =", scale_pos_weight, "\n")

#create DMatrix for XGBoost
pred_vars <- c("age", "comorb", "vaccination_status")

make_dmatrix <- function(df_in, pred_vars) {
  mm <- model.matrix(~ . - 1, data = df_in[, pred_vars])
  xgb.DMatrix(data = mm, label = df_in$label)
}

dtrain <- make_dmatrix(train, pred_vars)
dtest  <- make_dmatrix(test, pred_vars)

# 7. XGBoost Hyperparameter Tuning
param_grid <- expand.grid(
  eta = c(0.05, 0.1),
  max_depth = c(4, 6),
  min_child_weight = c(1, 3),
  subsample = c(0.7, 0.8),
  colsample_bytree = c(0.7, 0.8)
)

best_auc <- 0
best_params <- list()

for (i in 1:nrow(param_grid)) {
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = param_grid$eta[i],
    max_depth = param_grid$max_depth[i],
    min_child_weight = param_grid$min_child_weight[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    scale_pos_weight = scale_pos_weight
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 1500,
    nfold = 5,
    early_stopping_rounds = 40,
    verbose = FALSE,
    maximize = TRUE
  )
  
  if (cv$evaluation_log$test_auc_mean[cv$best_iteration] > best_auc) {
    best_auc <- cv$evaluation_log$test_auc_mean[cv$best_iteration]
    best_params <- params
    best_nrounds <- cv$best_iteration
  }
}

cat("Best cross-validated AUC =", best_auc, "\n")
print(best_params)

#train final XGBoost model
bst <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain),
  verbose = 1
)

#model evaluation
pred_prob <- predict(bst, dtest)

pred_label_05 <- ifelse(pred_prob > 0.5, 1, 0)

conf_05 <- confusionMatrix(
  factor(pred_label_05),
  factor(test$label),
  positive = "1"
)
print(conf_05)

roc_obj <- roc(test$label, pred_prob)
auc_val <- auc(roc_obj)
cat("Test AUC:", round(as.numeric(auc_val), 4), "\n")


#improved XGBoost model

#data preparation
mod_df <- df %>%
  select(age, comorb, vaccination_status, bid) %>%
  filter(!is.na(bid))

mod_df <- mod_df %>%
  mutate(
    label = if_else(bid == "dead", 1L, 0L),
    comorb = factor(comorb),
    vaccination_status = factor(vaccination_status)
  )

#check imbalance
table(mod_df$label)
prop.table(table(mod_df$label))  # Imbalance: 0 = 77.5%, 1 = 22.5%

#train-test split
set.seed(123)
train_idx <- createDataPartition(mod_df$label, p = 0.8, list = FALSE)
train <- mod_df[train_idx, ]
test  <- mod_df[-train_idx, ]

#recalculate scale_pos_weight
neg <- sum(train$label == 0)
pos <- sum(train$label == 1)
scale_pos_weight <- ifelse(pos == 0, 1, neg / pos)
cat("scale_pos_weight =", scale_pos_weight, "\n")

#optimized XGBoost training
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.03,              # Lower learning rate
  max_depth = 4,           # Moderate tree depth
  min_child_weight = 3,    # Reduce overfitting
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 1,               # Minimum loss reduction
  alpha = 0.1,             # L1 regularization
  lambda = 1,              # L2 regularization
  scale_pos_weight = scale_pos_weight
)

#cross-validation to determine optimal nrounds
set.seed(123)
cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 3000,
  nfold = 5,
  early_stopping_rounds = 50,
  verbose = 1,
  maximize = TRUE
)

best_nrounds <- cv$best_iteration
cat("Optimal number of boosting rounds =", best_nrounds, "\n")

#train final tuned model
bst_tuned <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain),
  verbose = 1
)

#evaluation of tned Model
pred_prob_tuned <- predict(bst_tuned, dtest)
pred_label_tuned <- ifelse(pred_prob_tuned > 0.5, 1, 0)

conf_tuned <- confusionMatrix(
  factor(pred_label_tuned),
  factor(test$label),
  positive = "1"
)
print(conf_tuned)

roc_obj_tuned <- pROC::roc(test$label, pred_prob_tuned)
auc_val_tuned <- pROC::auc(roc_obj_tuned)
cat("Test AUC after tuning:", round(as.numeric(auc_val_tuned), 4), "\n")


#feature importance & SHAP analysis

# Extract feature importance from tuned XGBoost model
importance <- xgb.importance(model = bst_tuned)

# Define selected features of interest
selected_features <- c("age",
                       "comorbyes_comorb",
                       "vaccination_statuspartially",
                       "vaccination_statusfully",
                       "vaccination_statusboosted")

filtered_importance <- importance %>%
  filter(Feature %in% selected_features)

cat("\nFiltered feature importance (gain):\n")
print(filtered_importance)

#plot feature importance
if (nrow(filtered_importance) > 0) {
  plot_data <- filtered_importance %>%
    arrange(Gain) %>%
    mutate(Feature = factor(Feature, levels = Feature))
  
  ggplot(plot_data, aes(x = Gain, y = Feature)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    labs(title = "Feature Importance (Gain) for Selected Variables") +
    theme_minimal()
} else {
  cat("No selected features found in the importance output.\n")
}

#SHAP value analysis
mm_train <- model.matrix(~ . -1, data = train[, pred_vars])

#compute SHAP values
shap_values <- shap.values(xgb_model = bst_tuned, X_train = mm_train)

#mean absolute SHAP scores for selected features
selected_shap_scores <- shap_values$mean_shap_score[
  names(shap_values$mean_shap_score) %in% selected_features
]

cat("\nMean absolute SHAP values for selected features:\n")
print(selected_shap_scores)

shap_long_raw <- shap.prep(shap_contrib = shap_values$shap_score,
                           X_train = mm_train)

shap_long <- shap_long_raw %>%
  filter(variable %in% selected_features)

#plot SHAP summary for selected features
if (nrow(shap_long) > 0) {
  shap.plot.summary(shap_long)
} else {
  cat("No selected features available for SHAP plotting.\n")
}
