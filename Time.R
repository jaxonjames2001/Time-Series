library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(recipes)
library(embed)
library(lme4)
library(kknn)
library(forecast)
library(patchwork)

test <- vroom("test.csv")
train <- vroom("train.csv")

storeItem <- train %>%
filter(store==1, item==1)

# storeItem2 <- train %>%
#   filter(store==2, item==2)
# 
# graph1 <- storeItem %>%
# ggplot(mapping=aes(x=date, y=sales)) +
# geom_line() +
# geom_smooth(se=FALSE)
# 
# graph2 <- storeItem2 %>%
#   ggplot(mapping=aes(x=date, y=sales)) +
#   geom_line() +
#   geom_smooth(se=FALSE)
# 
# graph3 <- storeItem %>%
# pull(sales) %>%
# forecast::ggAcf(.)
# 
# graph4 <- storeItem2 %>%
#   pull(sales) %>%
#   forecast::ggAcf(.)
# 
# graph5 <- storeItem %>%
# pull(sales) %>%
# forecast::ggAcf(., lag.max=2*365)
# 
# graph6 <- storeItem2 %>%
#   pull(sales) %>%
#   forecast::ggAcf(., lag.max=2*365)
# 
# (graph1 + graph3 + graph5) / (graph2 + graph4 + graph6)

recipe <- recipe(sales ~ ., data = storeItem) %>%
  step_date(date,features="month") %>%
  step_date(date,features="year") %>%
  step_date(date,features="dow") %>%
  step_date(date,features="decimal") %>%
  step_mutate_at('date_month','date_year','date_dow',fn = factor) 

tree_mod <- rand_forest(mtry = tune(), 
                        min_n = tune(), 
                        trees = tune()) %>%
  set_engine("ranger") %>%
  set_mode("regression") 

# Create workflow
tree_wf <- workflow() %>%
  add_recipe(recipe) %>%  # Ensure the recipe is properly defined
  add_model(tree_mod)

# Set up grid of tuning values
tuning_params <- grid_regular(
  mtry(range = c(1, 10)),  # Specify range for mtry
  min_n(range = c(2, 10)),  # Specify range for min_n
  trees(range = c(100, 1000)),  # Specify range for trees
  levels = 5  # Adjust levels as needed
)

# Set up k-fold cross-validation
folds <- vfold_cv(storeItem, v = 5, repeats = 1)

# Perform tuning
CV_results <- tree_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_params,
    metrics = metric_set(smape)  # RMSE is appropriate for regression
  )

# Find best tuning parameters based on RMSE
bestTune <- CV_results %>%
  show_best(metric = "smape")

# Finalize workflow with the best tuning parameters and fit the model
final_tree_wf <- tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = storeItem)  # Fit on the training data


RAND_predictions <- final_tree_wf %>%
  predict(new_data = test, type = "numeric")

submission <- RAND_predictions %>%
  bind_cols(., test) %>%
  select(Id, .pred) %>%
  rename(Prediction = .pred)

vroom_write(x=submission, file="./RANDPreds.csv", delim=",")

