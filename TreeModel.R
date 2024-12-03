library(tidymodels)
library(modeltime)
library(timetk)
library(readr)
library(embed)
library(patchwork)
library(vroom)

train_data <- read_csv("train.csv")
test_data <- read_csv("test.csv")


# nStores <- 2
# nItems <- 2
nStores <- max(train_data$store)
nItems <- max(train_data$item)
for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train_data %>%
      filter(store==s, item==i)
    storeItemTest <- test_data %>%
      filter(store==s, item==i)
    
    recipe <- recipe(sales ~ date, data = train_data) %>%
      step_date(date, features=c("dow", "month", "year", "decimal")) %>%
      step_mutate(date_dow = factor(date_dow), 
                  date_month = factor(date_month), 
                  sinDecimal = sin(date_decimal)) %>%
      step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales))
    
    prophet_model <- prophet_reg() %>%
      set_engine(engine = "prophet")
    
    cv_split <- time_series_split(storeItemTrain, assess="3 months", cumulative=TRUE)
    cv_split %>%
      tk_time_series_cv_plan() %>% #Put into a data frame
      plot_time_series_cv_plan(date, sales, .interactive=FALSE)
    
    prophet_wf <- workflow() %>%
      add_recipe(recipe) %>%
      add_model(prophet_model) %>%
      fit(data=training(cv_split))
    
    cv_results <- modeltime_calibrate(prophet_wf,
                                      new_data=testing(cv_split))
    
    cv_results %>%
      modeltime_forecast(new_data = testing(cv_split),
                         actual_data = training(cv_split)) %>%
      plot_modeltime_forecast(.interactive=FALSE)
    
    fullfit <- cv_results %>%
      modeltime_refit(data = storeItemTrain)
    
    preds <- fullfit %>%
      modeltime_forecast(new_data = storeItemTest, actual_data = storeItemTrain) %>%
      filter(.key == "prediction") %>% 
      select(.value) %>%
      mutate(id = storeItemTest$id) %>%
      rename(sales = .value) %>%
      select(id, sales)
  
      
    
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
  }
}

vroom_write(all_preds, "submission.csv", delim = ",")
