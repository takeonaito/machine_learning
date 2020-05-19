library(data.table)
library(tidyverse)
library(stringr)
library(rlang)
library(xgboost)
library(caret)
library(pROC)
library(ROSE)
library(rpart)
library(mlr)

# reset na.action to default
options(na.action='na.omit')
source('~/my_git_hub/GSA_analysis/GSA_code/PRS_clotting_rev/14_make_background_rev2.R')


## select necessary column

final1 <- final %>% 
  dplyr::select(clotting,Gender.x,Age.at.IBD.Dx,duration_follow,
                low) 

## input NA to "missing"
final1[final1 == "missing"] <- NA

## only use subjects whose clotting information is available
final1 <- final1 %>% 
  drop_na(clotting)


# convert character to factor
final1 <- final1 %>%
  mutate_at(
    .vars = vars("Gender.x","low","clotting"),
    .funs = funs(as.factor(.))
  )
summarizeColumns(final1)



# normalize numeric data 
combined1 <- normalizeFeatures(final1, target = "clotting")

# Convert factors to dummy variables by one-hot encoding
combined1 <- createDummyFeatures(
  combined1, target = "clotting",
  cols = c(
    "low",
    "Gender.x"
  )
)





# #### do ROSE to encounter impbalanced case and controls
# data.rose <- ovun.sample(clotting ~ ., data = final2, method = "over",N = 1304)$data

trainTask <- makeClassifTask(data = combined1, target = "clotting", positive = 1)

xgb_learner <- makeLearner(
  "classif.xgboost",
  predict.type = "response",
  par.vals = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    nrounds = 200
  ))

xgb_params <- makeParamSet(
  # The number of trees in the model (each one built sequentially)
  makeIntegerParam("nrounds", lower = 100, upper = 500),
  # number of splits in each tree
  makeIntegerParam("max_depth", lower = 1, upper = 10),
  # "shrinkage" - prevents overfitting
  makeNumericParam("eta", lower = .1, upper = .5),
  # L2 regularization - prevents overfitting
  makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x)
)

control <- makeTuneControlRandom(maxit = 50)
resample_desc <- makeResampleDesc("CV", iters = 3)
tuned_params <- tuneParams(
  learner = xgb_learner,
  task = trainTask,
  resampling = resample_desc,
  par.set = xgb_params,
  control = control
)


## make xgboost matrix
## make data of train and test for xgboost
train_data = combined1 %>% 
  select(-clotting) %>% 
  data.matrix()

train_label <- combined1 %>% 
  mutate(clotting == ifelse(clotting == 1,TRUE,FALSE)) %>% 
  pull(clotting)
dtrain <- xgb.DMatrix(data = train_data, label= as.numeric(train_label)-1)


## set best parameter from tuned_params 
my_para = tuned_params$x

# do CV 

xgbcv <- xgb.cv(data =  dtrain, 
                params = my_para,
                
                nrounds = 203,
                nfold = 3, 
                metrics = "auc", 
                verbose = TRUE, 
                "eval_metric" = "auc",
                "objective" = "binary:logistic",
                "colsample_bytree" = 1,
                print_every_n = 10, 
                "min_child_weight" = 1,
                booster = "gbtree",
                seed = 1234)

xgbcv <- xgb.cv(data =  dtrain, 
                nrounds = 100, 
                nfold = 3, 
                metrics = "auc", 
                verbose = TRUE, 
                "eval_metric" = "auc",
                "objective" = "binary:logistic", 
                "max.depth" = 6, 
                "colsample_bytree" = 1,
                print_every_n = 10, 
                "min_child_weight" = 1,
                booster = "gbtree",
                seed = 1234)
