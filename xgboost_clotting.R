library(data.table)
library(tidyverse)
library(stringr)
library(rlang)
library(xgboost)
library(caret)
library(pROC)
library(ROSE)
library(rpart)
# reset na.action to default
options(na.action='na.omit')
source('~/my_git_hub/GSA_analysis/GSA_code/PRS_clotting_rev/14_make_background_rev2.R')


## select necessary column

final1 <- final %>% 
  dplyr::select(clotting,Gender.x,Age.at.IBD.Dx,duration_follow,
         low) 

final1[final1 == "missing"] <- NA

final1 <- final1 %>% 
  drop_na(clotting)

final2 <- final1 %>% 
  mutate(Gender.x = ifelse(Gender.x == "M",1,0))
options(na.action='na.pass')





final_case <- final2 %>% 
  filter(clotting == 1)

final_control <- final2 %>% 
  filter(clotting == 0)

## shuffle data

final_case <- final_case[sample(1:nrow(final_case)),]
final_control <- final_control[sample(1:nrow(final_control)),]


# split into train and controls

train_case = final_case[1:round(nrow(final_case)*0.5),]
train_control = final_control[1:round(nrow(final_control)*0.5),]

train_data_merge = rbind(train_case,train_control)

#### do ROSE to encounter impbalanced case and controls
data.rose <- ovun.sample(clotting ~ ., data = train_data_merge, method = "over",N = 652)$data
train_data <- data.rose %>% 
  select(-clotting) %>% 
  data.matrix()

train_label <- data.rose %>% 
  pull(clotting)

test_case = final_case[-(1:round(nrow(final_case)*0.5)),]
test_control = final_control[-(1:round(nrow(final_control)*0.5)),]
test_data_merge = rbind(test_case,test_control)

test_data <- test_data_merge %>% 
  select(-clotting) %>% 
  data.matrix()
test_label <- test_data_merge %>% 
  pull(clotting)

## make xgboost matrix
## make data of train and test for xgboost
dtrain <- xgb.DMatrix(data = train_data, label= train_label)

dtest <- xgb.DMatrix(data = test_data, label= test_label)


# confirm colnames are identical, if it is not identical, predict will not work
identical(colnames(dtrain),colnames(dtest))



xgbcv <- xgb.cv(  
        data = dtrain, nrounds = 100, nfold = 5, 
        showsd = T, stratified = T, print_every_n  = 10, 
        early_stop_round = 20, maximize = F)

round_num = which(xgbcv$evaluation_log$test_rmse_mean == (min(xgbcv$evaluation_log$test_rmse_mean)))
positive_cases <- sum(train_label == TRUE)
negative_cases <- sum(train_label == FALSE)
## make model using training data
model <- xgboost(data = dtrain,
                 max.depth = 3,# the data   
                 nround = round_num,# max number of boosting iterations
                 objective = "binary:logistic",
                 print_every_n = 10,
                 gamma = 0)  # t
## generate predictions for ourtesting data
pred <- predict(model, dtest)
err <- mean(as.numeric(pred > 0.5) != test_label) # if pred is more than 0.5, that means case.
print(paste("test-error=", err))


xgbpred <- ifelse(pred > 0.5,TRUE,FALSE)
roc_test <- roc(as.numeric(test_label), as.numeric(xgbpred), algorithm = 2)
roc_test <- roc(as.numeric(xgbpred), as.numeric(test_label), algorithm = 2)
plot(roc_test ) 
auc(roc_test )
confusionMatrix(as.factor(as.numeric(xgbpred)), as.factor(as.numeric(test_label)))
table(as.numeric(xgbpred),as.numeric(test_label) )

## make feature importance
importance_matrix <- xgb.importance(names(test_data), model = model)
xgb.plot.importance(importance_matrix)

table(as.numeric(pred > 0.5),test_label)

fisher.test(table(train_data_merge$low,train_data_merge$clotting))

fisher.test(table(train_data_merge$PRS,train_data_merge$clotting))
