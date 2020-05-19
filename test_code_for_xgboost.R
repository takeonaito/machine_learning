# library(mlr)
library(data.table)
library(tidyverse)
library(stringr)
library(rlang)
library(xgboost)
library(caret)
library(pROC) 

## part2
setcol <- c("age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "target")

# read train and test data
train <- read.table("/mnt/share6/FOR_Takeo/temporary/adult.data", header = F, sep = ",", col.names = setcol, na.strings = c(" ?"), stringsAsFactors = F)
test <- read.table("/mnt/share6/FOR_Takeo/temporary/adult.test",header = F,sep = ",",col.names = setcol,skip = 1, na.strings = c(" ?"),stringsAsFactors = F)
setDT(train) 
setDT(test)

# make group column so that i can identify whether train or test data
train <- train %>% 
  mutate(group = "train")
test <- test %>% 
  mutate(group = "test")

# merge test and train
gattai <- rbind(train,test)
gattai <- gattai %>% 
  mutate(target = str_remove_all(target,"\\.")) # remove "." in target column

# get names of character columns
char_col <- gattai %>% 
  select_if(is.character) %>% 
  colnames()
for(i in char_col) set(gattai,j=i,value = str_trim(gattai[[i]],side = "left")) # remove space from character columns

# remove target and group columuns from data for making matrix of features
gattai_targetremoved <- gattai %>% 
  select(-target,group)

# make labels of target. in this case i will predict whether income is more than 50K or not
tr_labels <- gattai %>% 
  select(target) %>% 
  mutate(target= ifelse(target == "<=50K",FALSE,TRUE)) %>%
  pull(target)

# make labels of group so that i can split train and test
gr_labels <- gattai %>% 
  pull(group)


# extract only numeric columns from gattai_targetremoved
gattai_numeric <- gattai_targetremoved %>% 
  select_if(is.numeric)

# extract only character columns from gattai_targetremoved
gattai_character <- gattai_targetremoved %>% 
  select_if(is.character)

## set option --> dont drop na value
options(na.action='na.pass')

# make sperse matrix for character columns

region1 <- model.matrix(~.-1,gattai_character)

# merge numeric and character data 
gattai_numeric <- cbind(gattai_numeric,region1)
gattai_numeric <- data.matrix(gattai_numeric) # make the data table to matrix 


# split gattai_numeric into train and test
train_data <- gattai_numeric[gr_labels == "train",]
train_labels <- tr_labels[gr_labels == "train"]

test_data <- gattai_numeric[gr_labels == "test",]
test_labels <- tr_labels[gr_labels == "test"]


## make data of train and test for xgboost
dtrain <- xgb.DMatrix(data = train_data, label= train_labels)

dtest <- xgb.DMatrix(data = test_data, label= test_labels)

# confirm colnames are identical, if it is not identical, predict will not work
identical(colnames(dtrain),colnames(dtest))

# make positive case and negative case number. If the proportion of cases is skewed,
# you can adjust using these information
positive_cases = sum(train_labels == TRUE)
negative_cases = sum(train_labels == FALSE)

## make model using training data
model <- xgboost(data = dtrain,
                 max.depth = 5,# the data   
                 nround = 100,
                 early_stopping_rounds = 3,# max number of boosting iterations
                 objective = "binary:logistic",
                 print_every_n = 10,
                 gamma = 0.5)  # t

## generate predictions for ourtesting data
pred <- predict(model, dtest)
err <- mean(as.numeric(pred > 0.5) != test_labels) # if pred is more than 0.5, that means case.
print(paste("test-error=", err))

xgbpred <- ifelse(pred > 0.5,TRUE,FALSE)
xgb.plot.multi.trees(feature_names = names(gattai_numeric), 
                     model = model)
confusionMatrix(as.factor(xgbpred), as.factor(test_labels))

## make ROC curve
roc_test <- roc(as.numeric(test_labels), as.numeric(xgbpred), algorithm = 2)
plot(roc_test ) 
auc(roc_test )

## make feature importance
importance_matrix <- xgb.importance(names(gattai_numeric), model = model)
xgb.plot.importance(importance_matrix,top_n = 50)

# reset na.action to default
options(na.action='na.omit')
