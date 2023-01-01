######################
library(e1071)
library(caTools)
library(tidyverse)
library(mlbench)
library(caret)
library(datasets)
library(caTools)
library(party)
library(dplyr)
library(magrittr)
library(caTools)
library(ROCR) 
library(rpart)
library(rpart.plot)
library(randomForest)
#######################

# 1. Reading the original data
world_cup_matches <- read.csv("world_cup_matches.csv")
international_matches <- read.csv("international_matches.csv")
world_cups <- read.csv("world_cups.csv")
ranking <- read.csv("2022_world_cup_groups.csv")
qatar_matches <- read.csv("2022_world_cup_matches.csv")

############################
# 2. Create new data frames
# 2.1. Create data frame named "matches" concatenating world cups and international matches
# Creating new data frames and concatenating data, We select the most relevant Columns
columns = c('Date', 'Home.Team', 'Home.Goals', 'Away.Goals','Away.Team')
wc_df = world_cup_matches[,columns]
wc <- 1
wc_df = cbind(wc_df,wc)
wc <- 0
im_df = international_matches[,columns]
im_df = cbind(im_df,wc)
matches = rbind(im_df, wc_df)

# Cleaning Data and formatting it
matches <-  tibble::rowid_to_column(matches, "ID")
matches$Date <- gsub('/','-',matches$Date)
matches$Date<- as.Date(matches$Date)
Year<- as.numeric(format(matches$Date,'%Y'))
matches <-cbind(matches,Year)
matches <- matches[matches$Year>=1930,]
columns = c('Year', 'Home.Team', 'Home.Goals', 'Away.Goals','Away.Team','wc')
matches <- matches[,columns]
matches <-  tibble::rowid_to_column(matches, "ID")

status <- ""
matches<-cbind(matches,status)
# Home team wins
matches$status[matches$Home.Goals - matches$Away.Goals >0] <- 1
# Home and Away Tie
matches$status[matches$Home.Goals - matches$Away.Goals ==0] <- 2
# Away Team wins
matches$status[matches$Home.Goals - matches$Away.Goals <0] <- 3

# 2.2. Create data frames "countries": Teams attribute table
# Once we have the countries table completed,
# We'll build the data set that has the matches

Homeim <- str_sort(unique(international_matches[,'Home.Team']))
Awayim <-  str_sort(unique(international_matches[,'Away.Team']))
countries_im <- str_sort(unique(union(Homeim,Awayim)))

Homewm <-  str_sort(unique(world_cup_matches[,'Home.Team']))
Awaywm <-  str_sort(unique(world_cup_matches[,'Away.Team']))
countries_wm <- str_sort(unique(union(Homewm,Awaywm)))

countries <-data.frame(Name = str_sort(unique(union(countries_im,countries_wm))))

# International matches PLAYED,WON,LOSS,TIE
inter_match <- 0
countries <- cbind(countries,inter_match)
inter_won <- 0
countries <- cbind(countries,inter_won)
inter_loss <- 0
countries <- cbind(countries,inter_loss)
inter_tie <- 0
countries <- cbind(countries,inter_tie)
world_match <- 0

# WorldCup matches PLAYED,WON,LOSS,TIE
countries <- cbind(countries,world_match)
world_won <- 0
countries <- cbind(countries,world_won)
world_loss <- 0
countries <- cbind(countries,world_loss)
world_tie <- 0
countries <- cbind(countries,world_tie)

# Conceded and Scored Goals International
inter_scored_goals <- 0
countries <- cbind(countries,inter_scored_goals)
inter_conc_goals <- 0
countries <- cbind(countries,inter_conc_goals)

# Conceded and Scored Goals World
world_scored_goals <- 0
countries <- cbind(countries,world_scored_goals)
world_conc_goals <- 0
countries <- cbind(countries,world_conc_goals)

for (i in 1:nrow(countries)) {
  #International
  # Total amount of matches
  x <- nrow(international_matches[international_matches$Home.Team == countries[i,1],])
  x_a <- nrow(international_matches[international_matches$Away.Team == countries[i,1],])
  countries[c(i),c(2)] <- x +x_a
  
  # Amount Of Wins for each country
  z = nrow(international_matches[international_matches$Home.Team == countries[i,1] & international_matches$Home.Goals>international_matches$Away.Goals,])
  z_a = nrow(international_matches[international_matches$Away.Team == countries[i,1] & international_matches$Away.Goals>international_matches$Home.Goals,])
  countries[c(i),c(3)] <- z + z_a

  # Amount Of Loss for each country
  y = nrow(international_matches[international_matches$Home.Team == countries[i,1] & international_matches$Home.Goals<international_matches$Away.Goals,])
  y_a = nrow(international_matches[international_matches$Away.Team == countries[i,1] & international_matches$Away.Goals<international_matches$Home.Goals,])
  countries[c(i),c(4)] <- y + y_a
  
  # Amount Of Ties for each country
  t = nrow(international_matches[international_matches$Home.Team == countries[i,1] & international_matches$Home.Goals==international_matches$Away.Goals,])
  t_a = nrow(international_matches[international_matches$Away.Team == countries[i,1] & international_matches$Away.Goals==international_matches$Home.Goals,])
  countries[c(i),c(5)] <- t + t_a
  
  #World Matches
  # Total amount of matches
  m <- nrow(world_cup_matches[world_cup_matches$Home.Team == countries[i,1],])
  m_a <- nrow(world_cup_matches[world_cup_matches$Away.Team == countries[i,1],])
  countries[c(i),c(6)] <- m +m_a
  
  # Amount Of Wins for each country
  w = nrow(world_cup_matches[world_cup_matches$Home.Team == countries[i,1] & world_cup_matches$Home.Goals>world_cup_matches$Away.Goals,])
  w_a = nrow(world_cup_matches[world_cup_matches$Away.Team == countries[i,1] & world_cup_matches$Away.Goals>world_cup_matches$Home.Goals,])
  countries[c(i),c(7)] <- w + w_a
  
  # Amount Of Loss for each country
  l = nrow(world_cup_matches[world_cup_matches$Home.Team == countries[i,1] & world_cup_matches$Home.Goals<world_cup_matches$Away.Goals,])
  l_a = nrow(world_cup_matches[world_cup_matches$Away.Team == countries[i,1] & world_cup_matches$Away.Goals<world_cup_matches$Home.Goals,])
  countries[c(i),c(8)] <- l + l_a
  
  # Amount Of Ties for each country
  tt = nrow(world_cup_matches[world_cup_matches$Home.Team == countries[i,1] & world_cup_matches$Home.Goals==world_cup_matches$Away.Goals,])
  tt_a = nrow(world_cup_matches[world_cup_matches$Away.Team == countries[i,1] & world_cup_matches$Away.Goals==world_cup_matches$Home.Goals,])
  countries[c(i),c(9)] <- tt + tt_a
  
  # Goals Scored international away and home
  scored_goals_int_h = sum(international_matches[international_matches$Home.Team == countries[i,1] ,]$Home.Goals)
  scored_goals_int_a = sum(international_matches[international_matches$Away.Team == countries[i,1] ,]$Away.Goals)
  countries[c(i),c(10)] <-  scored_goals_int_h + scored_goals_int_a
  
  # Goals Conceded international away and home
  conceded_goals_int_h = sum(international_matches[international_matches$Home.Team == countries[i,1] ,]$Away.Goals)
  conceded_goals_int_a = sum(international_matches[international_matches$Away.Team == countries[i,1] ,]$Home.Goals)
  countries[c(i),c(11)] <- conceded_goals_int_h + conceded_goals_int_a
  
  # Goals Scored international away and home
  scored_goals_wc_h = sum(world_cup_matches[world_cup_matches$Home.Team == countries[i,1] ,]$Home.Goals)
  scored_goals_wc_a = sum(world_cup_matches[world_cup_matches$Away.Team == countries[i,1] ,]$Away.Goals)
  countries[c(i),c(12)] <-  scored_goals_wc_h + scored_goals_wc_a
  
  # Goals Conceded international away and home
  conceded_goals_wc_h = sum(world_cup_matches[world_cup_matches$Home.Team == countries[i,1] ,]$Away.Goals)
  conceded_goals_wc_a = sum(world_cup_matches[world_cup_matches$Away.Team == countries[i,1] ,]$Home.Goals)
  countries[c(i),c(13)] <- conceded_goals_wc_h + conceded_goals_wc_a
}
# Differences of matches, their outcome international 
dif_in = 0
matches <- cbind(matches,dif_in)
dif_in_w = 0
matches <- cbind(matches,dif_in_w)
dif_in_l = 0
matches <- cbind(matches,dif_in_l)
dif_in_t = 0
matches <- cbind(matches,dif_in_t)


# Differences of matches, their outcome world cup
dif_wc = 0
matches <- cbind(matches,dif_wc)
dif_wc_w = 0
matches <- cbind(matches,dif_wc_w)
dif_wc_l = 0
matches <- cbind(matches,dif_wc_l)
dif_wc_t = 0
matches <- cbind(matches,dif_wc_t)

dif_in_sco = 0
matches <- cbind(matches,dif_in_sco)
dif_in_con = 0
matches <- cbind(matches,dif_in_con)
dif_wc_sco = 0
matches <- cbind(matches,dif_wc_sco)
dif_wc_con = 0
matches <- cbind(matches,dif_wc_con)


for (i in 1:nrow(matches)) {
  matches[c(i),c(9)] = countries[countries$Name == matches[c(i),c(3)] ,c("inter_match")] - countries[countries$Name == matches[c(i),c(6)] ,c("inter_match")]
  matches[c(i),c(10)] = countries[countries$Name == matches[c(i),c(3)] ,c("inter_won")] - countries[countries$Name == matches[c(i),c(6)] ,c("inter_won")]
  matches[c(i),c(11)] = countries[countries$Name == matches[c(i),c(3)] ,c("inter_loss")] - countries[countries$Name == matches[c(i),c(6)] ,c("inter_loss")]
  matches[c(i),c(12)] = countries[countries$Name == matches[c(i),c(3)] ,c("inter_tie")] - countries[countries$Name == matches[c(i),c(6)] ,c("inter_tie")]
  matches[c(i),c(13)] = countries[countries$Name == matches[c(i),c(3)] ,c("world_match")] - countries[countries$Name == matches[c(i),c(6)] ,c("world_match")]
  matches[c(i),c(14)] = countries[countries$Name == matches[c(i),c(3)] ,c("world_won")] - countries[countries$Name == matches[c(i),c(6)] ,c("world_won")]
  matches[c(i),c(15)] = countries[countries$Name == matches[c(i),c(3)] ,c("world_loss")] - countries[countries$Name == matches[c(i),c(6)] ,c("world_loss")]
  matches[c(i),c(16)] = countries[countries$Name == matches[c(i),c(3)] ,c("world_tie")] - countries[countries$Name == matches[c(i),c(6)] ,c("world_tie")]
  matches[c(i),c(17)] = countries[countries$Name == matches[c(i),c(3)] ,c("inter_scored_goals")] - countries[countries$Name == matches[c(i),c(6)] ,c("inter_scored_goals")]
  matches[c(i),c(18)] = countries[countries$Name == matches[c(i),c(3)] ,c("inter_conc_goals")] - countries[countries$Name == matches[c(i),c(6)] ,c("inter_conc_goals")]
  matches[c(i),c(19)] = countries[countries$Name == matches[c(i),c(3)] ,c("world_scored_goals")] - countries[countries$Name == matches[c(i),c(6)] ,c("world_scored_goals")]
  matches[c(i),c(20)] = countries[countries$Name == matches[c(i),c(3)] ,c("world_conc_goals")] - countries[countries$Name == matches[c(i),c(6)] ,c("world_conc_goals")]
}
columns = c('wc',
            'dif_in', 'dif_in_w', 'dif_in_l',
            'dif_in_t', 'dif_in_sco', 'dif_in_con',
            'dif_wc', 'dif_wc_w', 'dif_wc_l',
            'dif_wc_t', 'dif_wc_sco', 'dif_wc_con','status')
matches = matches[,columns]
matches$status<-as.factor(matches$status) 

###########################################################
#Confusion Matrix displaying the Gaussian NAIVE BAYES MODEL
set.seed(579642)  #Set the seed for reproducibility
# Fitting Naive Bayes Model
# to training dataset
dt_acc<- 0
## 10 fold Cross validation
for (i in 1:10) {
  split <- sample.split(matches, SplitRatio = 0.7)
  train_cl <- subset(matches, split == "TRUE")
  test_cl <- subset(matches, split == "FALSE")
  # Feature Scaling
  train_scale <- scale(train_cl[, 1:13])
  test_scale <- scale(test_cl[, 1:13])
  NB_model <- naiveBayes(status ~ ., data = train_cl)
  NB_model
  NB_pred <- predict(NB_model, newdata = test_cl)
  cm <- table(test_cl$status, NB_pred)
  dt_acc <- c(dt_acc, sum(diag(cm)) / sum(cm))
}

plot(1-dt_acc, type="l", ylab="Error Rate", xlab="Iterations", main="Error Rate for Matches With Different Subsets of Data for GNB")

# Confusion Matrix
cm

# Model Evaluation
confusionMatrix(cm)

#######################
# KNN Model
library(class)    # Contains the "knn" function
set.seed(579642)  #Set the seed for reproducibility


matches_acc<-0
dt_acc<- 0
## 10 fold Cross validation
for (i in 1:10) {
  #Create partitions in the matches data set (70% for training, 30% for testing/evaluation)
  matches_sample <- sample(1:nrow(matches), size=nrow(matches)*0.7)
  matches_train <- matches[matches_sample, ] #Select the 70% of rows
  matches_test <- matches[-matches_sample, ] #Select the 30% of rows
  #Taking K as 48 from previous test as it has the highest accuracy
  KNN_model <- knn(train=matches_train[,-14], test=matches_test[,-14], cl=matches_train$status, k=48)
  matches_acc <- c(matches_acc, mean(KNN_model==matches_test$status))
}
plot(1-matches_acc, type="l", ylab="Error Rate", xlab="Iterations", main="Error Rate for Matches With Different Subsets of Data for KNN")

confusion_mtx = table(matches_test[, 14], KNN_model)
# Model Evaluation
confusionMatrix(confusion_mtx)
###########################
#Logistic Regression Model 
library(nnet)
set.seed(579642)  #Set the seed for reproducibility

for (i in 1:10) {
  #Create partitions in the matches data set (70% for training, 30% for testing/evaluation)
  matches_sample <- sample(1:nrow(matches), size=nrow(matches)*0.7)
  matches_train <- matches[matches_sample, ] #Select the 70% of rows
  matches_test <- matches[-matches_sample, ] #Select the 30% of rows
  #Taking K as 48 from previous test as it has the highest accuracy
  matches_train$status<-relevel(matches_train$status,ref=1)
  LR_model <- multinom(status~.,data = matches_train)
  # Two-tail z-test and p-value
  z<-summary(LR_model)$coefficients/summary(LR_model)$standard.errors
  p<-(1-pnorm(abs(z),0,1))*2
  p
  # if less than 0.05 drop the feature, No  need to redo the model
  prediction <- predict(LR_model,matches_train)
  tab<- table(prediction,matches_train$status)
  # classiffication rate
  sum(diag(tab))/sum(tab)
  # misclassification rate
  1-sum(diag(tab))/sum(tab)
  
  #Test data
  p1<-predict(LR_model,newdata = matches_test)
  tab1<- table(matches_test$status,p1)
  
  # classiffication rate
  sum(diag(tab1))/sum(tab1)
  # misclassification rate
  1-sum(diag(tab1))/sum(tab1)
}

confusionMatrix(factor(matches_test$status,levels = 1:3),factor(p1,levels = 1:3))

###############
#Decision Tree
for (i in 1:10) {
  #Create partitions in the matches data set (70% for training, 30% for testing/evaluation)
  matches_sample <- sample(1:nrow(matches), size=nrow(matches)*0.7)
  matches_train <- matches[matches_sample, ] #Select the 70% of rows
  matches_test <- matches[-matches_sample, ] #Select the 30% of rows
  DT_model <- rpart(status~., data = matches_train, method = 'class')
  DT_predict <-predict(DT_model, matches_test, type = 'class')
}

rpart.plot(DT_model, extra = 104)

table_mat <- table(matches_test$status, DT_predict)
table_mat

accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy for test', accuracy_Test))
# Model Evaluation
confusion_mtx = table(matches_test[, 14], DT_predict)
confusion_mtx
# Model Evaluation
confusionMatrix(confusion_mtx)
################
#Random forest
# Fitting Random Forest to the train dataset
set.seed(579642)  #Set the seed for reproducibility
for (i in 1:3) {
  #Create partitions in the matches data set (70% for training, 30% for testing/evaluation)
  matches_sample <- sample(1:nrow(matches), size=nrow(matches)*0.7)
  matches_train <- matches[matches_sample, ] #Select the 70% of rows
  matches_test <- matches[-matches_sample, ] #Select the 30% of rows
  RF_model = randomForest(x = matches_train[-14],
                          y = matches_train$status,
                          ntree = 500)
  # Predicting the Test set results
  RF_pred = predict(RF_model, newdata = matches_test[-14])
}


# Confusion Matrix
confusion_mtx = table(matches_test[, 14], RF_pred)
confusion_mtx
# Model Evaluation
confusionMatrix(confusion_mtx)
# Plotting model
plot(RF_model)

# Importance plot
importance(RF_model)

# Variable importance plot
varImpPlot(RF_model)

###########################
#Match prediction function:
match_predict <- function(tempdf,model){
  
  # Differences of matches, their outcome international 
  dif_in = 0
  tempdf <- cbind(tempdf,dif_in)
  dif_in_w = 0
  tempdf <- cbind(tempdf,dif_in_w)
  dif_in_l = 0
  tempdf <- cbind(tempdf,dif_in_l)
  dif_in_t = 0
  tempdf <- cbind(tempdf,dif_in_t)
  
  
  # Differences of matches, their outcome world cup
  dif_wc = 0
  tempdf <- cbind(tempdf,dif_wc)
  dif_wc_w = 0
  tempdf <- cbind(tempdf,dif_wc_w)
  dif_wc_l = 0
  tempdf <- cbind(tempdf,dif_wc_l)
  dif_wc_t = 0
  tempdf <- cbind(tempdf,dif_wc_t)
  
  dif_in_sco = 0
  tempdf <- cbind(tempdf,dif_in_sco)
  dif_in_con = 0
  tempdf <- cbind(tempdf,dif_in_con)
  dif_wc_sco = 0
  tempdf <- cbind(tempdf,dif_wc_sco)
  dif_wc_con = 0
  tempdf <- cbind(tempdf,dif_wc_con)
  wc <-1
  tempdf <- cbind(tempdf,wc)
  for (i in 1:nrow(tempdf)) {
    tempdf[c(i),c(4)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("inter_match")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("inter_match")]
    tempdf[c(i),c(5)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("inter_won")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("inter_won")]
    tempdf[c(i),c(6)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("inter_loss")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("inter_loss")]
    tempdf[c(i),c(7)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("inter_tie")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("inter_tie")]
    tempdf[c(i),c(8)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("world_match")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("world_match")]
    tempdf[c(i),c(9)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("world_won")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("world_won")]
    tempdf[c(i),c(10)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("world_loss")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("world_loss")]
    tempdf[c(i),c(11)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("world_tie")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("world_tie")]
    tempdf[c(i),c(12)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("inter_scored_goals")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("inter_scored_goals")]
    tempdf[c(i),c(13)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("inter_conc_goals")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("inter_conc_goals")]
    tempdf[c(i),c(14)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("world_scored_goals")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("world_scored_goals")]
    tempdf[c(i),c(15)] = countries[countries$Name == tempdf[c(i),c(2)] ,c("world_conc_goals")] - countries[countries$Name == tempdf[c(i),c(3)] ,c("world_conc_goals")]
  }
  xy = c('wc',
              'dif_in', 'dif_in_w', 'dif_in_l',
              'dif_in_t', 'dif_in_sco', 'dif_in_con',
              'dif_wc', 'dif_wc_w', 'dif_wc_l',
              'dif_wc_t', 'dif_wc_sco', 'dif_wc_con')
  tempdf = tempdf[,xy]
  prediction_result<-predict(model,newdata=tempdf)
  
  return(prediction_result)
}
###############
##PREDICTION CAN BE DONE WITH ANY MODEL HERE 
#Group Stages 
stages=qatar_matches[qatar_matches$'Stage'=='Group stage',]
columns = c("Stage","Home.Team","Away.Team")
stages=stages[,columns]
stat = match_predict(stages,RF_model)

stages<-cbind(stages,stat)
group<-"NA"
stages<-cbind(stages,group)
home_pred<-"NA"
stages<-cbind(stages,home_pred)
away_pred<-"NA"
stages<-cbind(stages,away_pred)

group<-"Na"
team<-"Na"
pts<-0
fifarank<-"Na"
groups_table = data.frame(group=ranking$Group,team=ranking$Team,pts=pts,fifarank=ranking$FIFA.Ranking)
#Show group Stages results
for (i in 1:nrow(stages)) {
  temp = 0
  stages[c(i),c("group")] = ranking[ranking$Team ==  stages[c(i),c("Home.Team")],c("Group")]
  if(stages[c(i),c("stat") ] == 1){
    stages[c(i),c("home_pred") ] = "Win"
    groups_table[groups_table$team == stages[c(i),c("Home.Team")],c("pts")] = 
      groups_table[groups_table$team == stages[c(i),c("Home.Team")],c("pts")] +3 
    stages[c(i),c("away_pred") ] = "Loss"
  }
  if(stages[c(i),c("stat") ] == 2){
    stages[c(i),c("home_pred") ] = "Tie"
    groups_table[groups_table$team == stages[c(i),c("Home.Team")],c("pts")] = 
      groups_table[groups_table$team == stages[c(i),c("Home.Team")],c("pts")] +1 
    groups_table[groups_table$team == stages[c(i),c("Away.Team")],c("pts")] = 
      groups_table[groups_table$team == stages[c(i),c("Away.Team")],c("pts")] +1 
    stages[c(i),c("away_pred") ] = "Tie"    
  }
  if(stages[c(i),c("stat") ] == 3){
    stages[c(i),c("home_pred") ] = "Loss"
    groups_table[groups_table$team == stages[c(i),c("Away.Team")],c("pts")] = 
      groups_table[groups_table$team == stages[c(i),c("Away.Team")],c("pts")] +3 
    stages[c(i),c("away_pred") ] = "Win"
  }
}

Group<-"Na"
first<-"Na"
second<-"Na"
at<-"NA"
ht<-"NAS"
group_wins<-data.frame(Group=unique(ranking$Group),first=first,second=second,ht=ht,at=at)
data_new1 <- groups_table[order(groups_table$pts, decreasing = TRUE), ]  # Order data descending
data_new1 <- Reduce(rbind,by(data_new1, data_new1["group"],head,n = 2))

index=0
for (i in 1:nrow(data_new1)) {
  if(i%%2 == 1){
    index = (i+1)
    index = index/2
    group_wins[c(index),c('first')]= data_new1[c(i),c("team")]
    result = paste(1,data_new1[c(i),c("group")],sep="")
    group_wins[c(index),c('ht')]= result
  }
  if(i%%2 == 0){
    index = i/2
    group_wins[c(index),c('second')]= data_new1[c(i),c("team")] 
    result = paste(2,data_new1[c(i),c("group")],sep="")
    group_wins[c(index),c('at')]= result
  }
}

###############
#Round of 16
round_16=qatar_matches[qatar_matches$Stage=='Round of 16',]
columns = c("Stage","Home.Team","Away.Team")
round_16=round_16[,columns]

for (i in 1:nrow(round_16)) {
  h_t = group_wins[round_16[c(i),"Home.Team"] == group_wins$ht,]$first
  a_t = group_wins[round_16[c(i),"Away.Team"] == group_wins$at,]$second
  round_16[c(i),"Home.Team"] = h_t
  round_16[c(i),"Away.Team"] = a_t
}
## We need to eleminate the option of getting a Tie
matches= matches[matches$status!=2,]
matches$status = factor(matches$status)
split <- sample.split(matches, SplitRatio = 0.7)
train_cl <- subset(matches, split == "TRUE")
test_cl <- subset(matches, split == "FALSE")
set.seed(579642)  #Set the seed for reproducibility
for (i in 1:3) {
  #Create partitions in the matches data set (70% for training, 30% for testing/evaluation)
  matches_sample <- sample(1:nrow(matches), size=nrow(matches)*0.7)
  matches_train <- matches[matches_sample, ] #Select the 70% of rows
  matches_test <- matches[-matches_sample, ] #Select the 30% of rows
  RF_model = randomForest(x = matches_train[-14],
                          y = matches_train$status,
                          ntree = 500)
}
for (i in 1:10) {
  #Create partitions in the matches data set (70% for training, 30% for testing/evaluation)
  matches_sample <- sample(1:nrow(matches), size=nrow(matches)*0.7)
  matches_train <- matches[matches_sample, ] #Select the 70% of rows
  matches_test <- matches[-matches_sample, ] #Select the 30% of rows
  split <- sample.split(matches, SplitRatio = 0.7)
  train_cl <- subset(matches, split == "TRUE")
  test_cl <- subset(matches, split == "FALSE")
  DT_model <- rpart(status~., data = matches_train, method = 'class')
  LR_model <- multinom(status~.,data = matches_train)
  KNN_model <- knn(train=matches_train[,-14], test=matches_test[,-14], cl=matches_train$status, k=48)
  NB_model <- naiveBayes(status ~ ., data = train_cl)
}

round_16_stat = match_predict(round_16,RF_model)

round_16 = cbind(round_16,round_16_stat)
round_16_winner = "NA"
round_16 = cbind(round_16,round_16_winner)

##########
# QUARTER - FINAL
quarter_final=qatar_matches[qatar_matches$Stage=='Quarter-finals',]
columns = c("Stage","Home.Team","Away.Team")
quarter_final=quarter_final[,columns]
for (i in 1:nrow(round_16)) {
  x= round_16[c(i),"round_16_stat"]
  if(x==1){
    round_16[c(i),"round_16_winner"]= round_16[c(i),"Home.Team"]
  }
  else{
    round_16[c(i),"round_16_winner"]= round_16[c(i),"Away.Team"]
  }
  if(i%%2==1){
    index = (i+1)
    index = index/2
    quarter_final[c(index),"Home.Team"] = round_16[c(i),"round_16_winner"]
  }
  else{
    index = i/2
    quarter_final[c(index),"Away.Team"] = round_16[c(i),"round_16_winner"]
  }
}
#View Round 16 winners
#Predicting Quarter Final
quarter_final
quarter_final_stat = match_predict(quarter_final,RF_model)
quarter_final_stat
quarter_final = cbind(quarter_final,quarter_final_stat)
quarter_final_winner = "NA"
quarter_final = cbind(quarter_final,quarter_final_winner)

##########
# Semi - FINAL
semi_final=qatar_matches[qatar_matches$Stage=='Semi-finals',]
columns = c("Stage","Home.Team","Away.Team")
semi_final=semi_final[,columns]
for (i in 1:nrow(quarter_final)) {
  x= quarter_final[c(i),"quarter_final_stat"]
  if(x==1){
    quarter_final[c(i),"quarter_final_winner"]= quarter_final[c(i),"Home.Team"]
  }
  else{
    quarter_final[c(i),"quarter_final_winner"]= quarter_final[c(i),"Away.Team"]
  }
  if(i%%2==1){
    index = (i+1)
    index = index/2
    semi_final[c(index),"Home.Team"] = quarter_final[c(i),"quarter_final_winner"]
  }
  else{
    index = i/2
    semi_final[c(index),"Away.Team"] = quarter_final[c(i),"quarter_final_winner"]
  }
}
#View Quarter Final winners
quarter_final
#Predicting Quarter Final
semi_final
semi_final_stat = match_predict(semi_final,RF_model)

semi_final = cbind(semi_final,semi_final_stat)
semi_final_winner = "NA"
semi_final = cbind(semi_final,semi_final_winner)
semi_final_loser = "NA"
semi_final = cbind(semi_final,semi_final_loser)

#################################
# Third Place and Final - FINAL
finals=qatar_matches[qatar_matches$Stage=='Final',]
thirdplace = qatar_matches[qatar_matches$Stage=='Third place',]
columns = c("Stage","Home.Team","Away.Team")
finals=finals[,columns]
thirdplace=thirdplace[,columns]

for (i in 1:nrow(semi_final)) {
  x= semi_final[c(i),"semi_final_stat"]
  if(x==1){
    semi_final[c(i),"semi_final_winner"]= semi_final[c(i),"Home.Team"]
    semi_final[c(i),"semi_final_loser"]= semi_final[c(i),"Away.Team"]
  }
  else{
    semi_final[c(i),"semi_final_winner"]= semi_final[c(i),"Away.Team"]
    semi_final[c(i),"semi_final_loser"]= semi_final[c(i),"Home.Team"]
  }
  if(i%%2==1){
    index = (i+1)
    index = index/2
    finals[c(index),"Home.Team"] = semi_final[c(i),"semi_final_winner"]
    thirdplace[c(index),"Home.Team"] = semi_final[c(i),"semi_final_loser"]
  }
  else{
    index = i/2
    finals[c(index),"Away.Team"] = semi_final[c(i),"semi_final_winner"]
    thirdplace[c(index),"Away.Team"] = semi_final[c(i),"semi_final_loser"]
  }
}
#View Semi Final
semi_final

#Third Place Prediction
thirdplace_stat = match_predict(thirdplace,RF_model)

thirdplace = cbind(thirdplace,thirdplace_stat)
thirdplace_winner = "NA"
thirdplace = cbind(thirdplace,thirdplace_winner)
for (i in 1:nrow(thirdplace)) {
  x= thirdplace[c(i),"thirdplace_stat"]
  if(x==1){
    thirdplace[c(i),"thirdplace_winner"]= thirdplace[c(i),"Home.Team"]
  }
  else{
    thirdplace[c(i),"thirdplace_winner"]= thirdplace[c(i),"Away.Team"]
  }
}
#View Final
thirdplace

#Final Prediction
finals_stat = match_predict(finals,RF_model)
finals = cbind(finals,finals_stat)
finals_winner = "NA"
finals = cbind(finals,finals_winner)
for (i in 1:nrow(finals)) {
  x= finals[c(i),"finals_stat"]
  if(x==1){
    finals[c(i),"finals_winner"]= finals[c(i),"Home.Team"]
  }
  else{
    finals[c(i),"finals_winner"]= finals[c(i),"Away.Team"]
  }
}
#View Final
finals
