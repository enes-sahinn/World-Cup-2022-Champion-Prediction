# World Cup 2022 Champion Prediction

## Table of Contents

* [Introduction](#introduction)
* [Dataset Pre-Procesing and Feature Exploration](#dataset-pre-procesing-and-feature-exploration)
* [Implementation Details of Machine Learning Models](#implementation-details-of-machine-learning-models)
  * [Gaussian Naive Bayes](#gaussian-naive-bayes)
  * [K-Nearest Neighbor (KNN)](#k-nearest-neighbor-knn)
  * [Logistic Regression](#logistic-regression)
  * [Decision Tree](#decision-tree)
  * [Random Forest](#random-forest)
* [Experimental Results](#experimental-results)
* [Conclusion](#conclusion)
* [Built With](#built-with)
* [Contact](#contact)

## Introduction

The World Cup is one of the most prestigious and highly anticipated international soccer tournaments, and the 2022 edition, which will be held in Qatar, promises to be no different. With the event fast approaching, many fans and analysts are eager to predict which team will emerge as the champion. In this project, it was aimed to build a machine learning model to predict the champion of the World Cup Qatar 2022 using R language.

The model was trained using the results of past world cup and international matches. Since there is more than one dataset, a pre-processing stage was made and the dataset was simplified. After the models were trained, one of them was selected as default and group stage matches, round of 16, quarter-final matches, semi-final matches and final match were predicted according to this model, respectively. The aim of this project is to provide a reliable prediction of the likely champion of the Qatar 2022 World Cup based on statistical analysis and machine learning techniques.

## Dataset Pre-Procesing and Feature Exploration
Before training the machine learning models, the dataset had to be pre-processed to make sure it was clean and ready for analysis. The World Cup is a long marathon lasting about 1 month consisting of many stages. The data preparation phase had to be done with care and attention, as it was really difficult to predict the champion. There were 5 datasets: "2022 world cup groups, 2022 world cup matches, international matches, world cup matches and world cups". And the past match results required to train the models were in the "international matches, world cup matches and world cups" datasets. These datasets should be simplified first and collected in a single dataset.

After reading the datasets required for the models, the matches before 1930 were deleted primarily with the thought that the prediction result would be more realistic. Some features such as "Year" have been formatted. International and world cup matches were concatenated and reduced to a single dataset called "matches". A feature called "status" has been added since there are 3 possibilities at the end of a football game. "1" indicates that the home team won, "2" indicated that the match ended in a draw, and "3" indicated that the home team lost.

A "countries" table was then created. And in this table, match statistics for each country's international and world cup are kept separately. These features are as follows: the number of matches played, the number of matches won, the number of matches he drew, the number of matches he lost, the number of goals scored and the number of goals conceded. Using the data in this table, we added additional features next to the matches in the "matches" dataset that will be used to train the model. These features were also kept separately for the world cup and international matches. And some of these features are as follows: "the difference in the number of games played, won, drawn, the difference in the number of goals scored and conceded". After all these features were prepared, the dataset was ready for training.

## Implementation Details of Machine Learning Models
In the project, 5 models were used, namely Gaussian Naive Bayes, K-Nearest Neighbor(KNN), Logistic Regression, Decision Tree and Random Forest. In each, 70% of the dataset was used for training and 30% for testing. And cross validation was applied in each of them.

### Gaussian Naive Bayes
Gaussian Naive Bayes was the first trained model. The best performance was measured using cross validation. And the error rate table was extracted. Also confusion matrix and certain statistics were calculated and printed.

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/GNB1.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/GNB2.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/GNB3.png)

### K-Nearest Neighbor (KNN)
K-Nearest Neighbor (KNN) was the second model trained. Necessary tests were applied to determine the K value and the highest accuracy value was obtained when "K=48". Next, the model was trained with cross validation. And the error rate table was extracted. Also confusion matrix and certain statistics were calculated and printed.

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/KNN1.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/KNN2.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/KNN3.png)

### Logistic Regression
The Logistic Regression Model was the third model to be trained. With cross validation, the model was trained. Also confusion matrix and certain statistics were calculated and printed.

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/LR1.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/LR2.png)

### Decision Tree
Decision Tree was the third model trained. 70% of the dataset was used for training and 30% for testing. The model was trained with cross validation. And the Decision Tree was written. In the model below, as mentioned earlier, "1" represents a win, "2" represents a draw, and "3" represents a loss. Finally, the confusion matrix and statistics were printed.

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/DT1.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/DT2.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/DT2.png)

### Random Forest
Random Forest was the last model trained and was also used as the default model in match prediction stages. First, the model was trained with cross validation. Then the model, confusion matrix and some statistics were printed.

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/RF1.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/RF2.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/RF3.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/RF4.png)

## Experimental Results
After training in 5 models, it was time to predict the "FIFA World Cup Qatar 2022" champion. First of all, a "match_prediction(data_frame, model)" function was created in order to use all models dynamically and Random Forest model was chosen as the default model. Then, the group stage matches, which is the first stage, started. There were 8 groups and 32 teams available. Since each team had to play 1 match against each other, the group stage matches consisted of a total of 48 matches. All of these matches were predicted and +3 points were added to the winning teams and +1 point to the tied teams. At the end of 48 matches, the 2 teams with the highest points in each group qualified for the round of 16.

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/table1.png)

After this stage, the matches consisted of one round and there was no possibility of a draw. Therefore, matches with this possibility were removed from the dataset, and 70% for training and 30% for testing were again reserved with the remaining dataset. After the round of 16 matches predicted by the default model, the teams that qualified for the quarter-finals were determined.

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/table2.png)

After that, the quarter-finals, semi-finals and final stages were continued in the same way. The winning teams have been determined. And the formed matches are predicted again until the champion is determined.

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/table3.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/table4.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/table5.png)

![alt text](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction/blob/master/Screnshots/table6.png)

## Conclusion
In conclusion, our machine learning model was able to accurately predict the outcomes of World Cup matches with a high degree of accuracy. Through the use of various algorithms and techniques, we were able to analyze and process large amounts of data in order to make informed predictions about the outcomes of matches.

Overall, the results of this project demonstrate the potential of machine learning in the realm of sports prediction. By harnessing the power of data and advanced computational techniques, we were able to gain valuable insights and make accurate predictions that would have been difficult to achieve through traditional methods.

While there is always room for improvement, the success of this project highlights the potential for machine learning to revolutionize the way we approach sports analysis and prediction. As such, it is likely that we will see an increasing adoption of these techniques in the future, as more and more organizations seek to harness the power of data to gain a competitive edge.

## Built With
* R

## Contact
Mail: enessah200@gmail.com\
LinkedIn: [linkedin.com/in/enes-sahinn](https://www.linkedin.com/in/enes-sahinn/)\
Project Link: [World-Cup-2022-Champion-Prediction](https://github.com/enes-sahinn/World-Cup-2022-Champion-Prediction)

