### Project Overview

 The data given is of cash deposits made in particular branches of a bank. The objective is to predict the cash deposit amount for branches for the year of 2016


### Learnings from the project

 I applied different data preprocessing techniques to treat the timestamp columns and  generated new features from existing timestamp columns. We tried label encoding and one hot encoding for converting the categorical varibles into numerical varibles. However , the categorical columns in train and validation set contained different values and hence one hot encoding led to mismatch of columns in both datasets. Hence, Linear regression would fail in this case. and its better to use tree based algorithms in such cases. 




### Approach taken to solve the problem

 I used a decision tree regressor as a model because they are able to operate on both continuous and categorical variables directly. I also tried applying XGB Regressor but that decreased the  accuracy and hence Decision Tree Regressor  with an accuracy of 0.93 would be the better model for the cash deposit prediction.


