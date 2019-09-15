# Document_Multi_Class_Classification
This is a typical Document Classification Problem whose sample pic is attached herewith
1. 3 Different Approaches were used to solve this problem
Steps follwed in Approach_1 were:
1) TF_IDF 
2) Logistic_Regression for training and prediction
Steps follwed in Approach_2 were:
1) TF_IDF 
b) SVD(Singular Value Decomposition) i.e a dimentionality reduction technique 
c) SVM linear for training and prediction
Steps follwed in Approach_3 were:
a) TF_IDF 
b) SVD(Singular Value Decomposition) 
c) Sorting based on mutual_info_score 
d) Builiding an accurate data structure for parameters used in binary classifiers(not included) 
e) Getting all binary classifiers ready 
f) Building a tree structure based predictor in ascending order[1,2,3,4,5,6,7,8,9] i.e (1,2) => (1,3) => (1,4) => (4,5) => (5,6) etc 
g) Builiding a tree structure based predictor using the above lines but node is picked based on accuracy score
These were applied on the data set choosen and can be easily modified for other datasets
