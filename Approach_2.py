# -*- coding: utf-8 -*-
"""Approach-2 
@author: Karri Vivek Reddy

Info:
1) This code takes 6-7 minutes to run
2) Variables you may need would be 'label_Y_test' and 'test_proba' 
"""

"""----------------------Loading Datasets and required variables-------------------------"""
import pandas as pd
import numpy as np
import sklearn
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
label_Y = np.array(train_data['y'])

"""----------------------TF_IDF Extraction-------------------------"""
"""1) Apply Pre-processing on column(f3) which is most probably be documents.
   2) Next Step is to convert this into a TF_IDF matrix 
   3) feature_matrix is the TF_IDF representation of the documents   """ 
from sklearn.feature_extraction.text import TfidfVectorizer
New_f3 = [train_data['f3'],test_data['f3']];
Final_doc = pd.concat(New_f3)
del New_f3
TF_IDF = TfidfVectorizer()
feature_matrix = TF_IDF.fit_transform(Final_doc)
del Final_doc

"""----------------------Finding Dimension value-------------------------""" 
#Uncomment this only when you need to find the Value of 'K'  
#  Optimal Value of K found was '318'                    
"""1) Now the features of feature_matrix are to be reduced to certain value 'K'. 
   2) This is done by eigen vector decomposition  of C*CT where C is feature 
       matrix and CT is its transpose."""
"""    
import scipy
scipy.sparse.csr_matrix.transpose(feature_matrix)
CCT = feature_matrix*scipy.sparse.csr_matrix.transpose(feature_matrix)
Sum_CCT = 0
for i in range(0,CCT.shape[0]):
    Sum_CCT = Sum_CCT + CCT[i,i]
from  scipy.sparse.linalg import eigs
vals, vecs = eigs(CCT, k=1000)
Sum_VECS = 0
for K in range(0,1000):
    Sum_VECS = Sum_VECS + abs(vals[K])
    if Sum_VECS >= 0.8*Sum_CCT:
        break
""" 


"""----------------------SVD decoposition of TF_IDF matrix-------------------------""" 
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=750) # K-value is used and tried on different values
feature_f3 = svd.fit_transform(feature_matrix)


"""------------ Modification of 'f1' and 'f2' features extracted ----------"""
# 1) arrays of F1 and F2 are made by splitting them into individual characters i.e 'a234' is made '2','3','4' 
def Feature_matrices():
    array_f1 = np.zeros([2656,3])
    for  i in range(0,len(train_data)):
         list1 = list(train_data['f1'][i])
         del list1[0]
         length = len(list1)
         list1 = ['0']*(3-length)+list1
         for j in range(0,3):
             array_f1[i,j] = int(list1[j])
    
    array_f2 = np.zeros([2656,4])
    for  i in range(0,len(train_data)):
         list1 = list(train_data['f2'][i])
         del list1[0]
         length = len(list1)
         list1 = ['0']*(4-length)+list1
         for j in range(0,4):
             array_f2[i,j] = int(list1[j])
             
    array_f1_test = np.zeros([665,3])
    for  i in range(0,len(test_data)):
         list1 = list(test_data['f1'][i])
         del list1[0]
         length = len(list1)
         list1 = ['0']*(3-length)+list1
         for j in range(0,3):
             array_f1_test[i,j] = int(list1[j])
    
    array_f2_test = np.zeros([665,4])
    for  i in range(0,len(test_data)):
         list1 = list(test_data['f2'][i])
         del list1[0]
         length = len(list1)
         list1 = ['0']*(4-length)+list1
         for j in range(0,4):
             array_f2_test[i,j] = int(list1[j])
             
    from sklearn.preprocessing import MinMaxScaler
    Scaler = MinMaxScaler()
    array_f1 = Scaler.fit_transform(array_f1)  
    array_f2 = Scaler.fit_transform(array_f2)
    array_f1_test = Scaler.fit_transform(array_f1_test)
    array_f2_test = Scaler.fit_transform(array_f2_test)     
    del list1,i,j
    
    # 2) Final_train_matrix 
    from sklearn.model_selection import train_test_split
    feature_matrix_train = feature_f3[0:len(train_data),:]
    feature_matrix_test = feature_f3[len(train_data):len(train_data)+len(test_data)]
    feature_matrix_train = np.c_[array_f1,array_f2,feature_matrix_train]
    feature_matrix_test = np.c_[array_f1_test,array_f2_test,feature_matrix_test]
    return feature_matrix_test,feature_matrix_train

feature_matrix_test,feature_matrix_train = Feature_matrices()

# 3) SVM Algorithm 
clf  = sklearn.svm.SVC(C=1, kernel='linear', gamma=3, probability=True) #play with different kernels and C Values
clf.fit(feature_matrix_train, label_Y)
test_proba = clf.predict_proba(feature_matrix_test)
label_Y_test = clf.predict(feature_matrix_test)

""" if the result should be in a .csv file uncomment this
y_df = pd.DataFrame(label_Y_test,columns=['Prediction Results'])
y_df.index += 1
y_df.to_csv('classificationresults.csv')
"""

"""
############------------Testing Phase-------------------########################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature_matrix_train,label_Y, test_size=0.15, random_state=0)

# KNN Algorithm 
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=30)
neigh.fit(X_train, y_train) 
array = neigh.predict_proba(X_test)
print(log_loss(y_test,array))

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)
print(clf.score(X_test,y_test))
print(clf.score(X_train,y_train))
print(sklearn.metrics.log_loss(y_test,y_proba))

#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)
print(clf.score(X_test,y_test))
print(clf.score(X_train,y_train))
print(sklearn.metrics.log_loss(y_test,y_proba))

"""
