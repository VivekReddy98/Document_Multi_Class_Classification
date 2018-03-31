# -*- coding: utf-8 -*-
"""
####   Approach-1
@author: Karri Vivek Reddy
"""
### Loading Datasets
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
"""-----------------------------------------------------------------"""

Feat = -60000
""" Sort TF_IDF based on the words(features) using mean value of the word across all rows """
TF_idf_train = feature_matrix[0:2656,:]
sorted_tfidf_index = np.array(TF_idf_train.mean(0)).reshape(153070,)
sorted_tfidf_index = sorted_tfidf_index.argsort()
new_tf_idf = TF_idf_train[:,sorted_tfidf_index[Feat:-1]]  #value of Feat can be played with

"""  Test Dataset"""
TF_idf_test = feature_matrix[2656:,:]
new_tf_idf_test = TF_idf_test[:,sorted_tfidf_index[Feat:-1]]

""" Training using Logistic Regression """ 
clf = LogisticRegression(solver='newton-cg',C=10,multi_class='ovr')
clf.fit(new_tf_idf,label_Y)
print(clf.score(new_tf_idf,label_Y))
y_proba = clf.predict_proba(new_tf_idf_test)

