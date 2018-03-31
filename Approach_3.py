# -*- coding: utf-8 -*-
"""
@author: Karri Vivek Reddy
Approach-3
"""
# Motivation: Using a Binary classifiers and a Tree type structure predictor rather than a multi-class classifier
# Parameters for each classifier has to be made prior and has to made ready by this time let it be (Dict_Final)

import pickle
from sklearn.model_selection import train_test_split
with open('Dict_final_36.pkl', 'rb') as f:    Dict_Final = pickle.load(f)

"""
Run Approach-2 code prior to this
Sorting the feature_matrix on the basis of mutual information 
"""
from sklearn.feature_selection import mutual_info_classif
Func = mutual_info_classif(feature_matrix_train, label_Y)
sort_index = (-Func).argsort()
feature_matrix_train_sorted = feature_matrix_train[:,sort_index]
feature_matrix_test_sorted = feature_matrix_test[:,sort_index]

"""
Names : Name of the classifer 'clf12','clf13','clf79' and these are keys of the Dictionary {Dict_Final}
Values of C and Number of features are to be accessed from the dictionary
"""
def select_rows(a,b):
    index_1  = np.array(np.where(label_Y == a))
    index_2  = np.array(np.where(label_Y == b))
    index = np.append(index_1,index_2)
    index.sort()
    Final_feature_matrix = feature_matrix_train_sorted[index,:]
    Label_Y_Final = label_Y[index]
    return Final_feature_matrix,Label_Y_Final
Names = list(Dict_Final.keys())
for i in Names:
    start_time = time.time()
    k = list(i)
    Final_feature_matrix,Label_Y_Final = select_rows(int(k[-1]),int(k[-2]))
    exec(i + " = sklearn.svm.SVC(C=Dict_Final[i][2]).fit(Final_feature_matrix[:,0:Dict_Final[i][0]], Label_Y_Final)")
    del k
del i,Names
# By this point classifiers are initialized and ready for prediction

""" Prediction of Test Dataset """
""" Structure: 1 , A basic structure which follows ascending order in its way [Eg: (1,2) ==> (1,3) ==> (3,4) etc] """
import time
y_pred = np.zeros([len(feature_matrix_test),],dtype=int)
for i in range(0,len(feature_matrix_test)):
    List_indices = [1,2,3,4,5,6,7,8,9]
    start_time = time.time()
    while len(List_indices)>1 :
        a = List_indices[0]
        b = List_indices[1]
        if a>b:
            key = ['c','l','f',str(b),str(a)]
            key = ''.join(key)
        else:
            key = ['c','l','f',str(a),str(b)]
            key = ''.join(key)
        out = eval(key + ".predict(feature_matrix_test_sorted[i,:])")
        if out==a:
            List_indices.remove(b)
        else:
            List_indices.remove(a)
    y_pred[i] = List_indices.pop()
    print("--- %s seconds ---" % (time.time() - start_time))
del a,b,i,key,out

""" Structure:2, Same as above structure but gives the preference according to the value having highest accuracy score """
y_pred = np.zeros([len(feature_matrix_test),])
for i in range(0,len(feature_matrix_test)):
    df_dummy = df.copy()
    df_dummy_2 = df.copy()
    list_index = list(df.index)
    while len(list_index)>1 :
        key = list(df_dummy_2.index)[0]
        small = int(list(key)[-2])
        large = int(list(key)[-1])
        out = eval(key + ".predict(feature_matrix_test_sorted[i,0:Dict_Final[key][0]])")
        if out!=small:
            df_dummy = df_dummy[df_dummy['label_small'] != small]
            df_dummy = df_dummy[df_dummy['label_large'] != small]
            df_dummy_2 = df_dummy[(df_dummy['label_small'] == large) | (df_dummy['label_large'] == large)].copy()
        else:
            df_dummy = df_dummy[df_dummy['label_large'] != large]
            df_dummy = df_dummy[df_dummy['label_small'] != large]
            df_dummy_2 = df_dummy[(df_dummy['label_small'] == small) | (df_dummy['label_large'] == small)].copy()
        list_index = list(df_dummy.index)
    y_pred[i] = out