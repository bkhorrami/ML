__author__ = 'babak_khorrami'

#-- K Nearest Neighbors (Regression) --

import numpy as np
import pandas as pd
import random
import math
import numpy.matlib
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


class KNN(object):
    def __init__(self, X_train,X_test,y_train):
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train

    def knn_distance(self):
        num_test = self._X_test.shape[0]
        num_train = self._X_train.shape[0]
        dist = np.zeros((num_test, num_train))
        for i in range(num_test):
          dst = np.sum(np.sqrt(((self._X_test[i,:]-self._X_train)**2)),axis=1)
          dist[i,:] = dst

        return dist

    def predict(self,k=2):
        dists = self.knn_distance()
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        idx = dists.argsort(axis = 1)
        y_augment = numpy.matlib.repmat(self._y_train,num_test,1)
        for i in range(num_test):
            y_i = self._y_train[idx[i,0:k]]
            y_pred[i] = float(np.sum(y_i)/k)
        return y_pred
 
#***** Test the Algorithm and Compare results with scikit-learn code   
def main():
    data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",sep = '\s+')
    data = np.array(data)
    data_x = data[:,0:6]
    X = np.c_[np.ones((data_x.shape[0])),data_x]
    y = data[:,6]
    
    knn_reg = KNN(X,X,y)
    y_pred = knn_reg.predict(k=6)
    res1 = np.c_[y_pred.T,y]
    print(res1)
    print(np.sum(np.sqrt((y_pred.T-y)**2)))
    # print("Code Accuracy : ",np.sum(res1[:,0]==res1[:,1])/res1.shape[0])

    # #--- Using scikit-learn to test : 
    # knn = KNeighborsClassifier(n_neighbors=2)
    # knn.fit(X_train,y_train)
    # y_prd = knn.predict(X_test)
    # res2 = np.c_[y_prd.T,y_test]
    # print("Scikit Accuracy : ",np.sum(res2[:,0]==res2[:,1])/res2.shape[0])

    # res = np.c_[y_prd.T,y_pred.T]
    # s = res[:,0]==res[:,1]



    
if __name__ == '__main__':
    main()
        