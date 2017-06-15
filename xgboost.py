__author__ = 'babak_khorrami'

#*** Extreme Boosting ***

import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

#--Read Train and Test data:
train=pd.read_csv('/Users/babak_khorrami/Documents/machine_learning/sklearn/data/RFdata.txt',sep='\s+',header = None)
test=pd.read_csv('/Users/babak_khorrami/Documents/machine_learning/sklearn/data/RFdata_test.txt',sep='\s+',header = None)

train = np.array(train)
test = np.array(test)

X_train = train[:,:-1]
y_train = train[:,-1]
T_train_xgb = xgb.DMatrix(X_train, y_train)

X_test = test[:,:-1]
y_test = test[:,-1]

T_test_xgb = xgb.DMatrix(X_test)

#-- Parameters of the model:
params = {"objective": "reg:linear","max_depth" : 3 , "eta" : 1 , "alpha" : 2.4 , "lambda" : 1}
num_rounds = 5
watch_list=[(T_test_xgb,'test'),(T_train_xgb,'train')]

gbm = xgb.train(dtrain=T_train_xgb,params=params)
trees = gbm.dump_model(fout='/Users/babak_khorrami/Documents/tmp/testtree.txt', with_stats=True)


Y_pred = gbm.predict(T_test_xgb)
# prc=100*(Y_pred - y_train)/y_train
prc=100*(Y_pred - y_test)/y_test

print(prc)

plt.plot(prc,'.')
plt.show()

# bst = xgb.train(params,train,rounds)
# print(bst.predict(test))
# # prd = bst.predict(test)
#
# T_train_xgb = xgb.DMatrix(X_train, Y_train)
#
# params = {"objective": "reg:linear", "booster":"gblinear"}
# gbm = xgb.train(dtrain=T_train_xgb,params=params)