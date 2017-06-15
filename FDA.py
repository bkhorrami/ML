__author__ = 'babak_khorrami'

import numpy as np
import scipy.optimize
import pandas as pd
from numpy.linalg import inv
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#****** Error Rates *******

#def classError(p,Y_actual,alpha):
def classError(Y_actual,pred,w0):
	Y_res = np.zeros(Y_actual.shape[0])
	ipx = np.where(pred>=w0)[0]
	inx = np.where(pred<w0)[0]
	Y_res[ipx]=1
	Y_res[inx]=-1
	#print(Y_res)
	pix = np.where(Y_res==1)
	nix = np.where(Y_res==-1)
	#print(pix,nix)
	no_r_pos = np.size(pix)
	no_r_neg = np.size(nix)
	#print(no_r_neg,no_r_pos)

	a_pix = np.where(Y_actual==1)
	a_nix = np.where(Y_actual==-1)
	#print("actual indeces " ,a_nix,a_pix)
	no_a_pos = np.size(a_pix)
	no_a_neg = np.size(a_nix)
	#print("test ",Y_res[a_nix]==1)
	fp = np.sum(Y_res[a_nix]==1) #false positive
	fn = np.sum(Y_res[a_pix]==-1) #false negative
	#print("fp = ",fp,", fn = ",fn)
	tp = no_a_pos - fn #true positive
	tn = no_a_neg - fp #true negative
	fpr = fp/(fp+tn) #False Positive Rate
	fnr = fn/(fn+tp) #False Negative Rate


	print("Requested Ratio = ",fn/no_r_neg,"+",fp/no_r_pos,"(",(fn/no_r_neg)+(fp/no_r_pos),")")
	print("FPR=",fpr,", FNR=",fnr,"(",fpr+fnr,")")
	#-------- Create table ---------:
	print("")
	print("                True ")
	print("             +        - ")
	print("           ----------------")
	print("        +  TP=",tp,"    FP=",fp)
	print("Predict    ----------------")
	print("        -  FN=",fn,"    TN=",tn)
	print("           ----------------")
	print("")
	print("")

#**************************

dataFile = '/Users/babak_khorrami/Documents/IEOR_290/midterm/data/train.txt'
#data = pd.read_csv(dataFile)
data = pd.read_table(dataFile)
data = np.array(data)
M = data.shape[1] # Features
N = data.shape[0] # Sample Size
# w0_x = np.ones((N,), dtype=np.int)
# X_train = np.hstack((w0_x,data[:,0:M-1]))   # X: data
# X_train = np.column_stack((w0_x,data[:,0:M-1]))
X_train = data[:,0:M-1]
Y_train = data[:,M-1]	# t: response/class

N_plus = np.sum(np.array(list(map(lambda x:(1+x)/2,list(Y_train)))))
N_minus = np.sum(np.array(list(map(lambda x:(1-x)/2,list(Y_train)))))

D_plus = np.zeros((N,N))
#numpy.fill_diagonal(a, val, wrap=False)
for i in range(N):
	D_plus[i,i]=(Y_train[i]+1)/2

D_minus = np.zeros((N,N))
#numpy.fill_diagonal(a, val, wrap=False)
for i in range(N):
	D_minus[i,i]=(1-Y_train[i])/2

c=np.dot(((1/N_plus)*D_plus - (1/N_minus)*D_minus),np.ones((N,1)))

B_plus = (1/N_plus)*np.ones((N,N))
for i in range(N):
	B_plus[i,i] = 1 - B_plus[i,i]

B_minus = (1/N_minus)*np.ones((N,N))
for i in range(N):
	B_minus[i,i] = 1 - B_minus[i,i]

inner_B = np.dot(np.transpose(B_plus),B_plus) + np.dot(np.transpose(B_minus),B_minus)
inner_1 = np.dot(np.transpose(X_train),inner_B)
inner_2 = inv(np.dot(inner_1,X_train))
mult1 = np.dot(np.transpose(c),X_train)
mult2 = np.dot(mult1,inner_2)
mult3 = np.dot(mult2,np.transpose(X_train))
denom = np.dot(mult3,c)
# print("denominator : ",1/denom)
# print("------------")

numer1 = mult3 = np.dot(inner_2,np.transpose(X_train))
numer = np.dot(numer1,c)
w_star = (1/denom)*numer
#print("w* : ",w_star)

alpha_plus = np.dot(np.dot(np.dot((1./N_plus)*np.ones((1,N)),D_plus),X_train),w_star)
alpha_minus = np.dot(np.dot(np.dot((1./N_minus)*np.ones((1,N)),D_minus),X_train),w_star)

# coef=[0.25,0.5,0.75]
# for c in coef:
# 	w0 = alpha_minus + c*(alpha_plus - alpha_minus)
# 	print(c,w0)
# 	pred = np.dot(X_train,w_star)
# 	res1 = pred[0:25]<=w0
# 	res0 = pred[25:]>w0
# 	print("Error rates = ",(np.sum(res1)+np.sum(res0))/50)
# 	classError(Y_train,pred,w0)




# w0 = alpha_minus + 0.75*(alpha_plus - alpha_minus)
# # print("w0 = ",w0)
# pred = np.dot(X_train,w_star)

# print("w0 = ",w0)
# classError(Y_train,pred,w0)

#res=pred>=w0
# res1 = pred[0:25]<=w0
# res0 = pred[25:]>w0
# print(np.sum(res1))
# print("------------------")
# print(np.sum(res0))
# print("Error rates = ",(np.sum(res1)+np.sum(res0))/50)


#-------******* Test Data *******-------:
dataFileTest = '/Users/babak_khorrami/Documents/IEOR_290/midterm/data/test.txt'
#data = pd.read_csv(dataFile)
dataTest = pd.read_table(dataFileTest)
dataTest = np.array(dataTest)
M = dataTest.shape[1] # Features
N = dataTest.shape[0] # Sample Size
X_test = dataTest[:,0:M-1]
Y_test = dataTest[:,M-1]	# t: response/class
#print(N,X_test,Y_test)
print("--------- Test Results ----------")
coef=[0.25,0.5,0.75]
for c in coef:
	w0 = alpha_minus + c*(alpha_plus - alpha_minus)
	print(c,w0)
	predTest = np.dot(X_test,w_star)
	t1 = predTest[0:10]<=w0
	t2 = predTest[10:]>w0
	print("Test Error rates = ",(np.sum(t1)+np.sum(t2))/20)
	classError(Y_test,predTest,w0)


# predTest = np.dot(X_test,w_star)
# t1 = predTest[0:10]<=w0
# t2 = predTest[10:]>w0
# print("Test Error rates = ",(np.sum(t1)+np.sum(t2))/20)
# print(predTest[0:10]>=w0)
# print("----------------------")
# print(predTest[10:]<=w0)





