__author__ = 'babak_khorrami'
#Kernel Fisher Discriminant:
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from math import pow,exp
from numpy import genfromtxt
from numpy import linalg


def polynomialKernel(x,y,d):
    """
    Calculates the polynomial kernel for two vectors, x,y
    """
    x=np.array(x)
    y=np.array(y)
    if(x.size != y.size):
        print("Vectors must have the same dimensions!")
        return

    return  pow(np.dot(x,y),d)


def polynomialKernelVec(x_new,X,d):
    """
    Calculates the polynomial kernel for a vector, x_new, and matrix X
    """
    x_new=np.array(x_new)
    X=np.array(X)
    if(x_new.size != X.shape[1]):
        print("Vectors must have the same dimensions!")
        return
    N = X.shape[0]
    res=np.ones((N))
    for i in range(N):
        res[i] = pow(np.dot(x_new,X[i,:]),d)

    return res


def gramMatrix(X, d):
    """
    Calculates Gram Matrix for a polynomial kernel type
    """
    X = np.array(X)
    N = X.shape[0] # No of data points
    gram = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            gram[i,j] = polynomialKernel(X[i,:],X[j,:],d)

    return gram

def fitKFD(X,y,gram,lmda):
	"""
	Fits Kernel Fisher Discriminant
	Input:
		X: Features of Training data
		y: Labels of Training data
		gram: Gram matrix
		lmda : Lambda for regularization term
	Output:
		z_star: optimal duals
		alpha_plus
		alpha_minus

	"""
	X = np.array(X)
	y = np.array(y)
	Y_train = y
	M = X.shape[1] # Features
	N = X.shape[0] # Sample Size
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

	B = np.dot(np.transpose(B_plus),B_plus) + np.dot(np.transpose(B_minus),B_minus)
	A = np.dot(B,gram)
	for i in range(N):
		A[i,i]+=lmda
	z_star = np.linalg.solve(A,c)
	alpha_plus = np.dot(np.dot(np.dot((1./N_plus)*np.ones((1,N)),D_plus),gram),z_star)
	alpha_minus = np.dot(np.dot(np.dot((1./N_minus)*np.ones((1,N)),D_minus),gram),z_star)

	return z_star , alpha_plus , alpha_minus

def predict(X_test,X_train,z_star,degree):
	"""
	Make prediction for X_test using the results of Kernel Fisher's Discriminant
	"""
	t = X_test.shape[0]
	pred = np.zeros(t)
	for i in range(t):
		pred[i] = np.dot(polynomialKernelVec(X_test[i,:],X_train,degree),z_star)

	return pred

def classError(Y_actual,pred,w0):
	Y_res = np.zeros(Y_actual.shape[0])
	ipx = np.where(pred>=w0)[1]
	inx = np.where(pred<w0)[1]
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



def main():
	dataFile = '/Users/babak_khorrami/Documents/IEOR_290/midterm/data/train.txt'
	data = pd.read_table(dataFile)
	data = np.array(data)
	M = data.shape[1] # Features
	N = data.shape[0] # Sample Size
	X_train = data[:,0:M-1]
	Y_train = data[:,M-1]	# t: response/class
	#*********** Parameters **********
	# degree = 4
	# lmda = 100
	# #*********************************
    #
	# gram = gramMatrix(X_train, degree)
    #
	# z_star , alpha_plus , alpha_minus = fitKFD(X_train,Y_train,gram,lmda)

	#******* TEST DATA ********:
	testFile = '/Users/babak_khorrami/Documents/IEOR_290/midterm/data/test.txt'
	testData = pd.read_table(testFile)
	testData = np.array(testData)
	M = testData.shape[1] # Features
	N = testData.shape[0] # Sample Size
	X_test = np.array(testData[:,0:M-1])
	Y_test = testData[:,M-1]	# t: response/class

	# pred = predict(X_test,X_train,z_star,degree)
	# w0 = 0.50*(alpha_plus - alpha_minus)
	#print(pred)
	# print("Class +1 : ",pred[0:10]>=w0)
	# print("------------------------------------------------")
	# print("Class -1 : ",pred[10:]<w0)

	#-------- Train and Test Data --------:
	# Deg = [2,4]
	# Lam = [1,10,100]
	# for l in Lam:
	# 	for d in Deg:
	# 		gram = gramMatrix(X_train, d)
	# 		z_star , alpha_plus , alpha_minus = fitKFD(X_train,Y_train,gram,l)
	# 		pred_train = predict(X_train,X_train,z_star,d)
	# 		w0 = alpha_minus + 0.50 * (alpha_plus - alpha_minus)
	# 		classError(Y_train,pred_train,w0)


	#------- TRAINING DATA -------
	# Deg = [2,4]
	# Lam = [1,10,100]
	# for d in Deg:
	# 	gram = gramMatrix(X_train, d)
	# 	for l in Lam:
	# 		z_star , alpha_plus , alpha_minus = fitKFD(X_train,Y_train,gram,l)
	# 		pred_train = predict(X_train,X_train,z_star,d)
	# 		w0 = alpha_minus + 0.50 * (alpha_plus - alpha_minus)
	# 		err = (np.sum(pred_train[0:25]<w0)+np.sum(pred_train[25:]>=w0))/50
	# 		print("lambda = ",l,"degree = ",d)
	# 		print("1 - Accuracy Rate = ",err)
	# 		classError(Y_train,pred_train,w0)
	# 		print("----------------------------------------------")


	#-------- TEST DATA --------
	Deg = [2,4]
	Lam = [1,10,100]
	for d in Deg:
		gram = gramMatrix(X_train, d)
		for l in Lam:
			z_star , alpha_plus , alpha_minus = fitKFD(X_train,Y_train,gram,l)
			pred_test = predict(X_test,X_train,z_star,d)
			w0 = alpha_minus + 0.50 * (alpha_plus - alpha_minus)
			err = (np.sum(pred_test[0:10]<w0)+np.sum(pred_test[10:]>=w0))/20
			print("lambda = ",l,"degree = ",d)
			print("1 - Accuracy Rate = ",err)
			classError(Y_test,pred_test,w0)
			print("----------------------------------------------")





if __name__ == '__main__': main()







