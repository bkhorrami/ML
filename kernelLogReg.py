__author__ = 'babak_khorrami'

#*** Kernelized Logistic Regression***

from math import pow,exp
from numpy import genfromtxt
from numpy import linalg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
# from scipy.optimize import minimize
from cvxopt import solvers, matrix, spdiag, log

class KerLogReg:

    def __init__(self,X=0,t=0,kernel="identity"):
        pass


    @staticmethod
    def gramMatrix(X, kerParam , kernel = "polynomial"):
        """
        Calculates Gram Matrix for a given kernel type
        :param X: matrix of features (Data)
        :param kernel:
        :return:
        """
        X = np.array(X)
        N = X.shape[0] # No of data points
        gram = np.zeros((N,N))
        if kernel == "gaussian":
            for i in range(N):
                for j in range(N):
                    gram[i,j] = KerLogReg.gaussianKernel(X[i,:],X[j,:],kerParam)
        elif kernel == "polynomial":
            for i in range(N):
                for j in range(N):
                    gram[i,j] = KerLogReg.polynomialKernel(X[i,:],X[j,:],kerParam)
        else:
            print("Please choose either polynomial or gaussian kernel.")
            return

        return gram


    @staticmethod
    def identityKernel(x,y):
        return np.dot(x,y)


    @staticmethod
    def gaussianKernel(x,y,sigma):
        """
        Calculates the Gaussian kernel for a vectors, x_new, and a Matrix X
        :param x_new: first vector
        :param X:  Matrix/vector
        :param sigma: Gaussian kernel parameter
        :return: gaussian kernel : k(x_new,X)
        """
        x=np.array(x)
        y=np.array(y)
        if(x.size != y.size):
            print("Vectors must have the same dimensions!")
            return
        v = np.subtract(x,y)
        return exp(-pow(linalg.norm(v,2),2)/(2*pow(sigma,2)))

    @staticmethod
    def gaussianKernelVec(x_new,X,sigma):
        """
        A Helper static method to calculate K_i, a row of kernel values for Log Likelihood
        Calculates the Gaussian kernel for a vectors, x_new, and a Matrix X
        :param x_new: first vector
        :param X:  Matrix/vector
        :param sigma: Gaussian kernel parameter
        :return: gaussian kernel : k(x_new,X)
        """
        x_new=np.array(x_new)
        X=np.array(X)
        N = X.shape[0]
        if(x_new.size != X.shape[1]):
            print("Vectors must have the same dimensions!")
            return
        res=np.zeros(N)
        for i in range(X.shape[0]):
            v = np.subtract(x_new,X[i,:])
            res[i] = exp(-pow(linalg.norm(v,2),2)/(2*pow(sigma,2)))

        return res



    @staticmethod
    def polynomialKernel(x,y,d):
        """
        Calculates the polynomial kernel for two vectors, x,y
        :param x: First vector
        :param y: Second vector
        :param d: degree of polynomial
        :return: polynomial kernel, k(x,y)
        """
        x=np.array(x)
        y=np.array(y)
        if(x.size != y.size):
            print("Vectors must have the same dimensions!")
            return

        return  pow(np.dot(x,y),d)

    @staticmethod
    def polynomialKernelVec(x_new,X,d):
        """
        Calculates the polynomial kernel for two vectors, x,y
        :param x: First vector
        :param y: Second vector
        :param d: degree of polynomial
        :return: polynomial kernel, k(x,y)
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

    @staticmethod
    def logistic(z):
        """
        Calculates the sigmoid function for vector z
        :param z: n-dimensional vector
        :return: logistic/sigmoid function of z
        """
        return np.array(1.0 / (1.0 + np.exp(-z)))


    @staticmethod
    def log_likelihood(K, t, v , C=1.0):
        N = K.shape[0] # No of data points (sample size)
        #w0 = v[0]
        z = v[0:N]
        alpha = v[N]
        log_sum = 0

        for i in range(N):
            log_sum += np.log(1+np.exp(-t[i]*(np.dot(K[i,:],z))))


        #Add (l/2)zKz:
        penalty = C*np.dot(np.dot(z,K),z)
        lagrange = alpha*np.sum(z)
        log_sum = log_sum + penalty + lagrange

        return log_sum

    @staticmethod
    def grad_log_likelihood(K, t, v , C=1.0):
        """
        Calculates the Gradient of the Objective function
        """
        N = K.shape[0] # No of data points (sample size)
        #w0 = v[0]
        z = v[0:N]
        alpha = v[N]
        grad = np.zeros(N+1) #Gradient vector has N+2 elements , (w0,z,alpha)


        #calculate elements 2nd to Nth elements of Gradient vector (dL/dz(1),...,dL/dz(N)):
        for k in range(0,N):
            grad[k]+=alpha
            for i in range(N):
                grad[k] = grad[k] - t[i] * K[i,k-1] * KerLogReg.logistic(t[i]*(np.dot(K[i,:],z)))\
                          + C * z[i] * K[i,k-1]


        grad[N] = grad[N]+np.sum(z)

        return grad



    def fit(self, K , t , C=1.0):
        """
        Fits a Kernel Logistic Regression for a Two-Class
        """
        def obj(v):
            return KerLogReg.log_likelihood(K, t, v , C)

        def objPrime(v):
            return KerLogReg.grad_log_likelihood(K, t, v , C)


        initial_guess = 0.001 * np.ones(K.shape[0]+1)
        return scipy.optimize.fmin_bfgs(obj, initial_guess, objPrime, disp=False)

    @staticmethod
    def predict(X_test,X_train,z,sigma):
        n=X_test.shape[0] # No of test data
        N = X_train.shape[0]
        prob = np.zeros(n)
        for i in range(n):
            k=KerLogReg.gaussianKernelVec(X_test[i],X_train,sigma)
            prob[i] = 1/(1+exp(-(np.dot(k,z))))

        return prob

    @staticmethod
    def classError(p,Y_actual,alpha):
        Y_res = np.zeros(Y_actual.shape[0])
        ipx = np.where(p>=alpha)
        inx = np.where(p<alpha)
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
    X_train = np.array(data[:,0:M-1])
    Y_train = data[:,M-1]	# t: response/class
    #*** Params ****
    lmbd = 100
    sigma = 10
    alph = 0.50
    #***************
    klr = KerLogReg()
    #K = klr.gramMatrix(X_train,2,"gaussian") #Gram Matrix
    K = klr.gramMatrix(X_train,sigma,"gaussian") #Gram Matrix

    r = klr.fit(K , Y_train ,lmbd)
    #w0 = r[0]
    z=r[0:50]
    #*****  TEST ON TRAINING DATA ******:
    p=KerLogReg.predict(X_train,X_train,z,sigma)
    #print(p)

    # print("RESULTS = ",p[0:25]>=alph,"---",p[25:]<alph)
    # print(np.sum(p[0:25]>alph))
    # print(np.sum(p[25:]<=alph))
    # print("Training Error Rate : ",(np.sum(p[0:25]<=alph)+np.sum(p[25:]>alph))/50)
    #
    #
    # print("------ TEST DATA -----")

    #************* TEST DATA  *****************
    testFile = '/Users/babak_khorrami/Documents/IEOR_290/midterm/data/test.txt'
    testData = pd.read_table(testFile)
    testData = np.array(testData)
    M = testData.shape[1] # Features
    N = testData.shape[0] # Sample Size
    X_test = np.array(testData[:,0:M-1])
    Y_test = testData[:,M-1]	# t: response/class
    p_test=KerLogReg.predict(X_test,X_train,z,sigma)
    # print(p_test)
    # print(p_test>alph)
    # print(np.sum(p_test[0:10]>alph))
    # print(np.sum(p_test[10:]<=alph))
    # print("Test Error Rate : ",(np.sum(p_test[0:10]<=alph)+np.sum(p_test[10:]>alph))/20)

    train_iter=[]
    test_iter=[]
    train_error=[]
    test_error=[]
    Lamb=[1,10,100]
    Sig = [1,10]
    for l in Lamb:
        for s in Sig:
            klogreg = KerLogReg()
            K = klogreg.gramMatrix(X_train,s,"gaussian") #Gram Matrix
            r = klr.fit(K , Y_train ,l)
            z=r[0:50]
            #***** TRAINING DATA ******:
            p=KerLogReg.predict(X_train,X_train,z,s)
            # print("lambda = ",l,", sigma = ",s,", Training Error Rate : ",(np.sum(p[0:25]<=alph)+np.sum(p[25:]>alph))/50)
            # print((np.sum(p[0:25]<=alph)+np.sum(p[25:]>alph))/50)
            # KerLogReg.classError(p,Y_train,alph)
            err = (np.sum(p[0:25]<=alph)+np.sum(p[25:]>alph))/50
            train_iter.append(l)
            train_iter.append(s)
            train_iter.append(err)
            train_error.append(train_iter)
            train_iter=[]
            #print("*************************************")
            # #**** TEST DATA *****
            p_test=KerLogReg.predict(X_test,X_train,z,s)
            t_err = (np.sum(p_test[0:10]<=alph)+np.sum(p_test[10:]>alph))/20
            test_iter.append(l)
            test_iter.append(s)
            test_iter.append(t_err)
            test_error.append(test_iter)
            print("lambda = ",l,", sigma = ",s,", Test Error Rate : ",t_err)
            KerLogReg.classError(p_test,Y_test,alph)
            print("-----------------------------------------------------")



if __name__ == '__main__':
    main()



# log_sum += np.sum(np.log(1+exp(-t[i]*(w0 + np.dot(K,z))))
# return np.sum(np.log(KerLogReg.logistic(Y * np.dot(X, z)))) - C/2 * np.dot(z, z)

