__author__ = 'babak_khorrami'

#*** Kernelized Support Vector Machine ****

import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from math import pow,exp
from numpy import genfromtxt
from numpy import linalg

class SVM:
    def __init__(self):
        pass

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
    def polynomialKernel(x,y,d=2):
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
                    gram[i,j] = SVM.gaussianKernel(X[i,:],X[j,:],kerParam)
        elif kernel == "polynomial":
            for i in range(N):
                for j in range(N):
                    gram[i,j] = SVM.polynomialKernel(X[i,:],X[j,:],kerParam)
        else:
            print("Please choose either polynomial or gaussian kernel.")
            return

        return gram

    @staticmethod
    def classError(Y_actual,pred,w0):
        Y_res = np.zeros(Y_actual.shape[0])
        ipx = np.where(pred>w0)[0]
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





def main():
    dataFile = '/Users/babak_khorrami/Documents/IEOR_290/midterm/data/train.txt'
    data = pd.read_table(dataFile)
    data = np.array(data)
    M = data.shape[1] # Features
    N = data.shape[0] # Sample Size
    X_train = np.array(data[:,0:M-1])
    Y_train = data[:,M-1]	# t: response/class
    #****** sklearn SVC ******
    #clf = svm.SVC(C=1,kernel='rbf',gamma=0.005)
    clf = svm.SVC(C=1,kernel='poly',degree=4,coef0=0)
    clf.fit(X_train,Y_train)
    # pred=clf.predict(X_train)
    # SVM.classError(Y_train,pred,0)
    # print("-----------------------------------")


    #******* TEST DATA ********:
    testFile = '/Users/babak_khorrami/Documents/IEOR_290/midterm/data/test.txt'
    testData = pd.read_table(testFile)
    testData = np.array(testData)
    M = testData.shape[1] # Features
    N = testData.shape[0] # Sample Size
    X_test = np.array(testData[:,0:M-1])
    Y_test = testData[:,M-1]	# t: response/class
    pred_test = clf.predict(X_test)
    print(clf.predict(X_test))
    SVM.classError(Y_test,pred_test,0)


    # gram = SVM.gramMatrix(X_train,10,"gaussian")
    # preCLF = svm.SVC(kernel='precomputed')
    # preCLF.fit(gram,Y_train)
    # print("---------- Precomputed ------------")
    # print(preCLF.predict(gram))


    #***** precomputed test *******
    # gram = SVM.gramMatrix(X_train,10,"gaussian")
    # preCLF = svm.SVC(kernel='precomputed')
    # preCLF.fit(gram,Y_train)
    # print("---------- Precomputed ------------")
    # print(preCLF.predict(gram))


    #********* PLOT THE SVM *********
    #create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    title = 'SVM with Polynomial Kernel, (degree = 4) for Test Data'
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

     # Plot also the training points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=plt.cm.Paired)
    #plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Paired)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

    plt.show()

    #--------- Training Plot -----------
    #********* PLOT THE SVM *********
    #create a mesh to plot in
    # h = .02  # step size in the mesh
    # x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    # y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))
    #
    # title = 'SVM with Polynomial Kernel, (degree = 4) for Training Data'
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    #
    #  # Plot also the training points
    # #plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=plt.cm.Paired)
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Paired)
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())
    # plt.title(title)
    #
    # plt.show()





if __name__ == '__main__':
    main()


