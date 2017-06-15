__author__ = 'babak_khorrami'

#*** Logistic Regression ****

import numpy as np
import scipy.optimize
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def logistic(z):
    return np.array(1.0 / (1.0 + np.exp(-z)))

def predict(w, X , alpha):
    chk = logistic(np.dot(X, w))
    res=[]
    for i in range(chk.shape[0]):
        if chk[i]>alpha:
            res.append(1)
        else:
            res.append(-1)

    prob = chk


    return prob , np.array(res)


def predict2(w, X , alpha):
    chk = logistic(np.dot(X, w))
    res=np.zeros(chk.shape[0],)
    res[np.where(chk>alpha)] = 1
    res[np.where(chk<=alpha)] = -1
    return chk , res

def log_likelihood(X, Y, w, C=0.1):
    return np.sum(np.log(logistic(Y * np.dot(X, w)))) - C/2 * np.dot(w, w)

def log_likelihood_grad(X, Y, w, C=0.1):
    K = len(w)
    N = len(X)
    s = np.zeros(K)

    for i in range(N):
        s += Y[i] * X[i] * logistic(-Y[i] * np.dot(X[i], w))

    s -= C * w

    return s


def train_w(X, Y, C=0.1):
    def f(w):
        return -log_likelihood(X, Y, w, C)

    def fprime(w):
        return -log_likelihood_grad(X, Y, w, C)

    K = X.shape[1]
    initial_guess = np.zeros(K)
    return scipy.optimize.fmin_bfgs(f, initial_guess, fprime, disp=False)


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
    print("FPR=",fpr,",FNR=",fnr,"(",fpr+fnr,")")
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
    #----- Training Data -------
    dataFile = '/Users/babak_khorrami/Documents/IEOR_290/midterm/data/train.txt'
    #data = pd.read_csv(dataFile)
    data = pd.read_table(dataFile)
    data = np.array(data)
    M = data.shape[1] # Features
    N = data.shape[0] # Sample Size
    w0_x = np.ones((N,), dtype=np.int)
    X_train = np.column_stack((w0_x,data[:,0:M-1]))
    Y_train = data[:,M-1]	# t: response/class
    w = train_w(X_train, Y_train, 0)
    print ("w was", w)
    alpha = [0.25,0.50,0.75]
    # for i in alpha:
    #     prob , pred = predict(w,X_train,i)
    #     classError(prob,Y_train,i)




    prd_50 = predict(w,X_train,0.5)
    print("PRD50 = ",prd_50[25:])
    print("Error Rate : ",(np.sum(prd_50[0:25]==-1)+np.sum(prd_50[25:]== 1))/50)
    print("----")

    #********** Plot the Logistic Reression results: TRAINING DATA ********
    X = data[:,0:M-1]
    h = .02  # step size in the mesh
    alph = 0.50
    #the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    n=xx.ravel().shape[0]
    # Z = model.predict(np.c_[np.ones(n),xx.ravel(), yy.ravel()])
    Z = predict(w, np.c_[np.ones(n),xx.ravel(), yy.ravel()] , alph)
    
    
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y_train, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('X1')
    plt.ylabel('X2')
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    plt.show()

    #********---------- TEST DATA ---------********:
    testDataFile = '/Users/babak_khorrami/Documents/IEOR_290/midterm/data/test.txt'
    #data = pd.read_csv(dataFile)
    testData = pd.read_table(testDataFile)
    testData = np.array(testData)
    M = testData.shape[1] # Features
    N = testData.shape[0] # Sample Size
    w0_x = np.ones((N,), dtype=np.int)
    #X_train = np.hstack((w0_x,data[:,0:M-1]))   # X: data
    X_test = np.column_stack((w0_x,testData[:,0:M-1]))
    Y_test = testData[:,M-1]	# t: response/class
    prd_50 = predict(w,X_test,0.50)
    for i in alpha:
        prob , pred = predict(w,X_test,i)
        print("Error Rate : ",(np.sum(pred[0:10]==-1)+np.sum(pred[10:]== 1))/20)
        classError(prob,Y_test,i)

    # print("________________ Test Data __________________")
    # print("Error Rate : ",(np.sum(prd_50[0:10]==-1)+np.sum(prd_50[10:]== 1))/20)

# h = .02  # step size in the mesh
# #the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, m_max]x[y_min, y_max].
# X_t = testData[:,0:M-1]
# x_min, x_max = X_t[:, 0].min() - .5, X_t[:, 0].max() + .5
# y_min, y_max = X_t[:, 1].min() - .5, X_t[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# n=xx.ravel().shape[0]
# Z = predict(w, np.c_[np.ones(n),xx.ravel(), yy.ravel()] , alph)
#
#
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
# # Plot also the training points
# plt.scatter(X_t[:, 0], X_t[:, 1], c=Y_test, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('X1')
# plt.ylabel('X2')
#
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
#
# plt.show()


if __name__ == "__main__":
    main()
