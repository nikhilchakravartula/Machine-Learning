import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import model_selection
import argparse

seed = 7

def sigmoid(X,theta):
    #print("Shapes",X.shape, "   ",theta.shape)
    #print("theta transpose is",np.dot(theta.T,X))
    return 1/(1+math.e**(-np.dot(theta.T,X)))



def hypothesis(X,theta):
    return sigmoid(X,theta)







def read_and_preprocess_dataset():
    global dataset
    columns = pd.read_table('spambase.names',names=['name'])
    #print(columns)
    dataset = pd.read_csv('spambase.data',names=columns['name'])

    #print(dataset.isnull().any())



    return dataset


def gradient_descent(X,Y):
    global dataset
    global theta
    global alpha
    #print("In GD")
    temp_sum =0
    for i in range(X.shape[0]):
        temp_sum=0
        x = X[i, :]
        y = Y[i, :]
        x = x.reshape(x.shape[0], 1)
        y = y.reshape(y.shape[0], 1)
        #print("X is",x.shape)
        for j in range(X.shape[1]):
                temp_sum = temp_sum+(hypothesis(x,theta)- y)*x[j]

    theta = theta - (alpha /X.shape[0])*( temp_sum )


def error(X,Y):
    global theta
    total_err=0
    for i in range(X.shape[0]):
        x=X[i,:]
        y=Y[i,:]
        x=x.reshape(x.shape[0],1)
        y=y.reshape(y.shape[0],1)
        #print("X  is",x.shape, y.shape)
        total_err+= (hypothesis(x,theta)-y)
    print("Error is",total_err)
    return total_err


def split_train_test():
    global dataset
    train, test = model_selection.train_test_split(dataset, test_size=0.20, random_state=seed)
    train_predictors = train.drop('Class', axis=1)
    test_predictors = test.drop('Class', axis=1)
    train_target = train["Class"]
    test_target = test['Class']
    return train_predictors,train_target,test_predictors,test_target


def logistic_regression(Xp,Yp):
    global alpha
    #print("Xp shape and Yp",Xp.shape,Yp.shape)
    preverror = error(Xp, Yp)
    while(True):

        gradient_descent(Xp,Yp)
        currenterror = error(Xp,Yp)
        if(abs(currenterror-preverror)<0.001):
            break
        preverror=currenterror

def predict(test_X,test_Y):
    global theta
    X= test_X.as_matrix()
    Y=np.array(test_Y.values).reshape(len(test_Y),1)
    for i in range(len(X)):
        if(hypothesis(X[i,:],theta)>0.5):
            print(1,end="    ")
        else:
            print(0,end="     ")
        print(Y[i,:])

def main():
    global dataset
    global theta
    global alpha
   
    dataset = read_and_preprocess_dataset()
    dataset.insert(0,'Dummy',1,allow_duplicates=True)

    train_X,train_Y,test_X,test_Y = split_train_test()
    theta = np.random.rand(len(train_X.columns),1)

    X = train_X.as_matrix()
    Y=np.array(train_Y.values).reshape(len(train_Y),1)
    #print(Y.shape)
    logistic_regression(X,Y)
    predictions = predict(test_X,test_Y)





if __name__=='__main__':
	global alpha
	parser = argparse.ArgumentParser()
	parser.add_argument('--alpha',help="Alpha value to be considered in gradient descent",type=float)
	args=parser.parse_args()
	alpha = args.alpha
	print("Alpha is ",alpha)
	main()