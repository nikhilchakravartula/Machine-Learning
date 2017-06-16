import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.metrics import r2_score


def efficieny(predicitions,targets):
    return r2_score(predicitions,targets)



def hypothesis(currentparameters,X):
 #   print("in hypothesis current parameters",currentparameters)
#    print("X is ",X)
    linear_eq = np.multiply(currentparameters,X)
    #print("multiplt is ",linear_eq)
    return np.sum(linear_eq)




def error_func():
    total_err=0.0
    global currentparameters
    predictor_values = train_predictors.as_matrix()
    target_value = train_target.as_matrix()
    for i in range(0,predictor_values.shape[0]):

        hypothesis_value = hypothesis(currentparameters,predictor_values[i])
       # print("hypothesis and target values ", hypothesis_value, "   ", target_value[i])
        temp_result = (hypothesis_value-target_value[i])**2
        total_err+=temp_result
    return total_err






def gradient_descent(alpha):
    global currentparameters
    global train_predictors
    global train_target
    predictor_values = train_predictors.as_matrix()
    target_value = train_target.as_matrix()
    temp_parameters =currentparameters.copy()


    m = len(predictor_values[0])
   # print ("m is",m)
    for j in range(0,m):
        total_result=0.0
        #print("inside j value is ",j)
        for i in range(0,predictor_values.shape[0]):
         #   print("inside i value is ",i)
            hypothesis_value = hypothesis(currentparameters,predictor_values[i])
            temp_result = (hypothesis_value-target_value[i])*predictor_values[i][j]
            total_result+=temp_result
       # print("alpha by ",total_result)
        temp_parameters[j]=temp_parameters[j]-(alpha/predictor_values.shape[0])*total_result

   # print("curr",currentparameters)
   # print("temp",temp_parameters)
    currentparameters=temp_parameters.copy()
    #print(currentparameters)







def linear_regression(alpha):

    prev_error=error_func()
    while(1):
        #print("error is ",prev_error)
        gradient_descent(alpha)
        error  = error_func()
        if(abs(error - prev_error)<0.01):
            break
        prev_error=error



def predict(test_predictors,test_targets):
    global currentparameters
    predictions= np.zeros((test_predictors.shape[0]),dtype=np.float)
    test_predictors_values = test_predictors.as_matrix()
    test_targets = test_targets.as_matrix()
    for i in range(0,test_predictors.shape[0]):
        print(i)
        predictions[i]= hypothesis(currentparameters,test_predictors_values[i])
        print(predictions[i],"   ",test_targets[i])

    print(efficieny(predictions,test_targets))


def main():
    global train_predictors
    global train_target
    global currentparameters
    alpha = 0.0004
    train_data_no_rows=0
    dataset = pd.read_table('winequality-red.csv', sep=';')

    train, test = train_test_split(dataset, test_size=0.2, random_state=7)

    train_predictors = train.drop('quality', axis=1)
    train_target = train['quality']
    train_predictors.insert(0, 'Dummy', 1)

    test_predictors = test.drop('quality', axis=1)
    test_target = test['quality']
    test_predictors.insert(0,'Dummy',1)

    no_parameters = len(train_predictors.columns)
    train_data_no_rows=len(train_predictors)
    #print("no of rows in training data",train_data_no_rows)
    currentparameters = np.zeros((no_parameters),dtype= np.float)
    #print(currentparameters)
    #gradient_descent(alpha,train_data_no_rows)
    linear_regression(alpha)
    predict(test_predictors,test_target)




if __name__ == '__main__':
    main()


