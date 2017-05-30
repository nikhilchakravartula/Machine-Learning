import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def read_datasets():
    global iris_dataset
    iris_dataset = pd.read_csv('iris.data.txt',names=['sepal_length','sepal_width','petal_length','petal_width','class'])
    print(iris_dataset.head())


def show_visuals():
    global iris_read_datasets
    fig = plt.figure()
    ax= fig.add_subplot(111)
    plt.legend()
    plt.show()

def split_params():

    global validation_size
    global seed
    validation_size = 0.2
    seed=7

def split_dataset():
    split_params()
    global iris_dataset
    global train_predictors,test_predictors,train_targets,test_targets
    train,test = train_test_split(iris_dataset,test_size=validation_size,random_state=seed)
    train_predictors = train.drop('class',axis=1)
    train_targets= train['class']
    test_predictors= test.drop('class',axis=1)
    test_targets= test['class']
    test_predictors.reset_index(inplace=True,drop=True)


def fit_to_model_and_test(model):
    global train_predictors,test_predictors,train_targets,test_targets
    model.fit(train_predictors,train_targets)
    print(model.score(test_predictors,test_targets)*100)
    predictions = model.predict(test_predictors)
    output_df= pd.DataFrame({'sepal_length':test_predictors['sepal_length'],'sepal_width':test_predictors['sepal_width'],
                             'petal_length':test_predictors['petal_length'],'petal_width':test_predictors['petal_width'],'class':predictions
                             })
    output_df.to_csv('results.csv')

def main():
    read_datasets()
    global iris_dataset
    print(iris_dataset.shape)
    print(iris_dataset.describe())
    print(iris_dataset.isnull().any())
    split_dataset()
    model= SVC()
    fit_to_model_and_test(model)

if __name__ == "__main__":
    main()