import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import operator

def perf(y_predict,y_actual):
    return r2_score(y_predict,y_actual)



def preprocess_for_NN():
    global dataset
    global columns
    for column in columns:
        if column!='Cost':
            xmax = dataset[column].max()
            xmin = dataset[column].min()
            dataset[column]=(dataset[column]- xmin)/(xmax - xmin)


def select_imp_features():
    global dataset
    global columns
    rfc = RandomForestRegressor()
    rfc. fit(dataset.drop('Cost',axis=1),dataset.Cost)
    feature_imp = rfc.feature_importances_






def fit_and_test(model):
    model.fit(train_predictors, train_targets)
    predictions = model.predict(test_predictors)

    output_df = pd.DataFrame()
    output_df = test_predictors
    output_df.reset_index(inplace=True)
    output_df['Cost']= predictions
   # print(predictions)
    print(perf(test_targets, predictions))
    output_df.to_csv('housing_predictions.csv')

desired_width = 320
pd.set_option('display.width', desired_width)
seed = 7
columns = ['CrimeRate','Lots','INDUS','RiverBound','NOX','AvgRooms','Age','EmpDistance','HighwayAccess','Tax','PTratio','Blacks','Lstat','Cost']
dataset= pd.read_table("housing.data.txt",sep='\s+',names= columns)
tempdataset = dataset.copy()
corr_values=dataset.corr(method = 'pearson')
corr_values=corr_values.Cost
#print(corr_values.sort_values(ascending=False))
preprocess_for_NN()
#select_imp_features()
print(dataset.head())
train, test = model_selection.train_test_split(dataset,test_size=0.2,random_state=seed)

train_predictors = train.drop('Cost',axis=1)
train_targets = train.Cost

test_predictors = test.drop('Cost',axis=1)
test_targets = test.Cost


model =  RandomForestRegressor()
fit_and_test(model)
