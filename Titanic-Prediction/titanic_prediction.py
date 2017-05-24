import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def finished(feature):
    print ("Finished ", feature)

def preprocess_age():
    global tt_train_dataset
    tt_train_dataset['Age'].fillna(tt_train_dataset["Age"].median(),inplace=True)
    finished("Age")


def findMax(list):
    max_value=max(list)
    return list.index(max_value)

def preprocess_embarked():
    global tt_train_dataset
    if tt_train_dataset['Embarked'].isnull().any():
        print ("Found null in embarked, replacing with highest value")
        print ("Values are ",tt_train_dataset['Embarked'].value_counts())
        value=tt_train_dataset['Embarked'].value_counts().index.tolist()[0]
        tt_train_dataset['Embarked'].fillna(value)

    dummies=pd.get_dummies(tt_train_dataset['Embarked'],prefix="Embarked")
    tt_train_dataset=pd.concat([tt_train_dataset,dummies ],axis=1)
    tt_train_dataset.drop("Embarked",axis=1,inplace=True)
    finished("Embarked")

def preprocess_cabin():
    global  tt_train_dataset
    tt_train_dataset['Cabin'].fillna('U',inplace=True)
    tt_train_dataset['Cabin']=tt_train_dataset['Cabin'].map(lambda ch : ch[0])
    dummies = pd.get_dummies(tt_train_dataset['Cabin'],prefix="Cabin")
    tt_train_dataset=pd.concat([tt_train_dataset,dummies],axis=1)
    tt_train_dataset.drop("Cabin",axis=1,inplace=True)
    finished("Cabin")

def preprocess_sex():
    global tt_train_dataset
    tt_train_dataset['Sex'].replace(['male','female'],[0,1],inplace=True)
    finished("Sex")

def preprocess_name():
    global  tt_train_dataset
    tt_train_dataset.drop("Name",axis=1,inplace=True)
    finished("Name")

def preprocess_pclass():
    global tt_train_dataset
    dummies=pd.get_dummies(tt_train_dataset['Pclass'],prefix="Pclass")
    tt_train_dataset=pd.concat([tt_train_dataset,dummies],axis=1)
    tt_train_dataset.drop("Pclass",axis=1,inplace=True)
    finished("Pclass")

def preprocess_ticket():
    global tt_train_dataset
    tt_train_dataset.drop("Ticket",axis=1,inplace=True)

def preprocess_fare():
    global tt_train_dataset
    tt_train_dataset['Fare'].fillna(tt_train_dataset['Fare'].median(),inplace=True)
    finished("Fair")

def preprocess_passengerid():
    global tt_train_dataset
    tt_train_dataset.drop("PassengerId",axis=1,inplace=True)

seed = 7
validation_size = 0.20

tt_train_dataset = pd.read_csv('tt_train.csv')
tt_test_dataset = pd.read_csv('tt_test.csv')

tt_train_dataset=tt_train_dataset.append(tt_test_dataset)
print(tt_train_dataset.columns)

tt_train_dataset.reset_index(inplace=True)
tt_train_dataset.drop("index",inplace=True,axis=1)


preprocess_passengerid()
preprocess_age()
preprocess_embarked()
preprocess_cabin()
preprocess_sex()
preprocess_name()
preprocess_pclass()
preprocess_ticket()
preprocess_fare()

print ( tt_train_dataset.shape)
print(tt_train_dataset.columns)
print(tt_train_dataset.isnull().any())
targets= tt_train_dataset.head(891).Survived
train = tt_train_dataset.head(891).drop("Survived",axis=1)
test = tt_train_dataset.iloc[891:].drop("Survived",axis=1)


print( "train dataset\n",train.head(10))


model =RandomForestClassifier()
model.fit(train,targets)
importances= model.feature_importances_

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = model.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='bar', figsize=(20, 20))
plt.show()


model= SelectFromModel(model,prefit=True)

train=model.transform(train)
test=model.transform(test)

print("Finished training")

model=RandomForestClassifier()
model.fit(train,targets)
predictions = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('tt_test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = predictions
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)




