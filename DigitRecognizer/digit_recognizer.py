import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier





dr_train_dataset= pd.read_csv('train.csv')
print(dr_train_dataset.columns)


predictors = dr_train_dataset.drop('label',axis=1)
targets= dr_train_dataset.label
model = KNeighborsClassifier(n_neighbors=5)

print(len(dr_train_dataset))

dr_test_dataset= pd.read_csv('test.csv')
print(dr_test_dataset.columns)



model.fit(predictors,targets)
predictions= model.predict(dr_test_dataset,targets)



df = pd.DataFrame()
df['ImageId']=np.arange(len(dr_train_dataset)+1)
df['label']=predictions






