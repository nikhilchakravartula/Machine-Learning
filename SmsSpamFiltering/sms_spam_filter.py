import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import pandas as pd
from sklearn.feature_selection import SelectFromModel



def read_datasets():
    global spambase_word_freq
    global spambase_names
    spambase_names = pd.read_table('spambase.names', sep='.\s+', skiprows=33, engine='python', names=['word', 'type'])
    spambase_names.drop('type', inplace=True, axis=1)
    df1 = pd.DataFrame({'word': ['label']}, index=[len(spambase_names)])
    print(df1)
    # print(spambase_names)
    spambase_names = pd.concat([spambase_names, df1])
    # print(spambase_names.index)
    spambase_word_freq = pd.read_csv('spambase.data', names=spambase_names['word'])
    # print(spambase_word_freq.shape)
    # print(spambase_word_freq.columns)
    # print(spambase_word_freq.isnull().any())
    # print(spambase_word_freq.describe())
    # print(spambase_word_freq.label)


def set_parameters_train_test():
    global validation_size
    global seed
    validation_size = 0.20
    seed = 7


def split_dataset():
    global train
    global test
    global train_predictors
    global train_targets
    global test_predictors
    global test_targets
    train, test = model_selection.train_test_split(spambase_word_freq, test_size=validation_size, random_state=seed)
    train_predictors = train.drop('label', axis=1)
    train_targets = train.label
    test_predictors = test.drop('label', axis=1)
    test_targets = test.label
    test.reset_index(inplace=True, drop=True)


def select_important_features():
    global train_predictors
    global train_targets
    global test_predictors
    global test_targets
    rfc = RandomForestClassifier()
    rfc.fit(train_predictors, train_targets)
    feature_imp = rfc.feature_importances_
    imp_df = pd.DataFrame({'feature': train_predictors.columns, 'importance': feature_imp})
    imp_df.sort_values(by='importance', ascending=True, inplace=True)
    imp_df.set_index('feature', inplace=True)
    imp_df.plot(kind='barh', figsize=(20, 20))

    plt.show()

    rfc = SelectFromModel(rfc, prefit=True)
    train_predictors = rfc.transform(train_predictors)
    test_predictors = rfc.transform(test_predictors)

    print("new shape of train dataset after feature elimination", train_predictors.shape)




def fit_and_test_model(model):
    global train_predictors
    global train_targets
    global test_predictors
    global test_targets
    model.fit(train_predictors, train_targets)
    predictions = model.predict(test_predictors)
    print(model.score(test_predictors, test_targets) * 100)
    results_df = pd.DataFrame({'label': predictions})
    output_df = pd.concat([test_predictors, results_df], axis=1)



def main():
    read_datasets()
    set_parameters_train_test()
    split_dataset()

    print("Before feature elimination")
    print("USING NAIVE BAYES")
    model = GaussianNB()
    fit_and_test_model(model)

    print("USING LOGISTIC REGRESSION")
    model = LogisticRegression()
    fit_and_test_model(model)

"""
    select_important_features()

    print("After feature elimination")
    print("USING NAIVE BAYES")
    model = GaussianNB()
    fit_and_test_model(model)

    print("USING LOGISTIC REGRESSION")
    model = LogisticRegression()
    fit_and_test_model(model)
"""

if __name__ =='__main__':
    main()




#output_df.to_csv('/Users/NC186016/PycharmProjects/SmsSpamFiltering/results.csv')

