import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)
pd.options.mode.chained_assignment = None

def status(feature):
    print(feature, "  :  OK")

def preprocess_MsZoning():
    global dataframe
    dataframe.fillna({'MSZoning': 'RL'}, inplace=True)
    status("MSZONING")

def replace_nan_with_median(x,flag):
    global grouped_train_median
    global grouped_test_median
    if(flag==0):
        return grouped_train_median.ix[x['MSSubClass']]['LotFrontage']
    else:
        return grouped_test_median.ix[x['MSSubClass']]['LotFrontage']

def contains_nan(series):
    return series.value_counts().sum() != len(series)


def preprocess_lot_frontage():
    global dataframe
    global grouped_train_median
    global grouped_test_median
    grouped_train_median =dataframe.head(1460).groupby(by='MSSubClass').median()
    grouped_test_median = dataframe.iloc[1460:].groupby(by="MSSubClass").median()
    dataframe.head(1460)['LotFrontage']=dataframe.head(1460).apply(lambda x: replace_nan_with_median(x,0) if np.isnan(x['LotFrontage']) else x['LotFrontage'],axis=1)
    dataframe.iloc[1460:]['LotFrontage'] = dataframe.iloc[1460:].apply(lambda x : replace_nan_with_median(x,1) if np.isnan(x['LotFrontage']) else x['LotFrontage'],axis=1)
    dataframe.fillna({'LotFrontage':grouped_train_median['LotFrontage'].median()},inplace=True)
    #print(dataframe)
    status("LotFrontage")

def preprocess_alley():
    global dataframe
    dataframe.drop('Alley',axis=1,inplace=True)
    status("Alley")

def preprocess_lotshape():
    global dataframe
    dummies = pd.get_dummies(dataframe['LotShape'],prefix='LotShape')
    #dataframe= pd.concat([dataframe,dummies],axis=1)
    dataframe.drop("LotShape",axis=1,inplace=True)
    status("LotShape")
    return dummies

def preprocess_land_contour():
    global dataframe
    dummies = pd.get_dummies(dataframe['LandContour'],prefix = 'LandContour')
    #dataframe = pd.concat([dataframe,dummies],axis =1)
    dataframe.drop("LandContour",axis=1,inplace=True)
    status("LandContour")
    return dummies

def preprocess_utilities():
    global dataframe
    #dropping because most of the values are same
    dataframe.drop('Utilities',axis=1,inplace=True)
    status("Utilities")


def preprocess_lotconfig():
    global dataframe
    dummies = pd.get_dummies(dataframe["LotConfig"],prefix="LotConfig")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop("LotConfig",axis=1,inplace=True)
    status("LotConfig")
    return dummies

def preprocess_landslope():
    global dataframe
    dummies = pd.get_dummies(dataframe["LandSlope"],prefix="LandSlope")
    #dataframe= pd.concat([dataframe,dummies],axis=1)
    dataframe.drop("LandSlope",axis=1,inplace=True)
    status("Landslope")
    return dummies

def preprocess_neighborhood():
    global dataframe
    dummies = pd.get_dummies(dataframe["Neighborhood"],prefix="Neighborhood")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('Neighborhood',axis=1,inplace=True)
    status("Neighborhood")
    return dummies

def preprocess_condition1():
    global dataframe
    dummies = pd.get_dummies(dataframe["Condition1"],prefix="Condition1")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('Condition1',axis=1,inplace=True)
    status("Condition1")
    return dummies

def preprocess_condition2():
    global dataframe
    dummies = pd.get_dummies(dataframe["Condition2"],prefix="Condition2")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('Condition2',axis=1,inplace=True)
    status("Condition2")
    return dummies


def preprocess_bldgtype():
    global dataframe
    dummies = pd.get_dummies(dataframe["BldgType"],prefix="BldgType")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('BldgType',axis=1,inplace=True)
    status("BldgType")
    return dummies

def preprocess_housestyle():
    global dataframe
    dummies = pd.get_dummies(dataframe["HouseStyle"],prefix="HouseStyle")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('HouseStyle',axis=1,inplace=True)
    status("HouseStyle")
    return dummies


def preprocess_roofstyle():
    global dataframe
    dummies = pd.get_dummies(dataframe["RoofStyle"],prefix="RoofStyle")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('RoofStyle',axis=1,inplace=True)
    status("RoofStyle")
    return dummies


def preprocess_roofmatl():
    global dataframe
    dummies = pd.get_dummies(dataframe["RoofMatl"],prefix="RoofMatl")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('RoofMatl',axis=1,inplace=True)
    status("RoofMatl")
    return dummies

def preprocess_masvnrtype():
    global dataframe
    dummies = pd.get_dummies(dataframe["MasVnrType"],prefix="MasVnrType")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('MasVnrType',axis=1,inplace=True)
    status("MasVnrType")
    return dummies

def preprocess_masvnrarea():
    global dataframe
    dataframe.fillna({"MasVnrArea":dataframe["MasVnrArea"].median()})
    status("MasVnrArea")


def preprocess_exterqual():
    global dataframe
    dummies = pd.get_dummies(dataframe["ExterQual"],prefix="ExterQual")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('ExterQual',axis=1,inplace=True)
    status("ExterQual")
    return dummies

def preprocess_extercond():
    global dataframe
    dummies = pd.get_dummies(dataframe["ExterCond"],prefix="ExterCond")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('ExterCond',axis=1,inplace=True)
    status("ExterCond")
    return dummies


def preprocess_foundation():
    global dataframe
    dummies = pd.get_dummies(dataframe["Foundation"],prefix="Foundation")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('Foundation',axis=1,inplace=True)
    status("Foundation")
    return dummies

def preprocess_bsmtqual():
    global dataframe
    dummies = pd.get_dummies(dataframe["BsmtQual"],prefix="BsmtQual")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('BsmtQual',axis=1,inplace=True)
    status("BsmtQual")
    return dummies

def preprocess_bsmtcond():
    global dataframe
    dummies = pd.get_dummies(dataframe["BsmtCond"],prefix="BsmtCond")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('BsmtCond',axis=1,inplace=True)
    status("BsmtCond")
    return dummies

def preprocess_bsmtexposure():
    global dataframe
    dummies = pd.get_dummies(dataframe["BsmtExposure"],prefix="BsmtExposure")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('BsmtExposure',axis=1,inplace=True)
    status("BsmtExposure")
    return dummies



def preprocess_exterior1st2nd():
    global dataframe
    dataframe.fillna({"Exterior1st":"VinylSd"},inplace=True)
    dataframe.fillna({"Exterior2nd": "VinylSd"}, inplace=True)
    status("Exterior1st")
    status("Exterior2nd")

def preprocess_bsmtfintype1and2():
    global dataframe
    dataframe.fillna({"BsmtFinType1": "Unf"}, inplace=True)
    dataframe.fillna({"BsmtFinType2": "Unf"}, inplace=True)
    dummies1 = pd.get_dummies(dataframe.BsmtFinType1,prefix='BsmtFinType1')
    dummies2= pd.get_dummies(dataframe.BsmtFinType2,prefix='BsmtFinType2')
    dummies = pd.concat([dummies1,dummies2],axis=1)
    status("BsmtFinType1")
    status("BsmtFinType2")
    return dummies

def preprocess_bsmt_finish_type1_type2_unf():
    global dataframe
    dataframe.fillna({'BsmtFinSF1':0,'BsmtFinSF2':0,'BsmtUnfSF':0},inplace=True)
    status("BsmtFinSF1")
    status("BsmtFinSF2")
    status("BsmtUnfSF")



def preprocess_heating():
    global dataframe
    dummies = pd.get_dummies(dataframe["Heating"],prefix="Heating")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('Heating',axis=1,inplace=True)
    status("Heating")
    return dummies

def preprocess_heating_qc():
    global dataframe
    dummies = pd.get_dummies(dataframe["HeatingQC"],prefix="HeatingQC")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('HeatingQC',axis=1,inplace=True)
    status("HeatingQC")
    return dummies

def preprocess_central_air():
    global dataframe
    dummies = pd.get_dummies(dataframe["CentralAir"],prefix="CentralAir")
    #dataframe = pd.concat([dataframe,dummies],axis=1)
    dataframe.drop('CentralAir',axis=1,inplace=True)
    status("CentralAir")
    return dummies

def preprocess_electrical():
    global dataframe
    dataframe.fillna({"Electrical":"Sbrkr"},inplace=True)
    status("Electrical")

def preprocess_bsmt_full_half_bth():
    global dataframe
    dataframe.fillna({'BsmtFullBath':0,'BsmtHalfBath':0},inplace=True)
    status("BsmtFullBath")
    status("BsmtHalfBath")

def preprocess_kitchen_qual():
    global dataframe
    dataframe.fillna({'KitchenQual':"TA"})
    status("KitchenQual")


def preprocess_functional():
    global dataframe
    dataframe.fillna({'Functional':"Typ"})
    status("Functional")

def preprocess_fireplaceQu():
    global dataframe
    dataframe["FireplaceQu"].
def preprocess_dataset():

    #print(train.isnull().any())
    #print(test.isnull().any())
    global dataframe
    preprocess_MsZoning()
    preprocess_lot_frontage()
    preprocess_alley()
    df1=preprocess_lotshape()
    df2=preprocess_land_contour()
    df3=preprocess_utilities()
    df4=preprocess_lotconfig()
    df5=preprocess_landslope()
    df6=preprocess_neighborhood()
    df7=preprocess_condition1()
    df8=preprocess_condition2()
    df9=preprocess_bldgtype()
    df10=preprocess_housestyle()
    df11=preprocess_roofstyle()
    df12=preprocess_roofmatl()
    df13= preprocess_masvnrtype()
    df14=preprocess_exterqual()
    df15= preprocess_extercond()
    df16=preprocess_foundation()
    df17=preprocess_bsmtqual()
    df18=preprocess_bsmtcond()
    df19=preprocess_bsmtexposure()
    df20 = preprocess_bsmtfintype1and2()
    df21 = preprocess_heating()
    df22 = preprocess_heating_qc()
    df23 = preprocess_central_air()

    preprocess_exterior1st2nd()
    preprocess_masvnrarea()
    preprocess_bsmt_finish_type1_type2_unf()
    preprocess_electrical()
    preprocess_kitchen_qual()
    preprocess_functional()
    preprocess_fireplaceQu()
    dataframe = pd.concat([dataframe,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23],axis=1)
    contains_nan(dataframe.BsmtFinSF1)
    print(dataframe.shape)

    #print(dataframe.isnull().any())

def main():
    global dataframe
    train = pd.read_csv('train.csv',index_col= 'Id')
    test = pd.read_csv('test.csv',index_col='Id')
    dataframe = pd.concat([train, test])
    print(dataframe.shape)
    preprocess_dataset()
if __name__=='__main__':
    main()