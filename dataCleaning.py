# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:16:58 2021

@author: Francesco Di Flumeri

Data cleaning and preparation
The data balanced was not performed because the occurences of the classes are written in the file .names
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.impute import KNNImputer
import os

"""reading raw data and saving them in pandas datsets"""
dirname = os.path.dirname(__file__)

cleveland_filename = os.path.join(dirname, 'cleveland.csv')
hungarian_filename = os.path.join(dirname, 'hungarian.csv')
longBeach_filename = os.path.join(dirname, 'long-beach-va.csv')
switzerland_filename = os.path.join(dirname, 'switzerland.csv')

cleveland_data = pd.read_csv(cleveland_filename, sep=',',  error_bad_lines=False, encoding='iso-8859-1')
hungarian_data = pd.read_csv(hungarian_filename, sep=',',  error_bad_lines=False, encoding='iso-8859-1')
longBeach_data = pd.read_csv(longBeach_filename, sep=',',  error_bad_lines=False, encoding='iso-8859-1')
switzerland_data = pd.read_csv(switzerland_filename, sep=',',  error_bad_lines=False, encoding='iso-8859-1')

"""all the data were without header so I needed to redefine them"""
columns_names = ["id","ccf","age","sex","painloc","painexer","relrest","pncaden",
                   "cp","trestbps","htn","chol","smoke","cigs","years","fbs","dm",
                   "famhist","restecg","ekgmo","ekgday","ekgyr","dig","prop","nitr",
                   "pro","diuretic","proto","thaldur","thaltime","met","thalach",
                   "thalrest","tpeakbps","tpeakbpd","dummy","trestbpd","exang",
                   "xhypo","oldpeak","slope","rldv5","rldv5e","ca","restckm","exerckm",
                   "restef","restwm","exeref","exerwm","thal","thalsev","thalpul",
                   "earlobe","cmo","cday","cyr","num","lmt","ladprox","laddist","diag",
                   "cxmain","ramus","om1","om2","rcaprox","rcadist","lvx1","lvx2","lvx3",
                   "lvx4","lvf","cathef","junk","name"]

df_cleveland = pd.DataFrame(columns = columns_names)
df_hungarian = pd.DataFrame(columns = columns_names)
df_longBeach = pd.DataFrame(columns = columns_names)
df_switzerland = pd.DataFrame(columns = columns_names)

first_values_cleveland = list(cleveland_data.columns)
first_values_hungarian = list(hungarian_data.columns)
first_values_longBeach = list(longBeach_data.columns)
first_values_switzerland = list(switzerland_data.columns)

cleveland_data_list = cleveland_data.values.tolist()
hungarian_data_list = hungarian_data.values.tolist()
longBeach_data_list = longBeach_data.values.tolist()
switzerland_data_list = switzerland_data.values.tolist()

cleveland_list = [first_values_cleveland] + cleveland_data_list
hungarian_list = [first_values_hungarian] + hungarian_data_list
longBeach_list = [first_values_longBeach] + longBeach_data_list
switzerland_list = [first_values_switzerland] + switzerland_data_list

def buildDataFrame(startList, df):
    record = []
    for index in range(len(startList)):
        if((index+1)%10==0):
            values = list(startList[index][0].split(";"))
            record = record + values[0:5]
            del record[7]
            df.loc[len(df)] = record
            record = []
        else:
            values = list(startList[index][0].split(";"))
            record = record + values
    return df
#building of the dataset with headers and columns
buildDataFrame(cleveland_list, df_cleveland)
buildDataFrame(hungarian_list, df_hungarian)
buildDataFrame(longBeach_list, df_longBeach)
buildDataFrame(switzerland_list, df_switzerland)
"""dropping corrupting data, they were detected only in the cleveland dataset""" 
df_cleveland = df_cleveland[:-16]
"""changing data type and replace the -9 with nan as it is stated in the .names file"""
def convertType(df):
    cols = df.columns
    for c in cols:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass

convertType(df_cleveland)
convertType(df_hungarian)
convertType(df_longBeach)
convertType(df_switzerland)

df_cleveland = df_cleveland.replace(-9, np.nan)
df_hungarian = df_hungarian.replace(-9, np.nan)
df_longBeach = df_longBeach.replace(-9, np.nan)
df_switzerland = df_switzerland.replace(-9, np.nan)
"""dropping the columns containing all nan values"""
df_cleveland = df_cleveland.dropna(axis=1, how="all")
df_hungarian = df_hungarian.dropna(axis=1, how="all")
df_longBeach = df_longBeach.dropna(axis=1, how="all")
df_switzerland = df_switzerland.dropna(axis=1, how="all")
"""in the column 'dm' all nan values have been considered equal to 0 since the type of this column is boolean"""
df_cleveland["dm"] = df_cleveland["dm"].replace(np.nan, 0)
df_hungarian["dm"] = df_hungarian["dm"].replace(np.nan, 0)
df_longBeach["dm"] = df_longBeach["dm"].replace(np.nan, 0)
df_switzerland["dm"] = df_switzerland["dm"].replace(np.nan, 0)
"""selecting the columns still containing some nan values and replacing them using KNN algorithm"""
cleveland_nancols = df_cleveland.loc[:, df_cleveland.isna().any()]
hungarian_nancols = df_hungarian.loc[:, df_hungarian.isna().any()]
longBeach_nancols = df_longBeach.loc[:, df_longBeach.isna().any()]
switzerland_nancols = df_switzerland.loc[:, df_switzerland.isna().any()]

cleveland_nancols_list = cleveland_nancols.values.tolist()
hungarian_nancols_list = hungarian_nancols.values.tolist()
longBeach_nancols_list = longBeach_nancols.values.tolist()
switzerland_nancols_list = switzerland_nancols.values.tolist()

imputer = KNNImputer(n_neighbors=2)

cleveland_nanrecovered = imputer.fit_transform(cleveland_nancols_list)
hungarian_nanrecovered = imputer.fit_transform(hungarian_nancols_list)
longBeach_nanrecovered = imputer.fit_transform(longBeach_nancols_list)
switzerland_nanrecovered = imputer.fit_transform(switzerland_nancols_list)

def fillNanValues(df, arrayFull):
    cols = df.columns
    cont = 0
    for c in cols:
        if df[c].isnull().values.any():
            df[c] = arrayFull[cont]
            cont = cont + 1
            
fillNanValues(df_cleveland, cleveland_nanrecovered.transpose())
fillNanValues(df_hungarian, hungarian_nanrecovered.transpose())
fillNanValues(df_longBeach, longBeach_nanrecovered.transpose())
fillNanValues(df_switzerland, switzerland_nanrecovered.transpose())
"""dropping duplicates records"""
df_cleveland = df_cleveland.drop_duplicates()
df_switzerland = df_switzerland.drop_duplicates()
df_hungarian = df_hungarian.drop_duplicates()
df_longBeach = df_longBeach.drop_duplicates()
"""dropping column name beacause it does not contain any information"""
df_cleveland = df_cleveland.drop(columns=['name'])
df_switzerland = df_switzerland.drop(columns=['name'])
df_hungarian = df_hungarian.drop(columns=['name'])
df_longBeach = df_longBeach.drop(columns=['name'])
"""writing csv files"""
df_cleveland.to_csv('clevelandFinal.csv', index=False)
df_switzerland.to_csv('switzerlandFinal.csv', index=False)
df_hungarian.to_csv('hungarianFinal.csv', index=False)
df_longBeach.to_csv('longBeachFinal.csv', index=False)


"""for reading data from .csv file you should use this line of code in order to have the dataset in a nice format
#cleveland_data = pd.read_csv(cleveland_filename, sep=',',  error_bad_lines=False, encoding='iso-8859-1')"""