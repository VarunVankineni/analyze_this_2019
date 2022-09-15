# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:47:25 2018

@author: Varun
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report as cr
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from scipy.stats import boxcox, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import matplotlib.pyplot as plt
from scipy.special import expit

folder = "E:/Acads/9th sem/Analyze This 2018/"
df = pd.read_csv(folder+"Training_dataset_Original.csv", index_col = 0)
columns = pd.read_csv(folder+"Data_Dictionary.csv")
df.columns = columns.iloc[1:,1]
df.replace(to_replace=["missing","na"], value=np.nan, inplace=True)
dum = pd.get_dummies(df.iloc[:,-2], prefix = df.columns[-2]).iloc[:,0]
df.drop(df.columns[-2], axis =1,  inplace =True)
df = pd.concat([df.iloc[:,:-1],dum,df.iloc[:,-1]],axis = 1)
df = df.astype("float64")


dft = pd.read_csv(folder+"Leaderboard_dataset.csv", index_col = 0)
dft.columns = columns.iloc[1:-1,1]
dft.replace(to_replace=["missing","na"], value=np.nan, inplace=True)
dumt = pd.get_dummies(dft.iloc[:,-1], prefix = dft.columns[-1]).iloc[:,0]
dft.drop(dft.columns[-1], axis =1,  inplace =True)
dft = pd.concat([dft,dumt],axis = 1)
dft = dft.astype("float64")

df = dft.copy()

dfX = df.iloc[:,:-1]
dfnan = np.isnan(dfX).astype(int)
dftnan = np.isnan(dft).astype(int)
dfnan.columns = [s+"_nan" for s in dfnan.columns]


def linearize_features(df):
  dftr = df.copy()
  loglist = np.arange(1,4).tolist()+np.arange(5,11).tolist()+[12,14,27,31,32]
  signlog = [4]
  gauss1_2 = [11]
  gauss0_9 = [13]
  sroot = np.arange(15,18).tolist()+ np.arange(24,27).tolist() + [28,29] + np.arange(33,39).tolist() + [42,45]
  croot = [39]
  square = [40]
  sign = [44]
  
  cols = dftr.columns.tolist()
  for i in range(len(cols)):
    axs = df[cols[i]]
    if(i in loglist):
      dftr[cols[i]] = np.log(axs+0.00001) 
    elif(i in signlog):
      dftr[cols[i]] = np.sign(np.log(axs+0.00001)+0.5)+1
    elif(i in gauss1_2):
      axs = np.log(axs+0.00001)
      axs = axs - np.mean(axs) + 1.2
      dftr[cols[i]] = np.abs(axs)**0.25
    elif(i in gauss0_9):
      axs = np.log(axs+0.00001)
      axs = axs - np.mean(axs) + 0.9
      dftr[cols[i]] = np.abs(axs)**0.25
    elif(i in sroot):
      dftr[cols[i]] = axs**0.5
    elif(i in croot):
      axs = axs -100
      dftr[cols[i]] = np.sign(axs)*(np.abs(axs)**(1/3))
    elif(i in square):
      dftr[cols[i]] = axs**2
    elif(i in square):
      dftr[cols[i]] = (np.sign(axs - 0.5)+1)//2
    elif(i in sign):
      dftr[cols[i]] = (np.sign(axs - 0.5)+1)//2
  return dftr

ldf = linearize_features(df)
ldfX = ldf
ldfX.columns = [s+"_l" for s in ldfX.columns]
ldfnan = np.isnan(ldfX).astype(int)
ldfnan.columns = [s+"_lnan" for s in ldfnan.columns]
ldfmean = ldfX.fillna(ldfX.mean())
ldfmean.columns = [s+"_lmean" for s in ldfmean.columns]
ldfzero = ldfX.fillna(0)
ldfzero.columns = [s+"_lzero" for s in ldfzero.columns]
ldfb = pd.concat([ldfX,ldfnan,ldfmean,ldfzero], axis = 1)

Bdfb = pd.concat([dfb,ldfb],axis =1)
dfy = df.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(dfb, dfy, test_size=0.4)
reg = XGBClassifier(booster = 'gbtree',n_estimators = 200, max_depth = 7)
reg.fit(np.array(X_train),y_train)
accuracy = reg.score(np.array(X_test),y_test)
haha = cr(y_test,reg.predict(np.array(X_test)))
imp_features1 = pd.Series(reg.feature_importances_, index = dfb.columns)

X_train, X_test, y_train, y_test = train_test_split(ldfb, dfy, test_size=0.4)
reg1 = XGBClassifier(booster = 'gbtree',n_estimators = 200, max_depth = 7)
reg1.fit(np.array(X_train),y_train)
accuracy = reg1.score(np.array(X_test),y_test)
haha2 = cr(y_test,reg1.predict(np.array(X_test)))
imp_features2 = pd.Series(reg1.feature_importances_, index = ldfb.columns)

X_train, X_test, y_train, y_test = train_test_split(Bdfb, dfy, test_size=0.4)
reg2 = XGBClassifier(booster = 'gbtree',n_estimators = 200, max_depth = 7)
reg2.fit(np.array(X_train),y_train)
accuracy = reg2.score(np.array(X_test),y_test)
haha3 = cr(y_test,reg2.predict(np.array(X_test)))
imp_features3 = pd.Series(reg2.feature_importances_, index = Bdfb.columns)
newdf = Bdfb[imp_featuresx.index]
newdft = Bdfb[imp_featuresx.index]
core = newdf.corr()
sns.heatmap(core, cmap = sns.color_palette("Blues"))

features = imp_featuresx.sort_values().index.tolist()
df = newdf[features]
for i in range(3):
  X_train, X_test, y_train, y_test = train_test_split(df, dfy, test_size=0.4)
  reg2 = XGBClassifier(booster = 'gbtree',n_estimators = 300, max_depth = 7)
  reg2.fit(np.array(X_train),y_train)
  accuracy = reg2.score(np.array(X_test),y_test)
  features= pd.Series(reg2.feature_importances_, index = df.columns).sort_values().index.tolist()
  df = df[features[1:]]
  print(accuracy,features[0])
goodf = df.columns.tolist()+ ["Type of product that the applicant applied for. (C = Charge; L = Lending)_C",
                         "Number of credit cards with an active tenure of at least 2 years",
                 "Number of days since last missed payment on any credit line",
                 "Number of active credit cards on which atleast 75% credit limit is utilized by the borrower"]
goodf = newdf.columns
finaldf = newdf[goodf]
finaldft = newdft[goodf]

X_train, X_test, y_train, y_test = train_test_split(finaldf, dfy, test_size=0.75)
reg2 = XGBClassifier(booster = 'gbtree',n_estimators = 300, max_depth = 7)
reg2.fit(np.array(X_train),y_train)
accuracy = reg2.score(np.array(X_test),y_test)

core = np.abs(finaldf.corr())
sns.heatmap(core, cmap = sns.color_palette("Blues"))

todrop = ["Maximum of credit available on all active credit lines (in $)",
 "Number of auto loans on which the borrower has missed 2 payments",
 "Number of active revolving credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit lines on which atleast 75% credit limit is utilized by the borrower",
 "Average utilization of line on all active credit cards activated in last 1 year (%)",
 "Tenure of oldest revolving credit card among all active revolving credit cards (in days)",
 "Minimum of credit available on all revolving credit cards (in $)",
 "Number of credit cards with an active tenure of at least 2 years"]

goodf = newdf.columns.tolist()
goodf = set(goodf) - set(todrop)
goodf = list(goodf)

df = newdf[goodf]
dft = newdft[goodf]

dfX = df.iloc[:,:-1]
dfy = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size = 0.4)
reg = XGBClassifier(n_estimators = 300, max_depth = 7, n_jobs = 4)
reg.fit(np.array(dfX),dfy)
accuracy = reg.score(np.array(X_test),y_test)






