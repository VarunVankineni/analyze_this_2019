# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 20:49:31 2018

@author: Varun
"""
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

plt.style.use('ggplot')

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

df = linearize_features(df)
dft = linearize_features(dft)

core = np.abs(df.corr())
sns.set_palette(sns.color_palette("Blues"))
plt.figure(figsize = (12,8))
sns.heatmap(core, cmap = sns.color_palette("Blues"))
plt.show()
nanlist = pd.Series([sum(np.isnan(df[x])) for x in df.columns], index = df.columns)
todrop = ["Maximum of credit available on all active credit lines (in $)",
 "Number of auto loans on which the borrower has missed 2 payments",
 "Number of active revolving credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit lines on which atleast 75% credit limit is utilized by the borrower",
 "Average utilization of line on all active credit cards activated in last 1 year (%)",
 "Tenure of oldest revolving credit card among all active revolving credit cards (in days)",
 "Minimum of credit available on all revolving credit cards (in $)"]


df.drop(todrop, axis =1, inplace = True)
dft.drop(todrop, axis =1, inplace = True)

df.to_csv(folder+"Linearized_reduced_train.csv")
dft.to_csv(folder+"Linearized_reduced_test.csv")

core = np.abs(df.corr())
sns.heatmap(core, cmap = sns.color_palette("Blues"))
dfC = df[df.iloc[:,-2]==1].drop(df.columns[-2],axis =1)
dfL = df[df.iloc[:,-2]==0].drop(df.columns[-2],axis =1)
Ccore =np.abs(dfC.corr())
Lcore =np.abs(dfL.corr())
sns.heatmap(Lcore, cmap = sns.color_palette("Blues"))


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.4)
reg = XGBClassifier(booster = 'gbtree',n_estimators = 200, max_depth = 7)
reg.fit(np.array(X_train),y_train)
accuracy = reg.score(np.array(X_test),y_test)
haha = cr(y_test,reg.predict(np.array(X_test)))
