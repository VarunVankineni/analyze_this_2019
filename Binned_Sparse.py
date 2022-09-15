# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 08:29:47 2018

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
heat = ["Number of active revolving credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit lines on which atleast 75% credit limit is utilized by the borrower"]
core = df.corr()
todrop = ["Maximum of credit available on all active credit lines (in $)",
 "Number of auto loans on which the borrower has missed 2 payments",
 "Number of active revolving credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit lines on which atleast 75% credit limit is utilized by the borrower",
 "Average utilization of line on all active credit cards activated in last 1 year (%)",
 "Tenure of oldest revolving credit card among all active revolving credit cards (in days)",
 "Minimum of credit available on all revolving credit cards (in $)",
 "Number of credit cards with an active tenure of at least 2 years"]
df.drop(todrop, axis =1, inplace = True)
nanlist = pd.Series([sum(np.isnan(df[x])) for x in df.columns], index = df.columns)
nany = [df[np.isnan(df[x])].iloc[:,-1] for x in df.columns]
valy = [df[~np.isnan(df[x])].iloc[:,-1] for x in df.columns]
val0 = [df[df[x]==0].iloc[:,-1] for x in df.columns]
nanmeans = pd.Series([sum(x)/len(x)  if len(x)!=0 else 0 for x in nany], index = df.columns)
valmeans = pd.Series([sum(x)/len(x)  if len(x)!=0 else 0 for x in valy], index = df.columns)
zeromeans = pd.Series([sum(x)/len(x)  if len(x)!=0 else 0 for x in val0], index = df.columns)
compare= pd.concat([nanlist,nanmeans,valmeans,zeromeans], axis = 1)
compare.columns = ["nan_count","nan","val","zero"]
core = np.abs(df.corr())
plt.figure(figsize = (6,6))
sns.heatmap(core, cmap = sns.color_palette("Blues"))



fillzero = ["Severity of default by the borrower on auto loan(s). Severity is a function of amount, time since default and number of defaults",
            "Severity of default by the borrower on education loan(s). Severity is a function of amount, time since default and number of defaults",
            "Severity of default by the borrower on any loan(s). Severity is a function of amount, time since default and number of defaults",
            "Annual amount paid towards all credit cards during the previous year (in $)",
            "Number of active credit lines over the last 6 months on which the borrower has missed 1 payment",
            "Number of active credit lines",
            "Number of credit lines activated in last 2 years"]
medianfill = ["Number of active credit cards on which atleast 75% credit limit is utilized by the borrower",
              "Financial stress index of the borrower. This index is a function of collection trades, bankruptcies files, tax liens invoked, etc. ",
              "Number of credit lines on which the borrower has never missed a payment in last 2 yrs, yet considered as high risk loans based on market prediction of economic scenario"]
nanbool = ["Sum of available credit on credit cards that the borrower has missed 1 payment (in $)",
           "Maximum of credit available on all active revolving credit cards (in $)",
           "Total amount of credit available on accepted credit lines (in $)",
           "Amount of dues collected post default where due amount was more than 500 (in $)",
           "Sum of amount due on active credit cards (in $)",
           "Estimated market value of a properety owned/used by the borrower (in $)",
           "Number of active credit lines on which full credit limit is utilized by the borrower",
           "Average utilization of active revolving credit card loans (%)",
           "Average utilization of line on all active credit lines activated in last 2 years (%)",
           "Average utilization of line on credit cards on which the borrower has missed 1 payment during last 6 months (%)",
           "Average tenure of active revolving credit cards (in days)",
           "Tenure of oldest credit card among all active credit cards (in days)",
           "Number of days since last missed payment on any credit line",
           "Tenure of oldest credit line (in days)",
           "Maximum tenure on all auto loans (in days)",
           "Maximum tenure on all education loans (in days)",
           "Sum of tenures (in months) of active credit cards",
           "Duration of stay at the current residential address (in years)",
           "Number of revolving credit cards over the last 2 years on which the borrower has missed 1 payment",
           "Number of credit lines on which the borrower has current delinquency",
           "Utilization of line on active education loans (%)",
           "Utilization of line on active auto loans (%)",
           "Ratio of maximum amount due on all active credit lines and sum of amounts due on all active credit lines ",
           "Number of mortgage loans on which the borrower has missed 2 payments"]


for x in nanbool:
  df[x+"_nan"] = np.isnan(df[x]).astype(int)

df[fillzero] = df[fillzero].fillna(0)
df[medianfill] = df[medianfill].fillna(df[medianfill].mean())
dfi = df.copy()

dropcols = ["Number of days since last missed payment on any credit line_nan",
 "Tenure of oldest credit line (in days)_nan",
 "Sum of tenures (in months) of active credit cards_nan"]

df.drop(dropcols, axis =1, inplace =True)
core = np.abs(df.corr())
sns.heatmap(core, cmap = sns.color_palette("Blues"))

#LinearSVC LogisticRegression LinearDiscriminantAnalysis KNeighborsClassifier
df.to_csv(folder+"lin_nasep_train.csv")

dfX = df.drop("Indicator for default",axis =1)
dfy = df["Indicator for default"]

X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=0.4)
reg = KNeighborsClassifier(7)#XGBClassifier(booster = 'gbtree',n_estimators = 200, max_depth = 7)
reg.fit(np.array(X_train),y_train)
accuracy = reg.score(np.array(X_test),y_test)
haha = cr(y_test,reg.predict(np.array(X_test)))
