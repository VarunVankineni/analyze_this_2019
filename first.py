# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 22:40:30 2018

@author: Varun
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report as cr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import matplotlib.pyplot as plt

folder = "E:/Acads/9th sem/Analyze This 2018/"
df = pd.read_csv(folder+"Training_dataset_Original.csv", index_col = 0)
dft = pd.read_csv(folder+"Leaderboard_dataset.csv", index_col = 0)
columns = pd.read_csv(folder+"Data_Dictionary.csv")
df.columns = columns.iloc[1:,1]
dft.columns = columns.iloc[1:-1,1]
dtlist = df.dtypes

df.replace(to_replace=["missing","na"], value=np.nan, inplace=True)
dft.replace(to_replace=["missing","na"], value=np.nan, inplace=True)
dum = pd.get_dummies(df.iloc[:,-2], prefix = df.columns[-2]).iloc[:,0]
dumt = pd.get_dummies(dft.iloc[:,-1], prefix = dft.columns[-1]).iloc[:,0]
df.drop(df.columns[-2], axis =1,  inplace =True)
dft.drop(dft.columns[-1], axis =1,  inplace =True)
df = pd.concat([df.iloc[:,:-1],dum,df.iloc[:,-1]],axis = 1)
dft = pd.concat([dft,dumt],axis = 1)

df = df.astype("float64")
dft = dft.astype("float64")
core = df.corr()

nanlist = pd.Series([sum(np.isnan(df[x])) for x in df.columns], index = df.columns)
pivots = [pd.concat([df[~np.isnan(df[x])][x],df[~np.isnan(df[x])].iloc[:,-1]], axis =1) for x in df.columns]
setpivots = [pd.concat([df[~np.isnan(df[x])][x],df[~np.isnan(df[x])].iloc[:,-1]], axis =1) for x in df.columns]
for i in pivots:
  i["counter"] = 1
for i in setpivots:
  i["counter"] = 1
#qbins = [0 for x in range(len(pivots))]
#bins = qbins.copy()
#for i in range(len(pivots)):
#  qbins[i],bins[i] = pd.qcut(pivots[i].iloc[:,0], 10, duplicates='drop',retbins = True)
qbins = [pd.qcut(i.iloc[:,0], 10, duplicates='drop',retbins = True) for i in pivots]
cols = df.columns[:-2]

#dfcut = df.copy()
#df = dfcut.copy()
for i in range(len(cols)):
  df[cols[i]] = pd.cut(df[cols[i]], qbins[i][1])
for i in range(len(cols)):
  dft[cols[i]] = pd.cut(dft[cols[i]], qbins[i][1])

nanprob = [0 for i in cols]
for i in range(len(cols)):
  dum = df[np.isnan(df[cols[i]])]
  if(len(dum)!=0):
    nanprob[i] = sum(dum.iloc[:,-1])/len(dum)
  
dfdummies = pd.get_dummies(df)
dfd = dfdummies.drop(dfdummies.columns[1], axis =1)
dfdy = dfdummies.iloc[:,1]

for i in range(len(pivots)):   
  pivots[i].iloc[:,0] = qbins[i][0]
newpivots = [i.groupby([i.columns[0]]).sum() for i in pivots[:-1]]
percentpivots = [i.iloc[:,0]/i.iloc[:,1] for i in newpivots]
k = 0
k += 1
plt.plot(percentpivots[k].values)

dfnew = pd.DataFrame(np.zeros(df.shape), index = df.index, columns = df.columns)

for i in range(len(df)):
  for j in range(len(cols)):
    if(type(df.iloc[i,j]) == pd._libs.interval.Interval):
      dfnew.iat[i,j] = percentpivots[j][df.iat[i,j]]
    else:
      dfnew.iat[i,j] = np.nan

for i in range(len(cols)):
  dfnew[cols[i]] = dfnew[cols[i]].fillna(nanprob[i])

dftnew = pd.DataFrame(np.zeros(dft.shape), index = dft.index, columns = dft.columns)

for i in range(len(dft)):
  for j in range(len(cols)):
    if(type(dft.iloc[i,j]) == pd._libs.interval.Interval):
      dftnew.iat[i,j] = percentpivots[j][dft.iat[i,j]]
    else:
      dftnew.iat[i,j] = np.nan
for i in range(len(cols)):
  dftnew[cols[i]] = dftnew[cols[i]].fillna(nanprob[i])
dfnew = pd.concat([dfnew.iloc[:,:-2],df.iloc[:,-2:]],axis = 1)
#dfnew.iloc[:,1].to_csv(folder+"prob_data_train.csv")
#dftnew.iloc[:,1].to_csv(folder+"prob_data_test.csv")

#4,40,44,45,46
check = dft.iloc[:5,:]
risks = df[df.iloc[:,-3]==1]
nonrisk = df[df.iloc[:,-3]==0]
risksmean = risks.mean()
nonriskmean = nonrisk.mean()
comparer = pd.concat([nonriskmean,risksmean],axis=1)
dfy = df.iloc[:,-3]
dfX = df.drop(df.columns[-3],axis =1)
numDef = int(sum(dfy))


#dfX_D = dfX[dfy.apply(lambda x: bool(x))]
#dfX_N = dfX[dfy.apply(lambda x: bool(1-x))]
#dfX_C = pd.concat([dfX_D,dfX_N.iloc[:numDef,:]])
#dfy_C = dfy[dfX_C.index]

X_train, X_test, y_train, y_test = train_test_split(dfnew.iloc[:,:-1], dfnew.iloc[:,-1], test_size=0.4)

#param_grid = dict(n_estimators = [75,100],
#                  max_depth = [7])
##                  gamma = [0.05,0.01,0.5,0.9],
##                  learning_rate = [0.01,0.05,0.1],
##                  min_child_weight = [1,3,5,7],
##                  reg_alpha = [0.01,0.1,1],
##                  reg_lambda = [0,0.1,0.5,1])
#grid = GridSearchCV(reg, param_grid, verbose = 10000, n_jobs = 7)
#
#grid.fit(np.array(dfX), dfy)
#reg = LinearSVC()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg = XGBClassifier(max_depth = 7)
reg = LogisticRegression()

reg.fit(np.array(X_train),y_train)
accuracy = reg.score(np.array(X_test),y_test)

conf = reg.decision_function(np.array(X_test))
predprob = reg.predict_proba(np.array(X_test))
haha = cr(y_test,reg.predict(np.array(X_test)))
dft = dftnew
dft = scaler.transform(dft)
predprobt = reg.predict_proba(np.array(dft))
predprobt[:,0] = predprobt[:,0]
predprobt[:,1] = predprobt[:,1]

conft = reg.decision_function(np.array(dft))
predi = np.argmax(predprobt, axis =1)
premax = np.max(predprobt, axis =1)
pred = pd.DataFrame(premax, index = dft.index, columns = ["probability"])
table = pd.concat([pred,pd.DataFrame(predi, index = dft.index, columns = ["prediction"])],axis = 1)
table.sort_values("probability", ascending = False, inplace = True)

#pred1 = pd.DataFrame(predprobt[:,0], index = dft.index, columns = ["probability"])
#table1 = pd.concat([pred1,pd.DataFrame(predi, index = dft.index, columns = ["prediction"])],axis = 1)
#table1.sort_values("probability", ascending = False, inplace = True)
#np.sum(table.iloc[:10000,1].values == table1.iloc[:10000,1].values)

table.iloc[:,1].to_csv(folder+"result2.csv", header = False)
tablecross = pd.read_csv(folder+"Kaushalarmy_IITM_2_best.csv", index_col = 0, names = ["prediction"])
set1 = sorted(tablecross.index.values[:10000])
set2 = sorted(table.index.values[:10000])
setnull = [x for x in set1 if x in set2]






