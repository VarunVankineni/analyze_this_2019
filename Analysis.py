# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:49:54 2018

@author: Varun
"""


import numpy as np
import pandas as pd
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
core = df.corr()

pivots = [pd.concat([df[~np.isnan(df[x])][x],df[~np.isnan(df[x])].iloc[:,-1]], axis =1) for x in df.columns]
pivots = pivots[:-1]
for i in pivots:
  i["counter"] = 1
newpivots = [i.groupby([i.columns[0]]).sum() for i in pivots]
p_pivots = [i.iloc[:,0]/i.iloc[:,1] for i in newpivots]

qbins = [pd.qcut(pivots[i].iloc[:,0], max(1,min(len(newpivots[i])//10,100)), duplicates='drop',retbins = True) for i in range(len(pivots))]
ebins = [pd.cut(pivots[i].iloc[:,0], max(1,min(len(newpivots[i])//10,100)), duplicates='drop',retbins = True) for i in range(len(pivots))]

for i in range(len(pivots)):   
  pivots[i].iloc[:,0] = qbins[i][0]
gq_pivots = [i.groupby([i.columns[0]]).sum() for i in pivots]
pq_pivots = [i.iloc[:,0]/i.iloc[:,1] for i in gq_pivots]

for i in range(len(pivots)):   
  pivots[i].iloc[:,0] = ebins[i][0]
ge_pivots = [i.groupby([i.columns[0]]).sum() for i in pivots]
pe_pivots = [i.iloc[:,0]/i.iloc[:,1] for i in ge_pivots]

k = 28
k+=1
plt.plot([x**0.5 for x in p_pivots[k].index],p_pivots[k])
plt.plot([x for x in p_pivots[k].index],p_pivots[k])
k+=1

vals = pq_pivots[k].values
axs = np.array([(x.right+ x.left) / 2 for x in pq_pivots[k].index])
axs = expit(axs)
axs = scale(axs)
axs = axs - 100
axs = axs/140
axs = axs**0.05
axs = np.sign(axs)*(np.abs(axs)**(1/3))
axs = np.abs(scale(np.log(axs)) +0.9)**0.25
axs = np.sign(np.log(axs)+0.1)+1
axs = np.log(axs)
plt.style.use('ggplot')
plt.xlabel(columns.iloc[k,1])
plt.ylabel(columns.iloc[-1,1])
plt.plot(axs,vals)
plt.plot(np.abs((axs +1.2))**0.25,vals)
plt.plot(scale(pq_pivots[k].values))
plt.plot(boxcox(np.log(axs)[0])
plt.plot(pe_pivots[k].values)
plt.plot(pq_pivots[k].values**2)
plt.plot(pq_pivots[k].values**0.5)
plt.plot(1 - np.exp(-pe_pivots[k].values))
plt.plot(np.exp(pe_pivots[k].values))
plt.plot(np.log(pq_pivots[k].values))
plt.plot(-np.log(pe_pivots[k].values))
"""
qbins
0, 18:23 ,30,39,41,43,46   = nothing
1:3, 5:10, 12,14,27,31:32   = np.log(axs)
4    = np.sign(np.log(axs)+0.5)+1
11   = np.abs(scale(np.log(axs)) +1.2)**0.25
13   = np.abs(scale(np.log(axs)) +0.9)**0.25
15:17,24:26,28:29,33:38, 42,45   = x**0.5
39   = axs-100 -> np.sign(axs)*(np.abs(axs)**(1/3))
40 = x**2
44 = (np.sign(x - 0.5)+1)//2
"""
loglist = np.arange(1,4).tolist()+np.arange(5,11).tolist()+[12,14,27,31,32]
signlog = [4]
gauss1_2 = [11]
gauss0_9 = [13]
sroot = np.arange(15,18).tolist()+ np.arange(24,27).tolist() + [28,29] + np.arange(33,39).tolist() + [42,45]
croot = [39]
square = [40]
sign = [44]

cols = df.columns.tolist()[:-1]
dftr = df.copy()
dfsave = df.copy()
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
    

dftrsave =dftr.copy()
todrop = ["Maximum of credit available on all active credit lines (in $)",
 "Number of auto loans on which the borrower has missed 2 payments",
 "Number of active revolving credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit cards on which full credit limit is utilized by the borrower",
 "Number of active credit lines on which atleast 75% credit limit is utilized by the borrower",
 "Average utilization of line on all active credit cards activated in last 1 year (%)",
 "Tenure of oldest revolving credit card among all active revolving credit cards (in days)"]

dftr.drop(todrop, axis =1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(dftr.iloc[:,:-1], dftr.iloc[:,-1], test_size=0.4)
reg = XGBClassifier()
reg.fit(np.array(X_train),y_train)
accuracy = reg.score(np.array(X_test),y_test)



dft = pd.read_csv(folder+"Leaderboard_dataset.csv", index_col = 0)
dft.columns = columns.iloc[1:-1,1]
dft.replace(to_replace=["missing","na"], value=np.nan, inplace=True)
dumt = pd.get_dummies(dft.iloc[:,-1], prefix = dft.columns[-1]).iloc[:,0]
dft.drop(dft.columns[-1], axis =1,  inplace =True)
dft = pd.concat([dft,dumt],axis = 1)
dft = dft.astype("float64")
dftr = dft.copy()
df = dft.copy()
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
    
dft = dftr.copy()
dft.drop(todrop, axis =1, inplace = True)
nlnlist = pd.Series([sum(np.isnan(dft[x])) for x in dft.columns], index = dft.columns)
bestfs = pd.Series(reg.feature_importances_ , index = dftr.columns[:-1])
bestfs = bestfs.sort_values(ascending = False)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
dft = scaler.transform(dft)
reg = KNeighborsClassifier()



for i in range(1,45):
  selfs = bestfs[:i].index
  seldftr = dftr[selfs]
  X_train, X_test, y_train, y_test = train_test_split(seldftr.iloc[:,:-1], dftr.iloc[:,-1], test_size=0.4)
  reg = XGBClassifier(n_jobs = 7)
  reg.fit(np.array(X_train),y_train)
  accuracy = reg.score(np.array(X_test),y_test)
  print(accuracy)

haha = cr(y_test,reg.predict(np.array(X_test)))
predprobt = reg.predict_proba(np.array(dft))
predi = np.argmax(predprobt, axis =1)
premax = np.max(predprobt, axis =1)
pred = pd.DataFrame(premax, index = dft.index, columns = ["probability"])
table = pd.concat([pred,pd.DataFrame(predi, index = dft.index, columns = ["prediction"])],axis = 1)

defaults = table.iloc[:,1]==1
nines = table.iloc[:,0]>0.90
changes = defaults & nines
for i in range(len(table)):
  if(changes.iat[i] == True):
    table.iat[i,0] = 1 - table.iat[i,0] 
    table.iat[i,1] = 0

table.sort_values("probability", ascending = False, inplace = True)
table.to_csv(folder+"38ColPob.csv", header = False)







