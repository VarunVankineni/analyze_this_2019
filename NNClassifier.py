# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:01:28 2018

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

dfk = pd.read_csv(folder+"Imputed_training_dataset_knn.csv")
dfk.index = df.index
dfkt = pd.read_csv(folder+"Imputed_testing_dataset_knn.csv")
dfkt.index = dft.index


dfX = df.iloc[:,:-1]
dfy = df.iloc[:,-1]

dfnan = np.isnan(dfX).astype(int)
dfnan.columns = [col+"_n" for col in dfX.columns]
dftnan = np.isnan(dft).astype(int)
dftnan.columns = [col+"_n" for col in dft.columns]

dfke = pd.concat([dfk,dfnan],axis =1)
dfkte = pd.concat([dfkt,dftnan],axis =1)


X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size = 0.4)
reg = XGBClassifier(n_estimators = 300, max_depth = 7, n_jobs = 4)
reg.fit(dfX,dfy)
accuracy = reg.score(X_test,y_test)
features = pd.Series(reg.feature_importances_, index = dfke.columns)

predprobt = reg.predict_proba(dft)
predi = np.argmax(predprobt, axis =1)
premax = np.max(predprobt, axis =1)
pred = pd.DataFrame(np.c_[predprobt,predi], index = dft.index, columns = ["probability0","probability1","prediction"])
table = pd.concat([pred,pd.DataFrame(predi, index = dft.index, columns = ["prediction"])],axis = 1)
pred.to_csv(folder+"col38prob.csv")
defaults = table.iloc[:,1]==1
nines = table.iloc[:,0]>0.92
changes = defaults & nines
for i in range(len(table)):
  if(changes.iat[i] == True):
    table.iat[i,0] = 1 - table.iat[i,0] 
    table.iat[i,1] = 0
    
table.sort_values("probability", ascending = False, inplace = True)
table.iloc[:,1].to_csv(folder+"knn_nan.csv", header = False)



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
dft.drop(todrop, axis =1, inplace = True)

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


df[fillzero] = df[fillzero].fillna(0)
dft[fillzero] = dft[fillzero].fillna(0)
df[medianfill] = df[medianfill].fillna(df[medianfill].mean())
dft[medianfill] = dft[medianfill].fillna(df[medianfill].mean())

remcols = set(df.columns.tolist()) - set(fillzero)
remcols = list(remcols - set(medianfill))
for c in remcols:
  df[c+"_nan"] = np.isnan(df[c]).astype(int)
df[remcols] = df[remcols].fillna(df[remcols].mean())

remcols = set(dft.columns.tolist()) - set(fillzero)
remcols = list(remcols - set(medianfill))
for c in remcols:
  dft[c+"_nan"] = np.isnan(dft[c]).astype(int)
dft[remcols] = dft[remcols].fillna(df[remcols].mean())

dropcols = ["Number of days since last missed payment on any credit line_nan",
 "Tenure of oldest credit line (in days)_nan",
 "Sum of tenures (in months) of active credit cards_nan",
 "Annual income (in $)_nan",
 "Type of product that the applicant applied for. (C = Charge; L = Lending)_C_nan"]


df.drop(dropcols, axis =1, inplace =True)
dft.drop(dropcols, axis =1, inplace =True)
df.drop("Indicator for default_nan",axis =1, inplace= True)


X_index = dft.index.tolist()
X_led = np.array(dft)
tf.reset_default_graph()



dfX = df.drop("Indicator for default",axis =1)
dfy = df["Indicator for default"]


scaler = MinMaxScaler()
dfke = pd.DataFrame(scaler.fit_transform(dfke), index = dfke.index, columns = dfke.columns)
X_led = scaler.transform(dfkte)
X_train, X_test, y_train, y_test = train_test_split(dfke, dfy, test_size=0.2)

X_train = np.array(dfke)
y_train = np.array(dfy)
X_test = np.array(X_test)
y_test = np.array(y_test)
cols = dfke.shape[1]

x = tf.placeholder(tf.float32, (None, cols))
y = tf.placeholder(tf.int32, (None))




fcon1 = tf.layers.dense(x,128, tf.nn.relu ,tf.nn.relu, name = 'fcon1')
fcon2 = tf.layers.dense(fcon1,256, tf.nn.relu,name = 'fcon2')
fcon3 = tf.layers.dense(fcon2,256, tf.nn.relu,name = 'fcon3')
fcon4 = tf.layers.dense(fcon3,128, tf.nn.relu,name = 'fcon4')
logits = tf.layers.dense(fcon4,2, activation = 'sigmoid')
a = 0.001

onehoty = tf.one_hot(y, 2)
loss = tf.losses.softmax_cross_entropy(onehoty, logits)
algo = tf.train.GradientDescentOptimizer(a).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) 
maxitr =3
bs = 1
runs = X_train.shape[0]//bs
for i in range(maxitr):
  for j in range(runs):
    cur = j*bs
    #print(sess.run(yp,feed_dict = {x:X_train[cur:cur+bs,:].reshape((bs,cols)),y:y_train[cur:cur+bs].reshape((bs))}))
    sess.run(algo,feed_dict = {x:X_train[cur:cur+bs,:].reshape((bs,cols)),y:y_train[cur:cur+bs].reshape((bs))})
    
true = sess.run(onehoty,feed_dict = {y:y_train})
pred = sess.run(logits,feed_dict = {x:X_train})
truet = sess.run(onehoty,feed_dict = {y:y_test})
predt = sess.run(logits,feed_dict = {x:X_test})
predl = sess.run(logits,feed_dict = {x:X_led})


dfled = pd.DataFrame(predl, index = X_index)
dfled.columns = ["prob0", "prob1"]
dfled.to_csv(folder+"knn_nan_nn.csv", header = False)
dfled = dfled.sort_values("prob0", ascending = False)
dffinal = pd.concat([dfled["prob0"],1-((np.sign(dfled["prob0"]-0.5)+1)//2)],axis =1)
dffinal.iloc[:,1].to_csv(folder +"nnpredb.csv", header = False)
dfpt = pd.DataFrame(predt)
dftr = pd.DataFrame(truet)
dfch = pd.concat([dfpt.iloc[:,0],dftr.iloc[:,0]],axis =1)
dfch.columns = ["prob", "pred"]
dfch.sort_values("prob", ascending = False, inplace =True)
tres = 6000
acc = dfch.iloc[:tres,1].sum()/tres

pred = (np.sign(pred - 0.5) + 1 )//2
predt = sess.run(logits,feed_dict = {x:X_test}).reshape(32000)
predt = (np.sign(predt - 0.5) + 1 )//2
accuracy = sum(y_train == pred)/len(y_train)
accuracyt = sum(y_test == predt)/len(y_test)


