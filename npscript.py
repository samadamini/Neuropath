#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:54:11 2021

@author: samadamini
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
import pickle
from sklearn.model_selection import cross_val_predict,cross_val_score

from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# data prepartion 
df_new = pd.read_json('remap_neuropath.json')
df_new = df_new.T
df_np = pd.read_csv('fhs_neuropath_2019.csv')
df_np['id'] = df_np['id'].apply(lambda x:format(int(x), '04d'))
df_np['id_idtype'] = df_np['idtype'].astype(int).astype(str) +'-'+ df_np['id']
df_new = df_new.replace(r'^\s*$', np.nan, regex=True)
df_new = df_new[df_new['cog_status_at_death'].notna()]
df_nplabeled = df_np[df_np['id_idtype'].isin(df_new.index.to_list())]
df_nplabeled['diagnosis'] = np.nan
for i, row in df_nplabeled.iterrows():
    df_nplabeled.at[i,'diagnosis'] = df_new[df_new.index ==row['id_idtype']].cog_status_at_death.values[0]

df_new.index.names = ['id_idtype']
df_new.reset_index(inplace=True)
df_nplabeled.drop(df_nplabeled[df_nplabeled.diagnosis == 4].index,inplace=True)
df_nplabeled['diagnosis'] = df_nplabeled['diagnosis'].replace([1.5,2.5],[1,2])
df_nplabeled['npnit'] = df_nplabeled['npnit'].replace([1,2,3,4],[2,1,0.5,0])


def NPclassification(task,model):
    assert task in ('NvsMCI','NvsD','MCIvsD'),"choose one of these: 'NvsMCI','NvsD','MCIvsD'"
    assert model in ('LR','SVM','RF','XGboost','catboost'),"choose one of these: 'LR', 'SVM', 'RF', 'XGboost', 'catboost'"
    
    #preparing the data - removing columns with too many missing values - filling the missing values with mode
    Xy = df_nplabeled.dropna(thresh=int(len(df_nplabeled)*0.25), axis=1)
    Xy = Xy.fillna(Xy.mode().iloc[0])
    
    if task == 'NvsMCI':
        Xy.drop(Xy.loc[Xy['diagnosis']>0.5].index, inplace=True)
        Xy["diagnosis"] = Xy["diagnosis"].apply(lambda a: int(float(a)>0))
        Xy["npnit"] = Xy["npnit"].apply(lambda a: int(float(a)>0))
    elif task == 'NvsD':
        Xy.drop(Xy.loc[Xy['diagnosis']==0.5].index, inplace=True)
        Xy["diagnosis"] = Xy["diagnosis"].apply(lambda a: int(float(a)>0))
        Xy["npnit"] = Xy["npnit"].apply(lambda a: int(float(a)>0.5))
    else :
        Xy.drop(Xy.loc[Xy['diagnosis']==0].index, inplace=True)
        Xy["diagnosis"] = Xy["diagnosis"].apply(lambda a: int(float(a)>0.5))
        Xy["npnit"] = Xy["npnit"].apply(lambda a: int(float(a)>0.5))
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sp = {}
    c = 0
    for train_index,test_index in skf.split(Xy,Xy.diagnosis):
        c += 1
        y = Xy.diagnosis.iloc[train_index]
        X = Xy.iloc[train_index]
        X = X.drop(columns=['idtype', 'id','npnit', 'id_idtype', 'diagnosis'])


        if model == 'LR':
            clf = LogisticRegression(random_state=0)
            param_grid = {'C' :[100, 10, 1.0,0.5, 0.1, 0.01,0.005,0.0001]}

        elif model == 'SVM':
            clf = SVC(probability=True)
            param_grid = {'kernel' :['poly', 'rbf', 'sigmoid'],
                          'C' : [50, 10, 1.0, 0.1, 0.01]}
        elif model == 'RF':
            clf = RandomForestClassifier(random_state=0,verbose=False)
            param_grid = {'n_estimators' : [100, 300, 500,None],
                          'max_depth' : range(1, 15, 2),
                          'max_features': ['auto', 'sqrt', 'log2'],
                          'criterion' :['gini', 'entropy'],
                          'min_samples_leaf' : [1, 2, 5, 10]}

        elif model == 'XGboost':
            clf = XGBClassifier()
            param_grid = {'min_child_weight': [1, 3, 5],
                          'objective':['reg:squarederror'],
                          "subsample":[0.5, 0.75, 1],
                          "colsample_bytree":[0.5, 0.75, 1],
                          "max_depth":range(1, 15, 2),
                          'gamma':[i/10.0 for i in range(0,5)],
                          "learning_rate":[0.3, 0.1, 0.05,0.01],
                          "n_estimators":range(50, 400, 50)}
        else :
            clf = CatBoostClassifier(verbose=False)
            param_grid = {'depth': [3,1,2,6,4,5,7,8,9,10],
                          'iterations':[250,100,500,1000],
                          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
                          'l2_leaf_reg':[3,1,5,10,100],
                          'border_count':[32,5,10,20,50,100,200]}

        search = HalvingGridSearchCV(clf, param_grid, 
                                random_state=0,
                                n_jobs=-1,
                                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)).fit(X, y)


        sp[f'split{c}'] = search.best_params_ 
        
    return sp

'''
results_param = {}
for i in ['NvsMCI','NvsD','MCIvsD']:
    r1 = {}
    for j in ['LR','SVM','RF','XGboost','catboost']:
        print(i+' '+j+'is running')
        r1[j] = NPclassification(i,j)
        
    results_param[i] = r1
    
a_file = open("results_param.pkl", "wb")
pickle.dump(results_param, a_file)
a_file.close()
print('Done')

'''
a_file = open("results_param.pkl", "rb")
res = pickle.load(a_file)
a_file.close()


def NPcv(task,model):
    assert task in ('NvsMCI','NvsD','MCIvsD'),"choose one of these: 'NvsMCI','NvsD','MCIvsD'"
    assert model in ('LR','SVM','RF','XGboost','catboost'),"choose one of these: 'LR', 'SVM', 'RF', 'XGboost', 'catboost'"
    
    #preparing the data - removing columns with too many missing values - filling the missing values with mode
    Xy = df_nplabeled.dropna(thresh=int(len(df_nplabeled)*0.25), axis=1)
    Xy = Xy.fillna(Xy.mode().iloc[0])
    
    if task == 'NvsMCI':
        Xy.drop(Xy.loc[Xy['diagnosis']>0.5].index, inplace=True)
        Xy["diagnosis"] = Xy["diagnosis"].apply(lambda a: int(float(a)>0))
        Xy["npnit"] = Xy["npnit"].apply(lambda a: int(float(a)>0))
    elif task == 'NvsD':
        Xy.drop(Xy.loc[Xy['diagnosis']==0.5].index, inplace=True)
        Xy["diagnosis"] = Xy["diagnosis"].apply(lambda a: int(float(a)>0))
        Xy["npnit"] = Xy["npnit"].apply(lambda a: int(float(a)>0.5))
    else :
        Xy.drop(Xy.loc[Xy['diagnosis']==0].index, inplace=True)
        Xy["diagnosis"] = Xy["diagnosis"].apply(lambda a: int(float(a)>0.5))
        Xy["npnit"] = Xy["npnit"].apply(lambda a: int(float(a)>0.5))
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sp = []
    c = 0
    res_task= res[task]
    res_model= res_task[model]
    for train_index,test_index in skf.split(Xy,Xy.diagnosis):
        c += 1
        y = Xy.diagnosis.iloc[train_index]
        X = Xy.iloc[train_index]
        X = X.drop(columns=['idtype', 'id','npnit', 'id_idtype', 'diagnosis'])

        res_split = res_model[f'split{c}']
        param_grid = res_split


        if model == 'LR':
            clf = LogisticRegression(random_state=0)

        elif model == 'SVM':
            clf = SVC(probability=True)
        elif model == 'RF':
            clf = RandomForestClassifier(random_state=0,verbose=False)

        elif model == 'XGboost':
            clf = XGBClassifier()
        else :
            clf = CatBoostClassifier(verbose=False)


        score = cross_val_score(clf,X,y,
                                   fit_params = param_grid,
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))


        sp.append(np.mean(score))
        
    return sp


M = ['LR','SVM','RF','XGboost','catboost']
T = ['NvsMCI','NvsD','MCIvsD']
scores = {}
for i in T:
    r1 = {}
    for j in M:
        r1[j] = NPcv(i,j)
    scores[i] = r1

task_top_model = {}

for i in T:
    model_scores = []
    r1 = scores[i]
    for j in M:
        model_scores.append(np.mean(r1[j]))

    task_top_model[i] = T[np.argmax(model_scores)]



def NPprediction(task,model):
    assert task in ('NvsMCI','NvsD','MCIvsD'),"choose one of these: 'NvsMCI','NvsD','MCIvsD'"
    assert model in ('LR','SVM','RF','XGboost','catboost'),"choose one of these: 'LR', 'SVM', 'RF', 'XGboost', 'catboost'"
    
    #preparing the data - removing columns with too many missing values - filling the missing values with mode
    Xy = df_nplabeled.dropna(thresh=int(len(df_nplabeled)*0.25), axis=1)
    Xy = Xy.fillna(Xy.mode().iloc[0])
    
    if task == 'NvsMCI':
        Xy.drop(Xy.loc[Xy['diagnosis']>0.5].index, inplace=True)
        Xy["diagnosis"] = Xy["diagnosis"].apply(lambda a: int(float(a)>0))
        Xy["npnit"] = Xy["npnit"].apply(lambda a: int(float(a)>0))
    elif task == 'NvsD':
        Xy.drop(Xy.loc[Xy['diagnosis']==0.5].index, inplace=True)
        Xy["diagnosis"] = Xy["diagnosis"].apply(lambda a: int(float(a)>0))
        Xy["npnit"] = Xy["npnit"].apply(lambda a: int(float(a)>0.5))
    else :
        Xy.drop(Xy.loc[Xy['diagnosis']==0].index, inplace=True)
        Xy["diagnosis"] = Xy["diagnosis"].apply(lambda a: int(float(a)>0.5))
        Xy["npnit"] = Xy["npnit"].apply(lambda a: int(float(a)>0.5))
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sp = []
    c = 0
    res_task= res[task]
    res_model= res_task[model]

    auc_avg = []
    acc_avg = []
    mcc_avg = []
    f1_avg = []
    spe_avg = []
    sen_avg = []

    auc_avgd = []
    acc_avgd = []
    mcc_avgd = []
    f1_avgd = []
    spe_avgd = []
    sen_avgd = []
    
    for train_index,test_index in skf.split(Xy,Xy.diagnosis):
        c += 1
        y = Xy.diagnosis.iloc[train_index]
        X = Xy.iloc[train_index]
        X = X.drop(columns=['idtype', 'id','npnit', 'id_idtype', 'diagnosis'])

        y_test = Xy.diagnosis.iloc[test_index]
        X_test = Xy.iloc[test_index]
        X_test = X_test.drop(columns=['idtype', 'id','npnit', 'id_idtype', 'diagnosis'])

        y_npnit = Xy.npnit.iloc[test_index]

        res_split = res_model[f'split{c}']
        param_grid = res_split


        if model == 'LR':
            clf = LogisticRegression(random_state=0)

        elif model == 'SVM':
            clf = SVC(probability=True)
        elif model == 'RF':
            clf = RandomForestClassifier(random_state=0,verbose=False)

        elif model == 'XGboost':
            clf = XGBClassifier()
        else :
            clf = CatBoostClassifier(verbose=False)
            
        clf.set_params(**param_grid)
        clf.fit(X,y)
        y_pred = clf.predict_proba(X_test)

        fpr, tpr, thresholds = metrics.roc_curve(np.array(y_test), np.array(y_pred[:,1]))
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        threshold = thresholds[ix]
        auc=metrics.auc(fpr, tpr)
        auc_avg.append(auc)
        acc_avg.append(np.max([metrics.accuracy_score(y_test, (1*(np.array(y_pred[:,1])>th)).tolist()) for th in thresholds]))
        mcc_avg.append(np.max([metrics.matthews_corrcoef(y_test, (1*(np.array(y_pred[:,1])>th)).tolist()) for th in thresholds]))
        f1score = np.max([metrics.f1_score(y_test, (1*(np.array(y_pred[:,1])>th)).tolist(), average='weighted') for th in thresholds])
        f1_avg.append(f1score)
        tn, fp, fn, tp = confusion_matrix(np.array(y_test), np.array(y_pred[:,1])>threshold).ravel()
        spe = tn / (tn+fp)
        sen = tp / (tp+fn)
        spe_avg.append(spe)
        sen_avg.append(sen)


        fpr, tpr, thresholds = metrics.roc_curve(np.array(y_test), np.array(y_npnit))
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        threshold = thresholds[ix]
        auc=metrics.auc(fpr, tpr)
        auc_avgd.append(auc)
        acc_avgd.append(np.max([metrics.accuracy_score(y_test, (1*(np.array(y_npnit)>th)).tolist()) for th in thresholds]))
        mcc_avgd.append(np.max([metrics.matthews_corrcoef(y_test, (1*(np.array(y_npnit)>th)).tolist()) for th in thresholds]))
        f1score = np.max([metrics.f1_score(y_test, (1*(np.array(y_npnit)>th)).tolist(), average='weighted') for th in thresholds])
        f1_avgd.append(f1score)
        tn, fp, fn, tp = confusion_matrix(np.array(y_test), np.array(y_npnit)>threshold).ravel()
        spe = tn / (tn+fp)
        sen = tp / (tp+fn)
        spe_avgd.append(spe)
        sen_avgd.append(sen)

    results_model = [np.mean(auc_avg),np.mean(acc_avg),np.mean(mcc_avg),np.mean(f1_avg),np.mean(spe_avg),np.mean(sen_avg)]
    results_npnit = [np.mean(auc_avgd),np.mean(acc_avgd),np.mean(mcc_avgd),np.mean(f1_avgd),np.mean(spe_avgd),np.mean(sen_avgd)]

        
    return results_model,results_npnit

r_model = {}
r_npnit = {}
for i in T:
    model_final = task_top_model[i]
    r_model[i],r_npnit[i] = NPprediction(i,model_final)
