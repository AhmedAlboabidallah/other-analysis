# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:06:13 2017

@author: ahalboabidallah
"""
"""


# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
X = pd.read_csv('C:/Users/ahalboabidallah/Desktop/X.csv').values
y = pd.read_csv('C:/Users/ahalboabidallah/Desktop/Y.csv').values

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestRegressor(n_jobs=-1, max_depth=10)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2)

# find all relevant features
feat_selector.fit(X, y)

# check selected features
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE#RFA stands for Recursive Feature Augmentation.
from sklearn.svm import SVR

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold#StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_regression

number_of_features=5
XY = pd.read_csv('C:/Users/ahalboabidallah/Desktop/feature_selection.csv').values
#y = pd.read_csv('C:/Users/ahalboabidallah/Desktop/Y.csv').values
X,y=XY[:,:-1],XY[:,-1]

#X, y = make_regression(n_samples=1000, n_features=20, n_informative=3, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=1.0, shuffle=True, coef=False, random_state=None)
#X[:,1]=X[:,0]
estimator = SVR(kernel="linear",degree=1)
selector = RFE(estimator, number_of_features, step=1)
selector1 = selector.fit(X, y)
print(selector1.support_)
print(selector1.ranking_)
#print(selector1.grid_scores_)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)


# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
standardizedX = scaler.transform(X)

#the code was based on http://fatihsarigoz.com/scaling-rfe.html
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
def RFAperf(ranking, modelstr, Xw, y, names, cv=False):
    '''
    RFAperf runs the given model using the provide feature ranking list starting 
    with the top feature and adding the next important feature in a recursive 
    fashion and reports the score at each step along with the features utilized
    in that step. 
    '''
    ranking = list(ranking)
    model = eval(modelstr)
    l = len(ranking) 
    f_inds = []
    f_names = np.array([])
    f_scorelist = []
    for i in range(1,l+1):
        f_ind = ranking.index(i)
        f_inds.append(f_ind)
        f_names = np.append( f_names, names[f_ind] )
        Xin = Xw[:,f_inds]
        score = runModel(model, Xin, y, cv)
        f_scorelist.append( (f_names, score) )
    return(f_scorelist)
def runModel(model, Xw, y, cv):#runModel is a generic function that runs a given model and produces the r2 score with or without cross-validation.
    if cv==False:
        model.fit(Xw,y)
        score = model.score(Xw,y)     
    else:
        kfold = KFold(n_splits=10, random_state=15, shuffle=True)
        scores = cross_val_score(model, Xw, y, cv=kfold, scoring='r2')
        score = np.array(scores).mean()
        print('scores',scores)
    return(score)
def rankRFE(models, Xversions, y, names):
    '''
    rankRFE runs the RFE feature elimination algorithm on the list of models
    provided by the models variable to provide a feature ranking for each which
    then is utilized by RFAperf to produce feature augmentation performance results.
    The resulting data is compiled into the modelsData variable which can then 
    be passed onto a plotting function.
    '''
    lnames = len(names)
    FAstr = 'RFE'
    modelsData = []
    results = pd.DataFrame([], index=range(1,lnames+1))
    for inputType, Xw in Xversions:
        for model in models:
            modelname = str(model).partition('(')[0]
            rfe = RFE(model, 1)
            # rank RFE results
            rfe.fit(Xw, y)
            ranking = rfe.ranking_
            f_scorelist = RFAperf(ranking, str(model), Xw, y, names, cv=True)
            modelsData.append( (inputType, str(model), FAstr, ranking, f_scorelist) ) 
            f_ranking = [n for r, n in sorted( zip( ranking, names ) )]
            results[modelname[0:3] + FAstr + '-' + inputType[0:2]] = f_ranking
    return(modelsData, results)
def plotRFAdata(modelsData, names):
    '''
    plotRFAdata extracts the information provided in the modelsData variable 
    which is compiled by running the RFAperf function over many different models 
    (an instance of which is the rankRFE function above utilizing the RFE method)
    and plots the score curve for each model/test case.
    '''
    n = len(modelsData)
    l = len(names)
    fig = plt.figure()
    xvals = range(1,l+1)
    colorVec = ['ro', 'go', 'bo', 'co', 'mo', 'yo', 'ko', 'rs', 'gs', 'bs', 'cs', 'ms', 'ys', 'ks','ro', 'go', 'bo', 'co', 'mo', 'yo', 'ko', 'rs', 'gs', 'bs', 'cs', 'ms', 'ys', 'ks']# only colours
    for i in range(n):
      if i==1:
        modelData = modelsData[i]
        inputType = modelData[0]
        modelstr = modelData[1]
        modelname = modelstr.partition('(')[0]
        FAstr = modelData[2]
        ranking = modelData[3]
        f_scorelist = modelData[4]
        f = np.array(f_scorelist)[:,0]
        s = np.array(f_scorelist)[:,1]
        labelstr = modelname[0:3] + FAstr + '-' + inputType[0:2]
        plt.plot(xvals, s, colorVec[i]+'-',  label=labelstr) 
    fig.suptitle('Recursive Feature Augmentation Performance')
    plt.ylabel('R^2')
    #plt.ylim(ymax=1)
    plt.xlabel('Number of Features')
    plt.xlim(1-0.1,l+0.1)
    plt.legend(loc='lower right', fontsize=10)
    ax = fig.add_subplot(111)
    ax.set_xticks(xvals)
    plt.show()



from sklearn.svm import SVR
print('1')
Smodels = [SVR(kernel="linear",degree=1)]
Xversions = [('original', X), ('rescaled', rescaledX), ('standardized', standardizedX)]
names = ['1', '2', '3', '4', '5', '6', '7','8','9','10','k','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41']
modelsData, results = rankRFE(Smodels, Xversions, y, names)
print('2')
print(results)

plotRFAdata(modelsData, names)



