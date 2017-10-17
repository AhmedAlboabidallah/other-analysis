# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:50:36 2017

@author: ahalboabidallah
"""
#http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
#http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=170DE91A676E6D5C328F5438063F57FF?doi=10.1.1.114.4288&rep=rep1&type=pdf

from sklearn.svm import SVR
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X, y) 
