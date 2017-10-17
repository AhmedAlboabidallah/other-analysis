# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:12:40 2017

@author: ahalboabidallah
"""
runfile('C:/Users/ahalboabidallah/Desktop/functions2.py', wdir='C:/Users/ahalboabidallah/Desktop/drake park')
X=np.array(readtolist('C:/Users/ahalboabidallah/Desktop/','csv.csv',NoOfColumns=3,alradyhasheader=0))
#from time import gmtime, strftime
print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
for i in range(100):
#    x0n, an, rn, d, sigmah, conv, Vx0n, Van, urn, GNlog,a, R0, R=lscylinder(X,([[10],[10],[10]]),np.transpose(np.array([1,1,0])),4,4, np.array([4]),w=1)
     #print(i)
     #skeletonizeIT('C:/Users/ahalboabidallah/Desktop/ash_farm_new/x-25y0/Series/','10.tif','C:/Users/ahalboabidallah/Desktop/ash_farm_new/x-25y0/Series/','del.tif')
     #a=princomp(X,numpc=1)
     t=fit_ellipse(X[:,:2])
     [X0,Y0,aa,bb,theta]=gen_elli2(t)
print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))