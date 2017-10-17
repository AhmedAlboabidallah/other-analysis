# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:35:33 2017

@author: ahalboabidallah
"""
import tkinter as tk
from tkinter import *
#from Tkinter import Tk
import math as m
import math
import csv
import numpy as np
import numpy
import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib
import sys
import shapefile#                                             pip install pyshp
from time import gmtime, strftime
import cv2 #            conda install -c https://conda.binstar.org/menpo opencv
import scipy.linalg
from scipy import signal as sg
from scipy.linalg import inv, eigh, solve
import pylab
from mpl_toolkits.mplot3d import Axes3D
#from tkinter import *
#import Image #http://www.pythonware.com/products/pil/
from PIL import Image
#from __future__ import print_function
import glob
import os
#from easygui import *
import vigra #conda create -n vigra -c ukoethe python=2.7.10.vc11 vigra=1.11.0.vc11
import pandas as pd          #activate vigra
from vigra import *

def readtopandas(path1,file1,alradyhasheader=0):#
    #F='C:/Users/ahalboabidallah/Desktop/test.csv'
    F=path1+file1
    #add header to the file if there is no header #
    if alradyhasheader==0:
        #generate a header 
        df = pd.read_csv(F,header=None)
    else:
        df = pd.read_csv(F)#needs a csv with a header line
    return df 

def readtolist(path,file1,NoOfColumns=3,alradyhasheader=0):
    df=readtopandas(path,file1,alradyhasheader=0)
    list1=df.values.tolist()
    return list1
    
X=readtolist('C:/Users/ahalboabidallah/Desktop/desktop/','radars.csv',NoOfColumns=3,alradyhasheader=1)
jan,mar,dec=list(np.array(X)[:,0])[1:],list(np.array(X)[:,1])[1:],list(np.array(X)[:,2])[1:]
jan = [ float(a)/.85 for a in jan]
mar = [ (float(a)-.15)/.6 for a in mar]
dec = [ (float(a)-.06)/.65 for a in dec]


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
#
#
imgm = mar
imgl = list( map(lambda x,y: (2*x+y)/3, dec,mar))
#
fig1 = plt.figure()
plt.hexbin(imgl,imgm,bins='log', cmap=cm.jet)
plt.axis([0,1,0,1])
plt.ylabel('28-Feb-2016,11-Mar-2016')
plt.xlabel('24-Dec-2016,12-Dec-2016')
plt.title('Scatter Plot')
plt.colorbar()
plt.show(fig1)
#'''
imgm = mar
imgl = jan

fig2 = plt.figure()
plt.hexbin(imgl,imgm,bins='log', cmap=cm.jet)
plt.axis([0,1,0,1])
plt.ylabel('28-Feb-2016,11-Mar-2016')
plt.xlabel('23-Jan-2016,28-Feb-2016')
plt.title('Scatter Plot')
plt.colorbar()
plt.show(fig2)
#''


