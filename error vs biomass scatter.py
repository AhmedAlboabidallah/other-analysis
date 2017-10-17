# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:35:33 2017

@author: ahalboabidallah
"""
runfile('C:/Users/ahalboabidallah/Desktop/functions2.py', wdir='C:/Users/ahalboabidallah/Desktop')
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

path1='C:/Users/ahalboabidallah/Desktop/mont_carlo/S1newPixel/'
image1='AGB.tif'
image2='errorBasedonRegReverseNorm.tif'
image3='errorRS_Norm.tif'
XYZ1=img2pandas(path1,image1)
XYZ2=img2pandas(path1,image2)
XYZ3=img2pandas(path1,image3)
#X=readtolist('C:/Users/ahalboabidallah/Desktop/desktop/','radars.csv',NoOfColumns=3,alradyhasheader=1)
X=XYZ1
X=X[['Z']]
X['Z2']=XYZ2[['Z']]
X['Z3']=XYZ3[['Z']]
X=X.dropna()
X.to_csv('C:/Users/ahalboabidallah/Desktop/mont_carlo/S1newPixel/AGBvsError1.csv',index=False,header=False)
