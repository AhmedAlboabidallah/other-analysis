# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:04:11 2017

@author: ahalboabidallah
"""
#copy british grid to all lidar
import gdal,glob,os
from gdal import *
import numpy as np
def FindFilesInFolder(folder1,extension):                            
    dr=os.getcwd()
    os.chdir(folder1)
    files=glob.glob(extension)
    os.chdir(dr)
    return files
dataset = gdal.Open('C:/Users/ahalboabidallah/Desktop/20151209-02/20150630-WV3-NPA-PIX-GBR-MS8-L3-02.tif')
projection = dataset.GetProjection()

#path='C:/Users/ahalboabidallah/Desktop/Lidar/LIDAR-DTM-1M-SX36nw/'
#path='C:/Users/ahalboabidallah/Desktop/Lidar/LIDAR-DSM-1M-SX36sw/'
#path='C:/Users/ahalboabidallah/Desktop/Lidar/LIDAR-LAZ-2010-SX47sw/'
#path='C:/Users/ahalboabidallah/Desktop/Lidar/LIDAR-DSM-1M-SX46nw/'
path='C:/Users/ahalboabidallah/Desktop/lidar/LIDAR-DSM-1M-SX46nw/
files=FindFilesInFolder(path,'*.asc')
new_resolution=0.05
for file1 in files:
    file1='C:/Users/ahalboabidallah/Desktop/Lidar/DCM3_final_georef.tif'
    dataset2 = gdal.Open(path+file1, gdal.GA_Update)
    geotransform = np.array(dataset.GetGeoTransform())
    geotransform[1]=new_resolution
    geotransform[5]=-new_resolution
    dataset2.SetGeoTransform( geotransform )
    dataset2.SetProjection(projection)