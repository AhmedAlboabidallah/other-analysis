# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:49:43 2017

@author: ahalboabidallah
"""

from skimage.morphology import skeletonize
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 


def skeletonizeIT(path1,image1,outpath):
    #read it
    image=(img2np(path1,image1))
    image = Image.fromarray(image)
    image.convert('L')
    image =binary(np.array(image.filter(ImageFilter.BLUR)),1,1)
    #blurred_image = original_image.filter(ImageFilter.BLUR)
    ds = gdal.Open(path1+image1)
    (xmino,res1,tilt1,ymino,tilt2,res2)=ds.GetGeoTransform()
    #blur it
    skeleton = skeletonize(image)*255
    #save the skeleton
    array2raster(outpath,'skil'+image1,xmino,ymino,res1,res2,skeleton,type1=gdal.GDT_Byte)
    