# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 23:02:03 2017

@author: ahalboabidallah
"""

runfile('C:/Users/ahalboabidallah/Desktop/old laptop/desktop/drake park/functions2.py', wdir='C:/Users/ahalboabidallah/Desktop/old laptop/desktop/drake park')

layers1=FindFilesInFolder('C:/Users/ahalboabidallah/Desktop/ash_farm_new/x-25y0/images/','*tif.tif')
layers=list(filter(lambda x: int(''.join(c for c in x if c.isdigit()))>0, layers1))
layers=sorted(layers,key=lambda x: int(''.join(c for c in x if c.isdigit())),reverse=False)
i=0
for layer1 in layers:
    if i==3:
        i=0
    i+=1
    image1=img2pandas('C:/Users/ahalboabidallah/Desktop/ash_farm_new/x-25y0/images/',layer1)    
    image1 = image1[(image1['Z'] > 0) & (image1['X']>-25.1) & (image1['X']<-22.6) & (image1['Y']>6.3) & (image1['Y']<9)]
    image1['Z']=[0.1*float(''.join(c for c in layer1 if c.isdigit()))]*len(image1)
    image1['b']=['voxel pixel']*len(image1)
    image1['sx']=[1]*len(image1)
    image1['sy']=[1]*len(image1)
    image1['sz']=[1]*len(image1)
    image1['angle']=[0]*len(image1)
    #save to file
    image1.to_csv('C:/Users/ahalboabidallah/Desktop/ash_farm_new/x-25y0/layers/s'+str(i)+'/toCad'+layer1[0:-4]+'.csv',mode = 'a',header=None)
    