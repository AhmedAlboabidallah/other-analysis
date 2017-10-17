from tkinter import *
from tkinter.filedialog import *
from tkinter.messagebox import showerror
from tkinter import messagebox
from tkinter import ttk
import os
import numpy as np
import pandas as pd
from pandas import *
import gdal
from gdal import *
import cython
from cython.parallel import *
from multiprocessing import Pool
import math
import numpy
import statsmodels.api
import statsmodels as sm
import statsmodels.robust
from rasterstats import zonal_stats
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.preprocessing import LabelEncoder  
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
#Field_sites,Independent_bands =0,0
# ask_yes_no.py
import shutil
from shutil import copyfile
import glob
#1

def normalize(array1,maxmins='not'):
    try:
        array1=np.array(array1)
    except:
        pass
    if maxmins=='not':
        maxmins=[]
        #i=-1
    for i in range(np.shape(array1)[1]):
        #print(i)
        column1=array1[:,i]
        if maxmins=='not':
            max1,min1=max(column1),min(column1)
        else:
            max1,min1=maxmins[i]
        for j in range(len(column1)):
            column1[j]=(float(column1[j])-float(min1))/(float(max1)-float(min1))
            array1[:,i]=column1
    return array1,maxmins
#array1,maxmins=normalize(xx)
def inv_normalize(array1,maxmins):
    for i in range(np.shape(array1)[1]):
        column1=array1[:,i]
        for i in range(len(column1)):
            max1,min1=maxmins[i]
            column1[i]=float(column1[i])*(float(max1)-float(min1))+float(min1)
            array1[:,i]=column1
    return column1
def FindFilesInFolder(folder1,extension):                            
    dr=os.getcwd()
    os.chdir(folder1)
    files=glob.glob(extension)
    os.chdir(dr)
    return files

#1
def object_zonal_stats(shp,TIF):
    stats = zonal_stats(shp, TIF,stats=['mean'])
    return list([f['mean'] for f in stats])
def object_zonal_stats_list(shp,TIFs_list):#for the all images of the same band same dataset(all subsets)
    for tif in TIFs_list:
        #statsall=[]
        stats=np.array(object_zonal_stats(shp,tif))
        try:
            statsall=statsall+stats
        except:
            statsall=np.array(stats)
    return statsall
#test
#shp='C:/Users/ahalboabidallah/Desktop/Hi_RS_Data/tree_Segments.shp'
#TIFs_list=['C:/Users/ahalboabidallah/Desktop/20151209-01/20150630-WV3-NPA-PIX-GBR-PAN-L3-01.tif', 'C:/Users/ahalboabidallah/Desktop/20151209-01/20150630-WV3-NPA-PIX-GBR-MS8-L3-01.tif']
#stats0=object_zonal_stats(shp,TIFs_list[0])
#stats1=object_zonal_stats(shp,TIFs_list[1])
#statsall=object_zonal_stats_list(shp,TIFs_list)
from gdal import *
#import ntpath
#def copy_tif(raster, outraster):
 #   #src_ds = gdal.Open(raster)
  #  #driver = gdal.GetDriverByName('GTiff')
   # #dst_ds = driver.CreateCopy(outraster, src_ds, 0 )
    # Once we're done, close properly the dataset
    #dst_ds = None
    #src_ds = None
    #folder1, file1= os.path.split(raster)#ntpath.split(raster)
#    folder2, file2=os.path.split(outraster)
 #   l=len(file1[:-4])
  #  files=FindFilesInFolder(folder1,file1[:l]+'*')
   # for f in files:
    #    shutil.copy(folder1+'/'+f,folder2+'/'+file2[:-4]+f[l:])
    
def shift_raster(raster, Dx,Dy,Dt, outraster):#Dx,Dy,Dt=sp_errorsX[L],sp_errorsY[L],sp_errorsT[L]*3.14/180
    shutil.copy(raster,outraster)
    #copy_tif(raster, outraster) 
    dataset = gdal.Open(raster)
    projection = dataset.GetProjection()
    geotransform = np.array(dataset.GetGeoTransform())+(Dx,0,Dt,Dy,Dt,0)
    dataset2 = gdal.Open(outraster, gdal.GA_Update)
    dataset2.SetGeoTransform( geotransform )
    dataset2.SetProjection( projection )
#test
#raster='C:/Users/ahalboabidallah/Desktop/20151209-01/20150630-WV3-NPA-PIX-GBR-PAN-L3-01.tif'
#outraster='C:/Users/ahalboabidallah/Desktop/test.tif'
#Dx,Dy,Dt=10.0,1.0,np.pi/4
#shift_raster(raster,Dx,Dy,Dt , outraster)

def writetofile(path,file1,list1,NoOfColumns=3):
    try:
        os.stat(path)
    except:
        os.makedirs(path)
    #convert to pandas
    df=DataFrame(list1)
    #write to csv
    df.to_csv(path+file1,index=False,header=False)

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

def img2pandas(path1,file1):
    #open file
    try:
        src_dataset = gdal.Open(path1+file1)
    except:
        src_dataset = gdal.Open(path1+'/'+file1)
    z = src_dataset.ReadAsArray()
    #read georeferencing
    (xmin,res1,tilt1,ymin,tilt2,res2)=src_dataset.GetGeoTransform()
    ys,xs=np.shape(z)
    x = np.array([list(np.linspace(xmin, xmin+(xs-1)*res1, xs))]*(ys))
    y = np.transpose(np.array([list(np.linspace(ymin, ymin+(ys-1)*res1, ys))]*(xs)))
    #z1=list(z.ravel())
    #y1=list(y.ravel())
    #1=list(x.ravel())
    data=np.array([list(x.ravel()),list(y.ravel()),list(z.ravel())])
    #'C:/Users/ahalboabidallah/Desktop/ash_farm_new/profiles/profiles/results/AllGround.tiff'
    return pd.DataFrame(data,index=['X','Y','Z']).transpose()



def frrot3(theta,U0=numpy.matrix(numpy.identity(3))):
    ct = np.mat([[math.cos(theta[0])],[math.cos(theta[1])],[math.cos(theta[2])]])
    st = np.mat([[math.sin(theta[0])],[math.sin(theta[1])],[math.sin(theta[2])]])
    if max((theta.shape)) > 0:
        R1 = np.mat([[1,0,0],[0,ct[0],-st[0]],[0,st[0],ct[0]]])
        R = R1;
    
    if max((theta.shape)) > 1:
        R2 = np.mat([[float(ct[1]),0,-st[1]],[0,1,0],[st[1],0,ct[1]]]);
        R = R2*R;
    
    if max((theta.shape)) > 2:
        R3 = np.mat([[float(ct[2]),-st[2],0],[st[2],ct[2],0],[0,0,1]]);
        R = R3*R;
    
    R = R*U0;
    #checked and vetted
    return (R,R1,R2,R3)


def add_degree(xx):
    cols=list(xx.columns.values)
    #cols = cols[-1] + cols[:-1]
    xx1 = xx[cols[:-1]]
    yy=cols[-1]
    xx2=xx1*1
    cols=list(xx1.columns.values)
    i=len(cols)
    i_up=i
    i_down=i
    for col in cols:
        for ii in range(i_down):
            i_up+=1
            xx2[i_up] = xx1[col]*xx1[cols[-ii]]
            print('i_up',i_up,'col',col,'*',cols[-ii])
        i_down-=1
    cols=list(xx2.columns.values)
    cols = cols[-1:] + cols[:-1]
    xx2['yy']=xx[yy]
    return xx2
import copy
from sklearn.externals import joblib
#joblib.dump(Reg, 'filename.pkl') 

def iteration1(table1):#
    #table1=tables[0]
    print('>',end="")
    #if 1==1:
    [field_data,RStable,Processing,model_spec]=table1[:]
    RStable2=[]
    i=0
    #to include all subsets
    #if 1==1:
    for fd1 in field_data:
        #fd1=field_data[0]
        #print('fd1',fd1)
        i+=1
        RStable1=RStable[:]
        for rs in RStable1:
            #rs=RStable1[2]
            #print('f',rs)
            rs1=[rs[0]+'subset/'+str(i)+'/']
            if rs[-1]==1 or rs[-1]=='1':
                expected_spectral_error=rs[0]+'subset/'+str(i)+'/error'+rs[1]
            else:
                expected_spectral_error=rs[-2]
            rs1.extend(rs[1:-2])
            rs1.extend([expected_spectral_error,rs[-1]])
            RStable2.append(rs1)
    RStable=RStable2[:]
    if Processing[0]=='Pixel':#if it is a pixel based model
        RStable=sorted(RStable,key=lambda l:-float(l[4]))#sort by resolution
        Lowest_resolution=float(RStable[0][4])# find lowest resolution
        fields,Fsp_errorsX,Fsp_errorsY,Fsp_errorsT=add_error(field_data)# adds errors to the field data 
        #fields=list(filter(lambda a: a[2] > 0 and a[2] < 100000,fields[0]))
        rss0=no_error(RStable)
        fields0=no_error(field_data)
        Biomass0,rs_bands0=creat_table4corlation(rss0,fields0,RStable)
        Biomass0= np.asarray(Biomass0, dtype=float)
        while len(Biomass0[0])==1:# to get rid of unnecessary brackets
            Biomass0=list(map(lambda x:x[0], Biomass0))
        rs_bands0[0]=list(filter(lambda a: a[2] > 0 and a[2] < 100000,rs_bands0[0]))   
        yy0=regression_images(rs_bands0[0],Lowest_resolution,Lowest_resolution,Biomass0)
        xx0=[np.array(rs_bands0[0])[:,2].tolist()]
        for i in rs_bands0[1:]:
            #print('i')
            try:
                xx0.append(regression_images(rs_bands0[0],Lowest_resolution,Lowest_resolution,i))#
            except:
                print ('error')
        xx0.append(yy0)# now the pixel based table is ready for the model
        xx0=pd.DataFrame(xx0)
        xx0=xx0.transpose()
        xx0=xx0.dropna()
        
        
        #XX0=
        rss,Rsp_errorsX,Rsp_errorsY,Rsp_errorsT=add_error(RStable)# adds errors to the rss data 
        Biomass,rs_bands=creat_table4corlation(rss,fields,RStable)# combines field data #combines any similar band and dataset RS inputs because they are subsetted from the same origional raster
        
        Biomass1 = np.asarray(Biomass, dtype=float)
        #k=0
        while len(Biomass1[0])==1:# to get rid of unnecessary brackets
            #k+=1
            #print('k')
            Biomass1=list(map(lambda x:x[0], Biomass1))
            print('<<>>')
        #Biomass1=list(filter(lambda a: a > -0.0001 and a[2] < 100000,Biomass1))
        rs_bands[0]=list(filter(lambda a: a[2] > 0 and a[2] < 100000,rs_bands[0]))
        yy=regression_images(rs_bands[0],Lowest_resolution,Lowest_resolution,Biomass1)
        xx=[np.array(rs_bands[0])[:,2].tolist()]
        for i in rs_bands[1:]:
            #print('i')
            try:
                xx.append(regression_images(rs_bands[0],Lowest_resolution,Lowest_resolution,i))#
            except:
                print ('error')
        xx.append(yy)# now the pixel based table is ready for the model
        xx=pd.DataFrame(xx)
        xx=xx.transpose()
        xx=xx.dropna()
    else:
        #if it is an object based
        fields,Fsp_errorsX,Fsp_errorsY,Fsp_errorsT=add_error_object(field_data,Processing[1])
        rss,Rsp_errorsX,Rsp_errorsY,Rsp_errorsT=add_error_object(RStable,Processing[1])
        Biomass,rs_bands=creat_table4corlation_object(rss,fields,RStable)
        xx=(np.array(rs_bands)).tolist()
        #xx=np.transpose(np.array(rs_bands)).tolist()
        xx.extend([Biomass])
        xx=(np.transpose(np.array(xx)).tolist())
        #filter sum(n < 0 for n in nums)
        xx=list(filter(lambda a: sum(n <= 0 for n in a)==0,xx))
        xx=(np.transpose(np.array(xx)).tolist())
        xx=pd.DataFrame(xx)
        xx=xx.transpose()
        xx=xx.dropna()
        #
        fields0=no_error_object(field_data,Processing[1])
        rss0=no_error_object(RStable,Processing[1])
        Biomass0,rs_bands0=creat_table4corlation_object(rss0,fields0,RStable)
        xx0=(np.array(rs_bands0)).tolist()
        #xx=np.transpose(np.array(rs_bands)).tolist()
        xx0.extend([Biomass0])
        xx0=(np.transpose(np.array(xx0)).tolist())
        #filter sum(n < 0 for n in nums)
        xx0=list(filter(lambda a: sum(n <= 0 for n in a)==0,xx0))
        xx0=(np.transpose(np.array(xx0)).tolist())
        xx0=pd.DataFrame(xx0)
        xx0=xx0.transpose()
        xx0=xx0.dropna()
        print('xx0',xx0)
    #xx.to_csv('C:/Users/ahalboabidallah/Desktop/mont_carlo/xx.csv',index=False,header=False)
    if model_spec[0]=='Parametric_Regression':#if the model is regression
        xx=xx.values.tolist()
        yy=list(map(lambda x:x[-1], xx))
        xx=list(map(lambda x:x[0:-1], xx))
        xx0=xx0.values.tolist()
        #yy0=list(map(lambda x:x[-1], xx))
        xx0=list(map(lambda x:x[0:-1], xx0))
        if model_spec[1]>1:
            xx=add_degrees(xx,model_spec[1])
            xx0=add_degrees(xx0,model_spec[1])
        model = sm.api.OLS(yy,xx)#len(final_table[1:].values.tolist())==len(final_table[0].values.tolist())
        results = model.fit()
        print(results.summary())    
        #print('Parameters: ', results.params)
        #print('R2: ', results.rsquared)
             #record.append([i,[results.params],results.rsquared])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','results_rsquared.csv',[results.rsquared])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','results_params.csv',list(results.params))
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','RS_spatial_errors_Dx.csv',Rsp_errorsX)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','RS_spatial_errors_Dy.csv',Rsp_errorsY)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','RS_spatial_errors_Dt.csv',Rsp_errorsT)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','F_spatial_errors_Dx.csv',Fsp_errorsX)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','F_spatial_errors_Dy.csv',Fsp_errorsY)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','F_spatial_errors_Dt.csv',Fsp_errorsT)
        #Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','maxmins.csv',maxmins)
        file1='C:/Users/ahalboabidallah/Desktop/reg/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        joblib.dump(model, file1)
        yy1=results.predict(xx0)
        #yy1=results.predict(xx)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','YY.csv',yy1.tolist())
        #print('yy1',yy1)
    elif model_spec[0]=='Support_Vector_Machine':#if the model is 'Support_Vector_Machine'
        #http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
        #http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=170DE91A676E6D5C328F5438063F57FF?doi=10.1.1.114.4288&rep=rep1&type=pdf
        #n_samples, n_features = 10, 5
        #np.random.seed(0)
        #y = np.random.randn(n_samples)
        #X = np.random.randn(n_samples, n_features)
        xx=xx.values.tolist()
        xx,maxmins=normalize(xx)
        xx=list(xx)
        yy=np.array(list(map(lambda x:x[-1], xx)))
        xx=np.array(list(map(lambda x:x[0:-1], xx)))
        
        #yy0=np.array(list(map(lambda x:x[-1], xx0)))
        xx0=np.array(list(map(lambda x:x[0:-1], xx0)))
        kernel,degree=model_spec[1:]
        Reg = SVR(kernel=kernel,C=2000, epsilon=0.2,degree=degree)
        Reg.fit(xx, yy) 
        R2=np.corrcoef([yy,Reg.predict(xx)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/SVM/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/SVM/','results_rsquared.csv',[file1,R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/SVM/','maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') 
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','YY.csv',yy1.tolist())
    elif model_spec[0]=='Neural Network':#if the model is 'Neural Network'
        NumberOfLayers,Layers=model_spec[1:]
        xx=xx.values.tolist()
        xx,maxmins=normalize(xx)
        xx=list(xx)
        yy=np.array(list(map(lambda x:x[-1], xx)))
        xx=np.array(list(map(lambda x:x[0:-1], xx)))
        #
        Reg = MLPRegressor(hidden_layer_sizes=Layers,  activation='relu', solver='adam', alpha=0.001, batch_size='auto',learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-05)
        Reg.fit(xx, yy) 
        R2=np.corrcoef([yy,Reg.predict(xx)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/Neural/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','files.csv',[file1])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','results_rsquared.csv',[R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') 
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','YY.csv',yy1.tolist())
    elif model_spec[0]=='Gaussian_Process':#if the model is 'Gaussian_Process'
        Gaussian_Process_kernal=model_spec[1]
        xx=xx.values.tolist()
        xx,maxmins=normalize(xx)
        xx=list(xx)
        yy=np.array(list(map(lambda x:x[-1], xx)))
        xx=np.array(list(map(lambda x:x[0:-1], xx)))
        #
        Reg = GaussianProcessRegressor(kernel=Gaussian_Process_kernal)
        Reg.fit(xx, yy) 
        R2=np.corrcoef([yy,Reg.predict(xx)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/Gaussian_Process/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Gaussian_Process/','files.csv',[file1])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Gaussian_Process/','results_rsquared.csv',[R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Gaussian_Process/','maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') '''
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','YY.csv',yy1.tolist())
    elif model_spec[0]=='Random_Forest':#if the model is 'Random_Forest'
        Random_Forest_Bootstrap=model_spec[1]
        xx=xx.values.tolist()
        xx,maxmins=normalize(xx)
        xx=list(xx)
        yy=np.array(list(map(lambda x:x[-1], xx)))
        xx=np.array(list(map(lambda x:x[0:-1], xx)))
        #
        Reg = RandomForestRegressor(bootstrap=Random_Forest_Bootstrap)
        Reg.fit(xx, yy) 
        R2=np.corrcoef([yy,Reg.predict(xx)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/Random_Forest/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Random_Forest/','files.csv',[file1])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Random_Forest/','results_rsquared.csv',[R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Random_Forest/','maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') '''
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','YY.csv',yy1.tolist())
    elif model_spec[0]=='K_Nearest_Neighbours':#if the model is 'K_Nearest_Neighbours'
        K_Nearest_Neighbours_weights,K_Nearest_Neighbours_k=model_spec[1:]
        xx=xx.values.tolist()
        xx,maxmins=normalize(xx)
        xx=list(xx)
        yy=np.array(list(map(lambda x:x[-1], xx)))
        xx=np.array(list(map(lambda x:x[0:-1], xx)))
        #
        Reg = KNeighborsRegressor(n_neighbors=2,weights =K_Nearest_Neighbours_weights)
        Reg.fit(xx, yy) 
        R2=np.corrcoef([yy,Reg.predict(xx)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/Random_Forest/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Random_Forest/','files.csv',[file1])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Random_Forest/','results_rsquared.csv',[R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Random_Forest/','maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') '''
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','YY.csv',yy1.tolist())
        print(np.shape(xx0))
    #return yy1,xx0

def spatial_error_xy(table3): #images of the same dataset should take the same spatial error
    datasets=np.array((np.array(table3)[:,3]),dtype=float)
    sp_errors=list(datasets)
    for data1 in list(set(datasets)):
        #print ('data1',data1)
        expected_spatial_error=float(list(filter(lambda x: float(x[3])==data1, table3))[0][7])
        e=np.random.normal(0, expected_spatial_error, 1)[0]
        sp_errors=list(map(lambda x:x if x!=data1 else e,sp_errors))
        #replace sp_errors dataset with a random error
    return sp_errors
def spatial_error_t(table3): #images of the same dataset should take the same spatial error
    datasets=np.array((np.array(table3)[:,3]),dtype=float)
    sp_errors=list(datasets)
    for data1 in list(set(datasets)):
        #print ('data1',data1)
        expected_orientation_error=float(list(filter(lambda x: float(x[3])==data1, table3))[0][8])
        if expected_orientation_error != 0:
            e=np.random.normal(0, expected_orientation_error, 1)[0]
        else:
            e=0
        t_errors=list(map(lambda x:x if x!=data1 else e,sp_errors))
        #replace sp_errors dataset with a random error
    return t_errors
#i=-1
def add_error(table2):#
    images=[]    
    sp_errorsX=spatial_error_xy(table2)
    sp_errorsY=spatial_error_xy(table2)
    sp_errorsT=spatial_error_t(table2)
    L=-1
    for row1 in table2:
        #row1=table2[1]
        print('row1',row1)
        if isinstance(row1[0][0], (list, tuple)):
            [path1,file1,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral,expected_spatial_error,expected_orientation_error,expected_spectral_error,ismap]=row1[0]
            #[path1,file1,expected_spatial_error,expected_spectral_error,expected_orientation_error,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral]=row1[0]
        else:
            [path1,file1,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral,expected_spatial_error,expected_orientation_error,expected_spectral_error,ismap]=row1
        #print (path1,file1)
        L+=1
        #read the image
        img=np.array(img2pandas(path1+'/',file1).values.tolist())
        if ismap==1 or ismap=='1':
            print('expected_spectral_error',expected_spectral_error)
            ErrorMap=img2pandas(expected_spectral_error,'').values.tolist()
            expected_spectral_error=1
        else:
            ErrorMap=img[:,:]*1
            ErrorMap[:,2]=[expected_spectral_error]*len(img)
        if is_Spectral==1 or is_Spectral=='1':
            #add the spectral error
            if isinstance(ErrorMap, (list)):
                ErrorMap=np.array(ErrorMap)
            if isinstance(img, (list)):
                img=np.array(img)
            img[:,2]= img[:,2]+ np.random.normal(0, expected_spectral_error, len(img))*ErrorMap[:,2]
        if is_Spatial==1 or is_Spatial=='1':
            if isinstance(img, (list)):
                img=np.array(img)
            #add the spatial error
            img[:,0]= img[:,0]+ sp_errorsX[L]
            img[:,1]= img[:,1]+ sp_errorsY[L]
            img_Cx,img_Cy=(min(img[:,0])+max(img[:,0]))/2,(min(img[:,1])+max(img[:,1]))/2
            img=((img-[img_Cx,img_Cy,0])*(frrot3(np.array([0,0,3.141592653589793/180*sp_errorsT[L]]))[0])+[img_Cx,img_Cy,0]).tolist()
        images.append(img)
    return images,sp_errorsX,sp_errorsY,sp_errorsT
#==============================================================================================================================================================
from random import randint
print()
def add_error_object(table2,shp):#table2,shp=field_data,Processing[1]
    images=[]    
    sp_errorsX=spatial_error_xy(table2)
    sp_errorsY=spatial_error_xy(table2)
    sp_errorsT=spatial_error_t(table2)
    L=-1
    for row1 in table2:#row1=table2[0]
        #print(row1)
        if isinstance(row1[0][0], (list, tuple)):
            [path1,file1,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral,expected_spatial_error,expected_orientation_error,expected_spectral_error,ismap]=row1[0]
        else:
            [path1,file1,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral,expected_spatial_error,expected_orientation_error,expected_spectral_error,ismap]=row1
        #print (path1,file1)
        L+=1
        #add spectral-spatial error to the image and save it to a temporary file
        temp=str(randint(1000, 9999))+str(randint(1000, 9999))+'/'
        try:
            os.stat(path1+temp)
        except:
            os.makedirs(path1+temp)
        try:
            os.stat(path1+temp+'/s/')
        except:
            os.makedirs(path1+temp+'/s/')
        img1=path1+file1
        img2s=path1+temp+file1
        if is_Spectral==1 or is_Spectral=='1':
            #add the spectral error
            if int(ismap)==1:
                ErrorMap, xx, yy, gt=read_raster(expected_spectral_error)#img2pandas(expected_spectral_error,'').values.tolist()
                OrigionalImage, xx, yy, gt=read_raster(img1)
                ErrorMap=OrigionalImage+ErrorMap*np.random.normal(0, 1, np.shape(ErrorMap))
                driverName= 'GTiff'    
                epsg_code=32630
                proj = osr.SpatialReference()
                proj.ImportFromEPSG(epsg_code)
                if is_Spatial==1 or is_Spatial=='1':
                    Dx,Dy,Dt=sp_errorsX[L],sp_errorsY[L],sp_errorsT[L]*3.14/180
                    Dt=0
                    gt=tuple(np.array(gt)+np.array([Dx,0,Dt,Dy,Dt,0]))
                CreateRaster(xx, yy, np.float32(numpy.absolute(OrigionalImage)), gt, proj,driverName,img2s)
            else:
                #ErrorMap, xx, yy, gt=read_raster(expected_spectral_error)#img2pandas(expected_spectral_error,'').values.tolist()
                OrigionalImage, xx, yy, gt=read_raster(img1)
                ErrorMap=OrigionalImage+np.random.normal(0, float(expected_spectral_error), np.shape(OrigionalImage))
                driverName= 'GTiff'    
                epsg_code=32630
                proj = osr.SpatialReference()
                proj.ImportFromEPSG(epsg_code)
                if is_Spatial==1 or is_Spatial=='1':
                    Dx,Dy,Dt=sp_errorsX[L],sp_errorsY[L],sp_errorsT[L]*3.14/180
                    Dt=0
                    gt=np.array(gt)+np.array([Dx,0,Dt,Dy,Dt,0])
                CreateRaster(xx, yy,np.float32(numpy.absolute(OrigionalImage)), gt, proj,driverName,img2s)
                #CreateRaster(xx, yy, numpy.absolute(ErrorMap), gt, proj,driverName,img2s)
                #np.random.normal(0, expected_spectral_error, np.shape(img))*ErrorMap[:,2] 
                #ErrorMap=img[:,:]
                #ErrorMap[:,2]=[expected_spectral_error]*len(img)
            #if isinstance(img, (list)):
            #    img=np.array(img)
            #img[:,-1]= img[:,-1]+ np.random.normal(0, expected_spectral_error, 1)*ErrorMap[:,-1]
        else:
            OrigionalImage, xx, yy, gt=read_raster(img1)
            driverName= 'GTiff' 
            epsg_code=32630
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(epsg_code)
            if is_Spatial==1 or is_Spatial=='1':
                    Dx,Dy,Dt=sp_errorsX[L],sp_errorsY[L],sp_errorsT[L]*3.14/180
                    Dt=0
                    gt=np.array(gt)+np.array([Dx,0,Dt,Dy,Dt,0])
            CreateRaster(xx, yy, OrigionalImage, gt, proj,driverName,img2s)
        img=object_zonal_stats(shp,img2s)
        img=DataFrame(img)
        img.fillna(0, inplace=True)
        img['FID'] = range(1, len(img) + 1)
        cols=list(img.columns.values)
        cols = [cols[-1]] + cols[:-1]
        #img = img[cols[:-1]].values.tolist()
        #img=list(filter(lambda a: a[-1] != 0.0,img))
        img = img.values.tolist()
        #delete temporary file 'img2'
        shutil.rmtree(path1+temp, ignore_errors=True)
        #read the image
        #img=img2pandas(path1,file1).values.tolist()
        images.append(img)
    return images,sp_errorsX,sp_errorsY,sp_errorsT

from operator import itemgetter
from itertools import groupby
def creat_table4corlation(rss,fields,RStable):#creat the table by adding the first RS band
    Biomass=[]
    for f1 in fields:
        #f1=list(map(lambda x: x.tolist(),f1))
        try:
            Biomass.extend(f1.tolist())
        except:
            Biomass.extend(f1)
    bs=(np.array((np.array(RStable)[:,2:4]),dtype=float)).tolist()
    #unique_list = list(map(itemgetter(0), groupby(bs)))
    unique_list=[]
    for i in bs:#i=bs[0]
        if i in unique_list:
            pass
        else:
            unique_list.append(i)
    rs_bands=[]
    for f1 in unique_list:#list(set(bands)):
        f1
        #extend all inputs of same dataset and same band 
        rs_band=[]
        j=-1
        for i in bs:
            i
            j+=1
            if i[0]==f1[0] and i[1]==f1[1]:
                #print(i)
                rs_band.extend(rss[j])
        rs_band=list(map(lambda x: list(x),rs_band))
        rs_bands.append(rs_band)
    Biomass=list(map(lambda x: list(x),Biomass))
    return Biomass,rs_bands
    
def regression_images(list1,pixelWidth,pixelHeight,list2):# list1 is the lowest resolution band 
    # filter the list with -1 values
    #list1=filter(lambda a: a[2] != -1, list1)
    #list2=list(filter(lambda a: a[2] > -0.000001 and a[2] < 100000, list2))
    list2=list(filter(lambda a: a[2] > 0 and a[2] < 100000, list2))
    column1=[]
    # for each pixle 
    for pixel1 in list1:
        #filter it spatially 
        try:
            x,y,v=pixel1
        except:
            x,y,v,v2=pixel1
        pixel2=list(filter(lambda a: a[0]>x-abs(pixelWidth)/2 and a[1] >y-abs(pixelHeight)/2  and a[0] <x+abs(pixelWidth)/2 and a[1] <y+abs(pixelHeight)/2, list2))
        # find the avarage of biomass
        try: 
            v2=np.mean(np.array(pixel2)[:,-1])
            # extend the list
            column1.append(v2)
            #print('ok')
        except:
            column1.append(np.nan)#print('empty')
    #bioRS=filter(lambda a: a[1] != -1, bioRS) 
    return column1
def creat_table4corlation_object(rss,fields,RStable):#creat the table by adding the first RS band
    Biomass=np.array(fields[0]).astype(float)
    for field1 in fields[1:]:
        Biomass[:,0]=Biomass[:,0].astype(float)+np.array(field1)[:,0].astype(float)
    bs=(np.array((np.array(RStable)[:,2:4]),dtype=float)).tolist()
    #unique_list = list(map(itemgetter(0), groupby(bs)))
    #unique_list = list(reversed(list(set(tuple(i) for i in bs))))
    unique_list=[]
    for i in bs:#i=bs[0]
        if i in unique_list:
            pass
        else:
            unique_list.append(i)
    rs_bands=[]
    for R1 in unique_list:#R1=unique_list[0]
        print(R1)
        #extend all inputs of same dataset and same band 
        (refBand,refDataset)=R1
        bs1=list(filter(lambda x: int(x[2])==int(refBand) and int(x[3])==int(refDataset) , RStable))
        i=RStable.index(bs1[0])
        rs_band=np.array(rss[i]).astype(float)
        #j=-1
        for ii in bs1[1:]:
            i=RStable.index(ii)
            rs_band[:,0]=rs_band[:,0].astype(float)+np.array(rss[i])[:,0].astype(float)
        #rs_bands.append(rs_band)
        rs_bands.append(list(rs_band[:,0]))
    Biomass=list(Biomass[:,0])
    #Biomass.append(rs_bands)
    return Biomass,rs_bands
# a function to add a list to an existed external file 
def Add_line_to_file(path,file1,line1):
    try:
        os.stat(path)
    except:
        os.makedirs(path)
    F=path+file1
    text1=str(line1)
    text1=text1.replace("[", "")
    text1=text1.replace("]", "")
    read=open(F,'a')
    read.write(text1)
    read.write('\n')
def Addtofile(path,file1,list1,NoOfColumns=3):
    try:
        os.stat(path)
    except:
        os.makedirs(path)
    F=path+file1
    text1=''
    for i in range(NoOfColumns):
        text1=text1+',x'+str(i+1)
    text1=text1[1:]
    read=open(F,'a')
    if NoOfColumns!=1:
        for line in list1:
            exec(text1+'= [float(value) for value in line]')
            for i in range(NoOfColumns):
                exec("read.write(str(x"+str(i+1)+'))')
                read.write(',')
            read.write('\n')
    else:
        for line in list1:
            exec("read.write(str(line))")
            read.write('\n')
    read=0
import os
def subset_image(extend_image, inDS, outDS,tolerance):
    path, file1= os.path.split(outDS)
    try:
        os.stat(path)
    except:
        os.makedirs(path)
    ds = gdal.Open(extend_image)
    (xmino,res1,tilt1,ymino,tilt2,res2)=ds.GetGeoTransform()
    xmaxo = xmino + (ds.RasterXSize * res1)
    ymaxo = ymino + (ds.RasterYSize * res2)
    xmino,ymino,xmaxo,ymaxo=xmino-tolerance,ymino+tolerance,xmaxo+tolerance,ymaxo-tolerance
    translate = 'gdal_translate -projwin %s %s %s %s %s %s' %(xmino,ymino , xmaxo,ymaxo , inDS, outDS)
    os.system(translate)
    #inDS='c:/Users/Public/Documents/ndvi.img'
    #outDS='c:/Users/Public/Documents/testsubset/ndvi.img'
    #extend_image='C:/Users/ahalboabidallah/Desktop/desktop/6akre2.tif'
    #tolerance=10
def calc_biomass_nonparametric(table1,Reg_file):
    #read the reg 
    Reg1 = joblib.load(file1) 
    #creat the xx
    #creat the parallel coordinate 
    #pridict yy
    #creat image

def CreateRaster(xx,yy,std,gt,proj,driverName,outFile):  
    '''
    Exports data to GTiff Raster
    '''
    #std = np.squeeze(std)
    #std[np.isinf(std)] = 0
    driver = gdal.GetDriverByName(driverName)
    rows,cols = np.shape(std)
    ds = driver.Create( outFile, cols, rows, 1, gdal.GDT_Float32)      
    if proj is not None:  
        ds.SetProjection(proj.ExportToWkt()) 
    ds.SetGeoTransform(gt)
    ss_band = ds.GetRasterBand(1)
    ss_band.WriteArray(std)
    ss_band.SetNoDataValue(0)
    ss_band.FlushCache()
    ss_band.ComputeStatistics(False)
    del ds
from gdalconst import *
def read_raster(in_raster,band=1):
    in_raster=in_raster
    ds = gdal.Open(in_raster)
    data = ds.GetRasterBand(band).ReadAsArray()
    #data[data<=0] = np.nan
    gt = ds.GetGeoTransform()
    xres = gt[1]
    yres = gt[5]
    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    del ds
    # create a grid of xy coordinates in the original projection
    xx, yy = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    return data, xx, yy, gt


def add_degrees(xx,degree):
    #colxx=list(xx.columns.values)
    #cols = cols[-1] + cols[:-1]
    #xx1 = xx[colxx[:-1]]
    #yy=colxx[-1]
    xx1=DataFrame(xx)
    cols=list(xx1.columns.values)
    list1=[]
    for i in cols:
        list1.append([i])
    cols22=list1[:]*1
    for d in range(degree-1):
        print('d',d+1)
        print(cols22)
        cols2=cols22[:]*1
        for i in list1:
            print ('i',i)
            for j in cols2:
                #print('d',d+1)
                #print ('ij',i,j)
                lllll=i+j
                lllll=sorted(lllll,key=lambda l:l)
                #print(lllll)
                cols22.append(lllll)
    cols22=list(set(tuple(i) for i in cols22))#
    cols22=sorted(cols22,key=lambda l:l)
    #cols22=sorted(cols22,key=lambda l:len(l))
    print(cols22)
    xx2=DataFrame([])
    c=0
    for i in cols22:
        c+=1
        xx2[c]=xx1[i[0]]
        for j in i[1:]:
            xx2[c]=xx2[c]*xx1[j]
            print (i,j)
    return xx2
def produce_map(RegFolder,table1):
    #read each input as xyz
    #the lowest resolution 
    regs=FindFilesInFolder(RegFolder,'*.pkl')
    for reg1 in regs:
        Reg.predict(xx)
        

def no_error(table2):#
    images=[]    
    sp_errorsX=spatial_error_xy(table2)
    sp_errorsY=spatial_error_xy(table2)
    sp_errorsT=spatial_error_t(table2)
    L=-1
    for row1 in table2:
        #row1=table2[1]
        print('row1',row1)
        if isinstance(row1[0][0], (list, tuple)):
            [path1,file1,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral,expected_spatial_error,expected_orientation_error,expected_spectral_error,ismap]=row1[0]
            #[path1,file1,expected_spatial_error,expected_spectral_error,expected_orientation_error,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral]=row1[0]
        else:
            [path1,file1,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral,expected_spatial_error,expected_orientation_error,expected_spectral_error,ismap]=row1
        #print (path1,file1)
        L+=1
        #read the image
        img=np.array(img2pandas(path1+'/',file1).values.tolist())
        #if ismap==1 or ismap=='1':
        #    print('expected_spectral_error',expected_spectral_error)
        #    ErrorMap=img2pandas(expected_spectral_error,'').values.tolist()
        #    expected_spectral_error=1
        #else:
        #    ErrorMap=img[:,:]*1
        #    ErrorMap[:,2]=[expected_spectral_error]*len(img)
        #if is_Spectral==1 or is_Spectral=='1':
        #    #add the spectral error
        #    if isinstance(ErrorMap, (list)):
        #        ErrorMap=np.array(ErrorMap)
        #    if isinstance(img, (list)):
        #        img=np.array(img)
        #    img[:,2]= img[:,2]+ np.random.normal(0, expected_spectral_error, len(img))*ErrorMap[:,2]
        #if is_Spatial==1 or is_Spatial=='1':
        #    if isinstance(img, (list)):
        #        img=np.array(img)
        #    #add the spatial error
        #    img[:,0]= img[:,0]+ sp_errorsX[L]
        #    img[:,1]= img[:,1]+ sp_errorsY[L]
        #    img_Cx,img_Cy=(min(img[:,0])+max(img[:,0]))/2,(min(img[:,1])+max(img[:,1]))/2
        #    img=((img-[img_Cx,img_Cy,0])*(frrot3(np.array([0,0,3.141592653589793/180*sp_errorsT[L]]))[0])+[img_Cx,img_Cy,0]).tolist()
        images.append(img)
    return images
def no_error_object(table2,shp):#table2,shp=field_data,Processing[1]
    images=[]    
    #sp_errorsX=spatial_error_xy(table2)
    #sp_errorsY=spatial_error_xy(table2)
    #sp_errorsT=spatial_error_t(table2)
    L=-1
    for row1 in table2:#row1=table2[0]
        #print(row1)
        if isinstance(row1[0][0], (list, tuple)):
            [path1,file1,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral,expected_spatial_error,expected_orientation_error,expected_spectral_error,ismap]=row1[0]
        else:
            [path1,file1,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral,expected_spatial_error,expected_orientation_error,expected_spectral_error,ismap]=row1
        #print (path1,file1)
        L+=1
        img1=path1+file1
        #img2s=path1+temp+file1
        #OrigionalImage, xx, yy, gt=read_raster(img1)
        #driverName= 'GTiff' 
        #epsg_code=32630
        #proj = osr.SpatialReference()
        #proj.ImportFromEPSG(epsg_code)
        #CreateRaster(xx, yy, OrigionalImage, gt, proj,driverName,img2s)
        img=object_zonal_stats(shp,img1)
        img=DataFrame(img)
        img.fillna(0, inplace=True)
        img[img < 0] = 0
        img['FID'] = range(1, len(img) + 1)
        cols=list(img.columns.values)
        cols = [cols[-1]] + cols[:-1]
        #img = img[cols[:-1]].values.tolist()
        #img=list(filter(lambda a: a[-1] != 0.0,img))
        img = img.values.tolist()
        #delete temporary file 'img2'
        #shutil.rmtree(path1+temp, ignore_errors=True)
        #read the image
        #img=img2pandas(path1,file1).values.tolist()
        images.append(img)
    return images   
# load yys  
# find the std for each column
# built the model of error
def error_pridiction_tool(table1):#
    #table1=tables[0]
    print('>',end="")
    #if 1==1:
    [field_data,RStable,Processing,model_spec]=table1[:]
    RStable2=[]
    i=0
    #to include all subsets
    for fd1 in field_data:
        #fd1=field_data[0]
        #print('fd1',fd1)
        i+=1
        RStable1=RStable[:]
        for rs in RStable1:
            #rs=RStable1[2]
            #print('f',rs)
            rs1=[rs[0]+'subset/'+str(i)+'/']
            if rs[-1]==1 or rs[-1]=='1':
                expected_spectral_error=rs[0]+'subset/'+str(i)+'/error'+rs[1]
            else:
                expected_spectral_error=rs[-2]
            rs1.extend(rs[1:-2])
            rs1.extend([expected_spectral_error,rs[-1]])
            RStable2.append(rs1)
    RStable=RStable2[:]
    if Processing[0]=='Pixel':#if it is a pixel based model
        RStable=sorted(RStable,key=lambda l:-float(l[4]))#sort by resolution
        Lowest_resolution=float(RStable[0][4])# find lowest resolution
        #fields,Fsp_errorsX,Fsp_errorsY,Fsp_errorsT=add_error(field_data)# adds errors to the field data 
        #fields=list(filter(lambda a: a[2] > 0 and a[2] < 100000,fields[0]))
        rss0=no_error(RStable)
        fields0=no_error(field_data)
        Biomass0,rs_bands0=creat_table4corlation(rss0,fields0,RStable)
        Biomass0= np.asarray(Biomass0, dtype=float)
        while len(Biomass0[0])==1:# to get rid of unnecessary brackets
            Biomass0=list(map(lambda x:x[0], Biomass0))
        rs_bands0[0]=list(filter(lambda a: a[2] > 0 and a[2] < 100000,rs_bands0[0]))   
        yy0=regression_images(rs_bands0[0],Lowest_resolution,Lowest_resolution,Biomass0)
        xx0=[np.array(rs_bands0[0])[:,2].tolist()]
        for i in rs_bands0[1:]:
            #print('i')
            try:
                xx0.append(regression_images(rs_bands0[0],Lowest_resolution,Lowest_resolution,i))#
            except:
                print ('error')
        xx0.append(yy0)# now the pixel based table is ready for the model
        xx0=pd.DataFrame(xx0)
        xx0=xx0.transpose()
        xx0=xx0.dropna()
    else:
        #if 1==1:
        fields0=no_error_object(field_data,Processing[1])
        rss0=no_error_object(RStable,Processing[1])
        Biomass0,rs_bands0=creat_table4corlation_object(rss0,fields0,RStable)
        xx0=(np.array(rs_bands0)).tolist()
        #xx=np.transpose(np.array(rs_bands)).tolist()
        xx0.extend([Biomass0])
        xx0=(np.transpose(np.array(xx0)).tolist())
        #filter sum(n < 0 for n in nums)
        xx0=list(filter(lambda a: sum(n <= 0 for n in a)==0,xx0))
        xx0=(np.transpose(np.array(xx0)).tolist())
        xx0=pd.DataFrame(xx0)
        xx0=xx0.transpose()
        xx0=xx0.dropna()
    #load the yys file
    yy=pd.read_csv('C:/Users/ahalboabidallah/Desktop/mont_carlo/YY.csv',header=None)
    stds=[]
    for column in yy:
        stds.append(yy[column].values.std(ddof=1))
    
    if model_spec[0]=='Parametric_Regression':#if the model is regression
        xx0=xx0.values.tolist()
        yy0=list(map(lambda x:x[-1], xx0))
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/analysis/','YY0.csv',yy0)
        xx0=list(map(lambda x:x[0:-1], xx0))
        if model_spec[1]>1:
            xx0=add_degrees(xx0,model_spec[1])
        model = sm.api.OLS(stds,xx0)#len(final_table[1:].values.tolist())==len(final_table[0].values.tolist())
        results = model.fit()
        print(results.summary())    
        #print('Parameters: ', results.params)
        #print('R2: ', results.rsquared)
             #record.append([i,[results.params],results.rsquared])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/analysis/','RS_STD_results_rsquared.csv',[results.rsquared])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/analysis/','RS_STD_results_params.csv',list(results.params))
        #Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/','maxmins.csv',maxmins)
        file1='C:/Users/ahalboabidallah/Desktop/reg/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        joblib.dump(model, file1)
        yy1=results.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/analysis/','RS_STD_YY.csv',yy1.tolist())
        #print('yy1',yy1)
    elif model_spec[0]=='Support_Vector_Machine':#if the model is 'Support_Vector_Machine'
        xx0=np.array(list(map(lambda x:x[0:-1], xx0)))
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        kernel,degree=model_spec[1:]
        Reg = SVR(kernel=kernel,C=2000, epsilon=0.2,degree=degree)
        Reg.fit(xx0, stds) 
        R2=np.corrcoef([stds,Reg.predict(xx0)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/SVM/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/SVM/','RS_STD_results_rsquared.csv',[file1,R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/SVM/','RS_STD_maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') 
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/analysis/','RS_STD_YY.csv',yy1.tolist())
    elif model_spec[0]=='Neural Network':#if the model is 'Neural Network'
        xx0=np.array(list(map(lambda x:x[0:-1], xx0)))
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        NumberOfLayers,Layers=model_spec[1:]
        #
        Reg = MLPRegressor(hidden_layer_sizes=Layers,  activation='relu', solver='adam', alpha=0.001, batch_size='auto',learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-05)
        Reg.fit(xx0, stds) 
        R2=np.corrcoef([yy,Reg.predict(xx)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/Neural/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','files.csv',[file1])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','RS_STD_results_rsquared.csv',[R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','RS_STD_maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') 
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/analysis/','RS_STD_YY.csv',yy1.tolist())
    elif model_spec[0]=='Gaussian_Process':#if the model is 'Gaussian_Process'
        xx0=np.array(list(map(lambda x:x[0:-1], xx0)))
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        Gaussian_Process_kernal=model_spec[1]
        Reg = GaussianProcessRegressor(kernel=Gaussian_Process_kernal)
        Reg.fit(xx0, stds) 
        R2=np.corrcoef([stds,Reg.predict(xx0)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/Neural/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','files.csv',[file1])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','RS_STD_results_rsquared.csv',[R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','RS_STD_maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') 
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/analysis/','RS_STD_YY.csv',yy1.tolist())
    elif model_spec[0]=='Random_Forest':#if the model is 'Random_Forest'
        Random_Forest_Bootstrap=model_spec[1]
        xx0=np.array(list(map(lambda x:x[0:-1], xx0)))
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        Gaussian_Process_kernal=model_spec[1]
        Reg = GaussianProcessRegressor(kernel=Gaussian_Process_kernal)
        Reg.fit(xx0, stds) 
        Reg = RandomForestRegressor(bootstrap=Random_Forest_Bootstrap)
        Reg.fit(xx0, stds) 
        R2=np.corrcoef([stds,Reg.predict(xx0)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/Neural/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','files.csv',[file1])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','RS_STD_results_rsquared.csv',[R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Neural/','RS_STD_maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') 
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/analysis/','RS_STD_YY.csv',yy1.tolist())
    elif model_spec[0]=='K_Nearest_Neighbours':#if the model is 'K_Nearest_Neighbours'
        K_Nearest_Neighbours_weights,K_Nearest_Neighbours_k=model_spec[1:]
        xx0=np.array(list(map(lambda x:x[0:-1], xx0)))
        xx0=xx0.values.tolist()
        xx0,maxmins=normalize(xx0,maxmins=maxmins)
        xx0=list(xx0)
        #
        Reg = KNeighborsRegressor(n_neighbors=2,weights =K_Nearest_Neighbours_weights)
        Reg.fit(xx0, stds) 
        R2=np.corrcoef([std,Reg.predict(xx0)])[0,1]
        file1='C:/Users/ahalboabidallah/Desktop/Random_Forest/'+str(randint(1000, 9999))+str(randint(1000, 9999))+'.pkl'
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Random_Forest/','files.csv',[file1])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Random_Forest/','results_rsquared.csv',[R2])
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/Random_Forest/','maxmins.csv',[maxmins])
        #read the last in the csv file
        #add line to the csv file
        #s = pickle.dumps(Reg)
        joblib.dump(Reg, file1)#joblib.dump(Reg, 'filename.pkl') 
        #Reg1 = joblib.load(file1) #Reg = joblib.load('filename.pkl') '''
        yy1=Reg.predict(xx0)
        Add_line_to_file('C:/Users/ahalboabidallah/Desktop/mont_carlo/analysis/','RS_STD_YY.csv',yy1.tolist())
        print(np.shape(xx0))
    #return yy1,xx0
