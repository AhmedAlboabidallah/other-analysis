import gdal, osr
import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.lib.stride_tricks import as_strided as ast
import dask.array as da
from joblib import Parallel, delayed, cpu_count
from joblib import *
import joblib
import os
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage
import glob

def FindFilesInFolder(folder1,extension):                            
    dr=os.getcwd()
    os.chdir(folder1)
    files=glob.glob(extension)
    os.chdir(dr)
    return files
def im_resize(im,Nx,Ny):
    '''
    resize array by bivariate spline interpolation
    '''
    ny, nx = np.shape(im)
    xx = np.linspace(0,nx,Nx)
    yy = np.linspace(0,ny,Ny)
    try:
        im = da.from_array(im, chunks=1000)   #dask implementation
    except:
        pass
    newKernel = RectBivariateSpline(np.r_[:ny],np.r_[:nx],im)
    return newKernel(yy,xx)

def p_me(Z, win):
    '''
    loop to calculate greycoprops
    '''
    try:
        glcm = greycomatrix(Z, [5], [0], 256, symmetric=True, normed=True)
        cont = greycoprops(glcm, 'contrast')
        diss = greycoprops(glcm, 'dissimilarity')
        homo = greycoprops(glcm, 'homogeneity')
        eng = greycoprops(glcm, 'energy')
        corr = greycoprops(glcm, 'correlation')
        ASM = greycoprops(glcm, 'ASM')
        return (cont, diss, homo, eng, corr, ASM)
    except:
        return (0,0,0,0,0,0)

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

def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass
   raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a, ws, ss = None, flatten = True):
    '''
    Source: http://www.johnvinyard.com/blog/?p=268#more-268
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''      
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = list(filter(lambda i : i != 1,dim))
    return a.reshape(dim), newshape

def CreateRaster(xx,yy,std,gt,proj,driverName,outFile):  
    '''
    Exports data to GTiff Raster
    '''
    std = np.squeeze(std)
    std[np.isinf(std)] = -99
    driver = gdal.GetDriverByName(driverName)
    rows,cols = np.shape(std)
    ds = driver.Create( outFile, cols, rows, 1, gdal.GDT_Float32)      
    if proj is not None:  
        ds.SetProjection(proj.ExportToWkt()) 
    ds.SetGeoTransform(gt)
    ss_band = ds.GetRasterBand(1)
    ss_band.WriteArray(std)
    ss_band.SetNoDataValue(-99)
    ss_band.FlushCache()
    ss_band.ComputeStatistics(False)
    del ds
from gdalconst import *

def seperate_to_bands(in_raster):#in_raster='C:/Users/ahalboabidallah/Desktop/20151209-02/pc12320151209-02.img'
    folder1, file1= os.path.split(in_raster)#ntpath.split(raster)#raster
    dataset = gdal.Open(in_raster, GA_ReadOnly )
    bands=dataset.RasterCount
    for b in range(bands):
        band=b+1
        BandFile   = folder1+'/'+'Band'+str(band)+file1
        print('BandFile=',BandFile)
        merge, xx, yy, gt = read_raster(in_raster,band=band)
        merge[np.isnan(merge)] = 0
        print('merge[100,100]',merge[100,100])
        driverName= 'GTiff'    
        epsg_code=32630
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(epsg_code)
        CreateRaster(xx, yy, merge, gt, proj,driverName,BandFile)
        dataset = gdal.Open(in_raster)
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        dataset2 = gdal.Open(BandFile, gdal.GA_Update )
        dataset2.SetGeoTransform( geotransform )
        dataset2.SetProjection( projection )
#raster ='C:/Users/ahalboabidallah/Desktop/PCs/pc123_20150630_1.tif'#pc123_20150630-2
#seperate_to_bands(raster)
#raster ='C:/Users/ahalboabidallah/Desktop/PCs//pc123_20150630-2.tif'
#seperate_to_bands(raster)
        

# origional texture data
if __name__ == '__main__':  
    #in_raster ='C:/Users/ahalboabidallah/Desktop/20151209-01/20150630-WV3-NPA-PIX-GBR-MS8-L3-01.tif' #Path to input raster
    in_raster ='C:/Users/ahalboabidallah/Desktop/20151209-02/20150630-WV3-NPA-PIX-GBR-MS8-L3-02.tif'
    #in_raster ='C:/Users/ahalboabidallah/Desktop/HiResolution/image1band5.tif'
    win = 15
    convolution_size=win
    is_filter=1
    image='20150630_2_'
    for b in range(1):
        meter = str(win/4)
        band=b+1
        #is_filter=1
        #Define output file names
        BandFile   = 'C:/Users/ahalboabidallah/Desktop/texture/Image'+image+'Band'+str(band)+'.tif'
        contFile   = 'C:/Users/ahalboabidallah/Desktop/texture/Image'+image+'Band'+str(band)+'cont'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        dissFile   = 'C:/Users/ahalboabidallah/Desktop/texture/Image'+image+'Diss'+str(band)+'Diss'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        homoFile   = 'C:/Users/ahalboabidallah/Desktop/texture/Image'+image+'Band'+str(band)+'Homo'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        energyFile = 'C:/Users/ahalboabidallah/Desktop/texture/Image'+image+'Band'+str(band)+'Energy'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        corrFile   = 'C:/Users/ahalboabidallah/Desktop/texture/Image'+image+'Band'+str(band)+'Corr'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        ASMFile    = 'C:/Users/ahalboabidallah/Desktop/texture/Image'+image+'Band'+str(band)+'ASM'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        merge, xx, yy, gt = read_raster(in_raster,band=band)
        merge[np.isnan(merge)] = 0
        driverName= 'GTiff'    
        epsg_code=32630
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(epsg_code)
        CreateRaster(xx, yy, merge, gt, proj,driverName,BandFile)
        if is_filter==1:
            ker = (1 / convolution_size**2) * np.ones((convolution_size, convolution_size))
            merge=ndimage.convolve(merge, ker)
        Z,ind = sliding_window(merge,(win,win),(win,win))
        Ny, Nx = np.shape(merge)
        w = Parallel(n_jobs = 7, verbose=0)(delayed(p_me)(Z[k],win) for k in range(len(Z)))
        cont = [a[0] for a in w]
        diss = [a[1] for a in w]
        homo = [a[2] for a in w]
        eng  = [a[3] for a in w]
        corr = [a[4] for a in w]
        ASM  = [a[5] for a in w]
        #Reshape to match number of windows
        plt_cont = np.reshape(cont , ( ind[0], ind[1] ) )
        plt_diss = np.reshape(diss , ( ind[0], ind[1] ) )
        plt_homo = np.reshape(homo , ( ind[0], ind[1] ) )
        plt_eng = np.reshape(eng , ( ind[0], ind[1] ) )
        plt_corr = np.reshape(corr , ( ind[0], ind[1] ) )
        plt_ASM =  np.reshape(ASM , ( ind[0], ind[1] ) )
        del cont, diss, homo, eng, corr, ASM
        #Resize Images to receive texture and define filenames
        contrast = im_resize(plt_cont,Nx,Ny)
        contrast[merge==0]=np.nan
        dissimilarity = im_resize(plt_diss,Nx,Ny)
        dissimilarity[merge==0]=np.nan    
        homogeneity = im_resize(plt_homo,Nx,Ny)
        homogeneity[merge==0]=np.nan
        energy = im_resize(plt_eng,Nx,Ny)
        energy[merge==0]=np.nan
        correlation = im_resize(plt_corr,Nx,Ny)
        correlation[merge==0]=np.nan
        ASM = im_resize(plt_ASM,Nx,Ny)
        ASM[merge==0]=np.nan
        del plt_cont, plt_diss, plt_homo, plt_eng, plt_corr, plt_ASM
        del w,Z,ind,Ny,Nx
        CreateRaster(xx, yy, contrast, gt, proj,driverName,contFile) 
        CreateRaster(xx, yy, dissimilarity, gt, proj,driverName,dissFile)
        CreateRaster(xx, yy, homogeneity, gt, proj,driverName,homoFile)
        CreateRaster(xx, yy, energy, gt, proj,driverName,energyFile)
        CreateRaster(xx, yy, correlation, gt, proj,driverName,corrFile)
        CreateRaster(xx, yy, ASM, gt, proj,driverName,ASMFile)
        dataset = gdal.Open( in_raster )
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        for i in [BandFile,contFile,dissFile,homoFile,energyFile,corrFile,ASMFile]:
            dataset2 = gdal.Open( i, gdal.GA_Update )
            dataset2.SetGeoTransform( geotransform )
            dataset2.SetProjection( projection )
        dataset ,dataset2 = None,None
        del contrast, merge, xx, yy, gt,meter, dissimilarity, homogeneity, energy, correlation, ASM


#monte carlo calculation
mc=20
#in_raster ='C:/Users/ahalboabidallah/Desktop/20151209-02/20150630-WV3-NPA-PIX-GBR-MS8-L3-02.tif'
in_raster ='C:/Users/ahalboabidallah/Desktop/20151209-01/20150630-WV3-NPA-PIX-GBR-MS8-L3-01.tif'
if __name__ == '__main__':  
    for i in range(mc):
        print('i',i)
        #in_raster ='C:/Users/ahalboabidallah/Desktop/20151209-02/20150630-WV3-NPA-PIX-GBR-MS8-L3-02.tif'
        #in_raster ='C:/Users/ahalboabidallah/Desktop/HiResolution/image1band5.tif'
        win = 13
        convolution_size=win
        is_filter=1
        image='20150630_1_'+str(i)+'_'
        #print('b',b)
        meter = str(win/4)
        band=5
        #b=5
        #is_filter=1
        #Define output file names
        #BandFile   = 'C:/Users/ahalboabidallah/Desktop/texture/mc/Image'+image+'Band'+str(band)+'.tif'
        #contFile   = 'C:/Users/ahalboabidallah/Desktop/texture/mc/Image'+image+'Band'+str(band)+'cont'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        #dissFile   = 'C:/Users/ahalboabidallah/Desktop/texture/mc/Image'+image+'Diss'+str(band)+'Diss'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        #homoFile   = 'C:/Users/ahalboabidallah/Desktop/texture/mc/Image'+image+'Band'+str(band)+'Homo'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        #energyFile = 'C:/Users/ahalboabidallah/Desktop/texture/mc/Image'+image+'Band'+str(band)+'Energy'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        #corrFile   = 'C:/Users/ahalboabidallah/Desktop/texture/mc/Image'+image+'Band'+str(band)+'Corr'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        ASMFile    = 'F:/texture/mc1/Image'+image+'Band'+str(band)+'ASM'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif'
        merge, xx, yy, gt = read_raster(in_raster,band=band)
        merge[np.isnan(merge)] = 0
        #########################
        #add errors to merge
        error=np.random.normal(0, 0.02, np.shape(merge))
        merge=merge+merge*error
        #########################
        driverName= 'GTiff'    
        epsg_code=32630
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(epsg_code)
        #CreateRaster(xx, yy, merge, gt, proj,driverName,BandFile)
        if is_filter==1:
            ker = (1 / convolution_size**2) * np.ones((convolution_size, convolution_size))
            merge=ndimage.convolve(merge, ker)
        Z,ind = sliding_window(merge,(win,win),(win,win))
        Ny, Nx = np.shape(merge)
        w = Parallel(n_jobs = 7, verbose=0)(delayed(p_me)(Z[k],win) for k in range(len(Z)))
        #cont = [a[0] for a in w]
        #diss = [a[1] for a in w]
        #homo = [a[2] for a in w]
        #eng  = [a[3] for a in w]
        #corr = [a[4] for a in w]
        ASM  = [a[5] for a in w]
        #Reshape to match number of windows
        #plt_cont = np.reshape(cont , ( ind[0], ind[1] ) )
        #plt_diss = np.reshape(diss , ( ind[0], ind[1] ) )
        #plt_homo = np.reshape(homo , ( ind[0], ind[1] ) )
        #plt_eng = np.reshape(eng , ( ind[0], ind[1] ) )
        #plt_corr = np.reshape(corr , ( ind[0], ind[1] ) )
        plt_ASM =  np.reshape(ASM , ( ind[0], ind[1] ) )
        del ASM
        #Resize Images to receive texture and define filenames
        #contrast = im_resize(plt_cont,Nx,Ny)
        #contrast[merge==0]=np.nan
        #dissimilarity = im_resize(plt_diss,Nx,Ny)
        #dissimilarity[merge==0]=np.nan    
        #homogeneity = im_resize(plt_homo,Nx,Ny)
        #homogeneity[merge==0]=np.nan
        #energy = im_resize(plt_eng,Nx,Ny)
        #energy[merge==0]=np.nan
        #correlation = im_resize(plt_corr,Nx,Ny)
        #correlation[merge==0]=np.nan
        ASM = im_resize(plt_ASM,Nx,Ny)
        ASM[merge==0]=np.nan
        #del plt_cont, plt_diss, plt_homo, plt_eng, plt_corr, plt_ASM
        del w,Z,ind,Ny,Nx
        #CreateRaster(xx, yy, contrast, gt, proj,driverName,contFile) 
        #CreateRaster(xx, yy, dissimilarity, gt, proj,driverName,dissFile)
        #CreateRaster(xx, yy, homogeneity, gt, proj,driverName,homoFile)
        #CreateRaster(xx, yy, energy, gt, proj,driverName,energyFile)
        #CreateRaster(xx, yy, correlation, gt, proj,driverName,corrFile)
        CreateRaster(xx, yy, ASM, gt, proj,driverName,ASMFile)
        dataset = gdal.Open( in_raster )
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        #for i in [BandFile,contFile,dissFile,homoFile,energyFile,corrFile,ASMFile]:#
        for i in [ASMFile]:
            i
            dataset2 = gdal.Open( i, gdal.GA_Update )
            dataset2.SetGeoTransform( geotransform )
            dataset2.SetProjection( projection )
        dataset ,dataset2 = None,None
        #del contrast, merge, xx, yy, gt,meter, dissimilarity, homogeneity, energy, correlation, ASM
        del merge, xx, yy, gt,meter, ASM


objectedWithError=FindFilesInFolder('F:/texture/mc1/','*Band5ASM13*')

import numpy as np
NoError, xx, yy, gt = read_raster('F:/texture/mc1/'+objectedWithError[0],1)
NoError[np.isnan(NoError)] = 0
Error=NoError*0
for im1 in objectedWithError[1:]:
    Errori, xx, yy, gt = read_raster('F:/texture/mc1/'+im1,1)
    Errori[np.isnan(NoError)] = 0
    s=np.isnan(Errori)
    Errori[s]=0.0
    Error=Error+(Errori-NoError)*(Errori-NoError)/len(objectedWithError)
CreateRaster(xx, yy, Error, gt, proj,driverName,'F:/texture/mc1/errorImage'+image+'Band'+str(band)+'ASM'+str(win)+'filter'+str(is_filter)+'conv'+str(convolution_size)+'.tif') 

in_raster ='C:/Users/ahalboabidallah/Desktop/20151209-01/20150630-WV3-NPA-PIX-GBR-MS8-L3-01.tif'
mc=20
image='20150630_1_'
if __name__ == '__main__':
    allimages=[]
    for i in range(mc):
        print('i',i)
        bands=[]
        for b in range(8):
            band=b+1
            merge, xx, yy, gt = read_raster(in_raster,band=band)
            error=np.random.normal(0, 0.02, np.shape(merge))
            merge=merge+merge*error
            BandFile   = 'F:/texture/mcPC/Image'+image+'Band'+str(band)+'MC'+str(i)+'.tif'
            driverName= 'GTiff'    
            epsg_code=32630
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(epsg_code)
            CreateRaster(xx, yy, merge, gt, proj,driverName,BandFile)
            bands.append(BandFile)
        allimages.append(bands)

objectedWithError=FindFilesInFolder('C:/Users/Public/Documents/','test*.img')
NoError, xx, yy, gt = read_raster('C:/Users/Public/Documents/'+objectedWithError[0],1)
NoError[np.isnan(NoError)] = 0
Error=NoError*0
for im1 in objectedWithError[1:]:
    Errori, xx, yy, gt = read_raster('C:/Users/Public/Documents/'+im1,1)
    Errori[np.isnan(NoError)] = 0
    #s=np.isnan(Errori)
    #Errori[s]=0.0
    Error=Error+(Errori-NoError)*(Errori-NoError)/len(objectedWithError)
driverName= 'GTiff'    
epsg_code=32630
proj = osr.SpatialReference()
proj.ImportFromEPSG(epsg_code)
CreateRaster(xx, yy, (Error)**0.5, gt, proj,driverName,'F:/texture/mcpc/errorImage'+image+'pc.tif') 
