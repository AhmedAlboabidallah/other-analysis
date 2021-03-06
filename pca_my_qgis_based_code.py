# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:07:36 2017

@author: ahalboabidallah
"""

# get small DEM from https://github.com/GeospatialPython/Learn/blob/master/dem.zip

# Prepare the environment
import sys
sys.path.extend(['C:/PROGRA~1/QGIS2~1.18/apps/qgis/./python/plugins\\processing', 'C:/PROGRA~1/QGIS2~1.18/apps/qgis/./python', u'C:/Users/ahalboabidallah/.qgis2/python', u'C:/Users/ahalboabidallah/.qgis2/python/plugins', 'C:/PROGRA~1/QGIS2~1.18/apps/qgis/./python/plugins', 'C:\\PROGRA~1\\QGIS2~1.18\\bin\\python27.zip', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\DLLs', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\plat-win', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\lib-tk', 'C:\\PROGRA~1\\QGIS2~1.18\\bin', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\PIL', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\jinja2-2.7.2-py2.7.egg', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\markupsafe-0.23-py2.7-win-amd64.egg', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\win32', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\win32\\lib', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\Pythonwin', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\Shapely-1.2.18-py2.7-win-amd64.egg', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\wx-2.8-msw-unicode', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\xlrd-0.9.2-py2.7.egg', 'C:\\PROGRA~1\\QGIS2~1.18\\apps\\Python27\\lib\\site-packages\\xlwt-0.7.5-py2.7.egg', u'C:/Users/ahalboabidallah/.qgis2//python', u'C:/Users/ahalboabidallah']) # Folder where Processing is located
from qgis.core import *
from PyQt4.QtGui import *
app = QApplication([], True)
QgsApplication.setPrefixPath("C:/OSGeo4W64/apps/qgis", True)
QgsApplication.initQgis()
print ("QGIS successfully Initialised")
# Prepare processing framework 
# https://gis.stackexchange.com/questions/129513/how-can-i-access-processing-with-python

from processing.core.Processing import Processing
Processing.initialize()
from processing.tools import *

from PyQt4.QtCore import QFileInfo
from osgeo import gdal
from osgeo.gdalconst import *
import numpy
import sys 
#sys.path.append('C:\Program Files\QGIS 2.18\apps\Python27\Lib\site-packages\networkx\algorithms')#"C:\Program Files\QGIS Pisa\apps\Python27\Lib") 
#import core
#'C:/Program Files/QGIS 2.18/apps/qgis/bin/'
from qgis.core import QgsRasterLayer, QgsMapLayerRegistry

def pca(inputRasterFileName, outputRasterFileName, outPCBands):
    # Open the input raster file
    # register the gdal drivers
    gdal.AllRegister()

    # Open and assign the contents of the raster file to a dataset
    dataset = gdal.Open(inputRasterFileName, GA_ReadOnly)

    # Compute raster correlation matrix    
    bandMean = numpy.empty(dataset.RasterCount)
    for i in xrange(dataset.RasterCount):
        band = dataset.GetRasterBand(i+1).ReadAsArray(0, 0,
                                                      dataset.RasterXSize,
                                                      dataset.RasterYSize)
        bandMean[i] = numpy.amin(band, axis = None)

    corrMatrix = numpy.empty((dataset.RasterCount, dataset.RasterCount))
    for i in xrange(dataset.RasterCount):
        band = dataset.GetRasterBand(i+1)
        bandArray = band.ReadAsArray(0, 0,
                                     dataset.RasterXSize,
                                     dataset.RasterYSize).astype(numpy.float).flatten()

        bandArray = bandArray - bandMean[i]
        corrMatrix[i][i] = numpy.corrcoef(bandArray, bandArray)[0][1]

    band = None
    bandArray = None

    for i in xrange(1, dataset.RasterCount + 1):
        band1 = dataset.GetRasterBand(i)
        bandArray1 = band1.ReadAsArray(0, 0,
                                       dataset.RasterXSize,
                                       dataset.RasterYSize).astype(numpy.float).flatten()
        bandArray1 = bandArray1 - bandMean[i - 1]

        for j in xrange(i + 1, dataset.RasterCount + 1):
            band2 = dataset.GetRasterBand(j)
            bandArray2 = band2.ReadAsArray(0, 0,
                                           dataset.RasterXSize,
                                           dataset.RasterYSize).astype(numpy.float).flatten()

            bandArray2 = bandArray2 - bandMean[j - 1]

            corrMatrix[j - 1][i - 1] = corrMatrix[i - 1][j - 1] = numpy.corrcoef(bandArray1, bandArray2)[0][1]

    # Calculate the eigenvalues and the eigenvectors of the covariance
    # matrix and calculate the principal components
    # debug print
    #print corrMatrix
    eigenvals, eigenvectors = numpy.linalg.eig(corrMatrix)

    # Just for testing
    #print eigenvals
    #print eigenvectors

    # Create a lookup table and sort it according to
    # the index of the eigenvalues table
    # In essence the following code sorts the eigenvals
    indexLookupTable = [i for i in xrange(dataset.RasterCount)]

    for i in xrange(dataset.RasterCount):
        for j in xrange(dataset.RasterCount - 1, i, -1):
            if eigenvals[indexLookupTable[j]] > eigenvals[indexLookupTable[j - 1]]:
                temp = indexLookupTable[j]
                indexLookupTable[j] = indexLookupTable[j - 1]
                indexLookupTable[j - 1] = temp

    # Calculate and save the resulting dataset
    driver = gdal.GetDriverByName("GTiff")
    outDataset = driver.Create(outputRasterFileName,
                               dataset.RasterXSize,
                               dataset.RasterYSize,
                               outPCBands,
                               gdal.GDT_Float32)

    for i in xrange(outPCBands):
        pc = 0
        for j in xrange(dataset.RasterCount):
            band = dataset.GetRasterBand(j + 1)
            bandAdjustArray = band.ReadAsArray(0, 0, dataset.RasterXSize,
                                               dataset.RasterYSize).astype(numpy.float) - bandMean[j]

            pc = pc + eigenvectors[j, indexLookupTable[i]] * bandAdjustArray

        pcband = outDataset.GetRasterBand(i + 1)
        pcband.WriteArray(pc)

    # Check if there is geotransformation or geoprojection
    # in the input raster and set them in the resulting dataset
    if dataset.GetGeoTransform() != None:
        outDataset.SetGeoTransform(dataset.GetGeoTransform())

    if dataset.GetProjection() != None:
        outDataset.SetProjection(dataset.GetProjection())


    # write the statistics of the PCA into a file
    # first organize the statistics into lists
    corrBandBand = [['' for i in xrange(dataset.RasterCount + 1)] for j in xrange(dataset.RasterCount + 1)]
    corrBandBand[0][0] = "Correlation Matrix"
    for j in xrange(1, 1 + dataset.RasterCount):
        header = 'Band' + str(j)
        corrBandBand[0][j] = header
    for i in xrange(1, 1 + dataset.RasterCount):
        vertical = 'Band' + str(i)
        corrBandBand[i][0] = vertical
    for i in xrange(1, 1 + dataset.RasterCount):
        for j in xrange(1, 1 + dataset.RasterCount):
            corrBandBand[i][j] = "%.3f" % corrMatrix[i - 1, j - 1]

    covBandPC = [['' for i in xrange(dataset.RasterCount + 1)] for j in xrange(dataset.RasterCount + 1)]
    covBandPC[0][0] = "Cov.Eigenvectors"
    for j in xrange(1, 1 + dataset.RasterCount):
        header = 'PC' + str(j)
        covBandPC[0][j] = header
    for i in xrange(1, 1 + dataset.RasterCount):
        vertical = "Band" + str(i)
        covBandPC[i][0] = vertical
    for i in xrange(1, 1 + dataset.RasterCount):
        for j in xrange(1, 1 + dataset.RasterCount):
            covBandPC[i][j] = "%.3f" % eigenvectors[i - 1, indexLookupTable[j - 1]]


    covEigenvalMat = [['' for i in xrange(dataset.RasterCount + 1)] for j in xrange(5)]
    covEigenvalMat[0][0] = "Bands"
    covEigenvalMat[1][0] = "Cov.Eigenvalues"
    covEigenvalMat[2][0] = "Sum of Eigenvalues"
    covEigenvalMat[3][0] = "Eigenvalues/Sum"
    covEigenvalMat[4][0] = "Percentages(%)"

    eigvalSum = 0.0
    sum = numpy.sum(eigenvals)
    for i in xrange(dataset.RasterCount):
        covEigenvalMat[0][i + 1] = "PC" + str(i + 1)
        covEigenvalMat[1][i + 1] = "%.3f" % eigenvals[indexLookupTable[i]]
        eigvalSum = eigvalSum + eigenvals[indexLookupTable[i]]
        covEigenvalMat[2][i + 1] = "%.3f" % eigvalSum
        covEigenvalMat[3][i + 1] = "%.3f" % (eigvalSum / sum)
        covEigenvalMat[4][i + 1] = "%.1f" % (eigvalSum / sum * 100.0)

    # Debug printout
    #print corrBandBand
    #print covBandPC
    #print covEigenvalMat

    statText = ""
    statFileName = outputRasterFileName.split('.')[0] + "_statistics.txt"
    statFile = open(statFileName, "w")

    for i in xrange(len(corrBandBand)):
        for j in xrange(len(corrBandBand)): # symmetrical matrix
            statText = statText + corrBandBand[i][j]
            if (j < len(corrBandBand[0]) - 1):
                statText = statText + " "
        statText = statText + "\n"

    statText = statText + "\n"

    for i in xrange(len(covBandPC)):
        for j in xrange(len(covBandPC[0])):
            statText = statText + covBandPC[i][j]
            if (j < len(covBandPC[0]) - 1):
                statText = statText + " "
        statText = statText + "\n"

    statText = statText + "\n"

    for i in xrange(len(covEigenvalMat)):
        for j in xrange(len(covEigenvalMat[0])):
            statText = statText + covEigenvalMat[i][j]
            if (j < len(covEigenvalMat[0]) - 1):
                statText = statText + " "
        statText = statText + "\n"

    statFile.write(statText)
    statFile.close()
    dataset = None
    outDataset = None

    # insert the output raster into QGIS interface
    outputRasterFileInfo = QFileInfo(outputRasterFileName)
    baseName = outputRasterFileInfo.baseName()
    rasterLayer = QgsRasterLayer(outputRasterFileName, baseName)
    #if not rasterLayer.isValid():
    #    print "Layer failed to load"
    QgsMapLayerRegistry.instance().addMapLayer(rasterLayer)
#inputRasterFileName='F:/texture/mc/all_bands.tif'
#pca(inputRasterFileName, 'PC_OUT.tif', 3)
    
    
    
    
    

        