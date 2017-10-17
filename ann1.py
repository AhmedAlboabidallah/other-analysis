# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:31:58 2017

@author: ahalboabidallah
"""
max1=20.0
min1=0.0
#----------
# build the dataset
#----------

def normalize(column1,max1,min1):
    try:
        column1=np.array(column1)
    except:
        pass
    for i in range(len(column1)):
        column1[i]=(float(column1[i])-float(min1))/(float(max1)-float(min1))
    return column1
def inv_normalize(column1,max1,min1):
    for i in range(len(column1)):
        column1[i]=float(column1[i])*(float(max1)-float(min1))+float(min1)
    return column1
from pybrain.datasets import SupervisedDataSet
import numpy, math
from pybrain.structure import TanhLayer

xvalues = numpy.transpose(numpy.array([[1.,10.,3.,4,5,6,15,8,9,10,11,0,5,5,5,5,5,6,6,3.,4.,5,6,7],[11,12,13,14,20,16,17,18,19,20,21,0,3,4,5,6,7,6,1,12,13,14,10,16]]))
xvalues.astype(float)

yvalues = list(map(lambda x: x[0]**2+x[1],xvalues))

#print('yy',inv_normalize(normalize(yvalues,max1,min1),max1,min1),'y',yvalues)
ds = SupervisedDataSet(2, 1)
for i in range(len(xvalues[0,:])):
    xvalues[:,i]=normalize(xvalues[:,i],max1,min1)
yvalues=normalize(yvalues,max1,min1)
for i in range(len(yvalues)):
    ds.addSample((xvalues[i,:]), (yvalues[i]))

#----------
# build the network
#----------
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork

net = buildNetwork(2,
                   2,5, # number of hidden units
                   1,
                   bias = True,
                   hiddenclass=TanhLayer,#hiddenclass = SigmoidLayer,#
                   outclass = LinearLayer
                   )
#----------
# train
#----------
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, verbose = True)
trainer.trainUntilConvergence(maxEpochs = 1000)

#----------
# evaluate
#----------
Xvalidation=numpy.transpose(numpy.array([[1.,3.,4.,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[11,12,13,14,10,16,17,18,19,20,21,0,3,4,5,6,7]]))
Yvalidation=list(map(lambda x: x[0]**2+x[1],Xvalidation))
for i in range(len(Xvalidation[0,:])):
    Xvalidation[:,i]=normalize(Xvalidation[:,i],max1,min1)
#print('norm_Xval=',Xvalidation)
k=inv_normalize(Xvalidation[:,0],max1,min1)
#print('k=',k)
import pylab
# neural net approximation
pylab.plot(k,
           inv_normalize([ net.activate(x) for x in Xvalidation],max1,min1), linewidth = 2,
           color = 'blue', label = 'NN output')

# target function
pylab.plot(k,
           Yvalidation, linewidth = 2, color = 'red', label = 'target')

# neural net approximation
#pylab.plot(inv_normalize(xvalues[:,0],max1,min1),
#           inv_normalize([ net.activate(x) for x in xvalues],max1,min1), linewidth = 2,
#           color = 'green', label = 'NN output')

# target function
#pylab.plot(inv_normalize(xvalues[:,0],max1,min1),
#           inv_normalize(yvalues,max1,min1), linewidth = 2, color = 'gray', label = 'target')

pylab.grid()
pylab.legend()
pylab.show()
#'''

