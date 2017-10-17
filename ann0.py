from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FullConnection, FeedForwardNetwork, TanhLayer, LinearLayer, BiasUnit
import matplotlib.pyplot as plt
from numpy import *

n = FeedForwardNetwork()
n.addInputModule(LinearLayer(1, name = 'in'))
n.addInputModule(BiasUnit(name = 'bias'))
n.addModule(TanhLayer(3,name = 'gotan'))
n.addOutputModule(LinearLayer(1, name = 'out'))
n.addConnection(FullConnection(n['bias'], n['gotan']))
n.addConnection(FullConnection(n['in'], n['gotan']))
n.addConnection(FullConnection(n['gotan'], n['out']))
n.sortModules()

# initialize the backprop trainer and train
t = BackpropTrainer(n, learningrate = 0.1, momentum = 0.0, verbose = True)
#DATASET

DS = SupervisedDataSet( 1, 1 )
X = random.rand(100,1)*100
Y = X**3+random.rand(100,1)*5
maxy = float(max(Y))
maxx = 100.0

for r in range(X.shape[0]):
    DS.appendLinked((X[r]/maxx),(Y[r]/maxy))

t.trainOnDataset(DS, 200)

plt.plot(X,Y,'.b')
X=[[i] for i in arange(0,100,0.1)]
Y=list(map(lambda x: n.activate(array(x)/maxx)*maxy,X))
plt.plot(X,Y,'-g')
#