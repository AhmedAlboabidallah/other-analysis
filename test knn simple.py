# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:15:13 2017

@author: ahalboabidallah
"""

# Trying to define the simplest possible neural net where the output layer of the neural net is a single
# neuron with a "continuous" (a.k.a floating point) output.  I want the neural net to output a continuous
# value based off one or more continuous inputs.  My real problem is more complex, but this is the simplest
# representation of it for explaining my issue.  Even though I've oversimplified this to look like a simple
# linear regression problem (y=m*x), I want to apply this to more complex neural nets.  But if I can't get
# it working with this simple problem, then I won't get it working for anything more complex.
import tensorflow as tf
import random
import numpy as np

INPUT_DIMENSION  = 3
OUTPUT_DIMENSION = 1
hm_epochs        = 1000#hm_epochs
BATCH_SIZE       = 10000
VERF_SIZE        = 2
learning_rate    = 0.01
# Generate two arrays, the first array being the inputs that need trained on, and the second array containing outputs.
def generate_test_point():
    x1,x2,x3 = random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)
    # To keep it simple, output is just -x.  
    out = x1+(x2*2)+(x3*3)
    return ( np.array([x1,x2,x3]), np.array([ out ]) )

# Generate a bunch of data points and then package them up in the array format needed by
# tensorflow
def generate_batch_data( num ):
     xs = []
     ys = []
     for i in range(num):
       x, y = generate_test_point()
       xs.append( x )
       ys.append( y )
     return (np.array(xs), np.array(ys) )

# Define a single-layer neural net.  Originally based off the tensorflow mnist for beginners tutorial
# Create a placeholder for our input variable
x = tf.placeholder(tf.float32, [None, INPUT_DIMENSION])
# Create variables for our neural net weights and bias
W = tf.Variable(tf.random_normal([INPUT_DIMENSION, OUTPUT_DIMENSION]))
b = tf.Variable(tf.random_normal([OUTPUT_DIMENSION]))
y = tf.placeholder(tf.float32, [None, OUTPUT_DIMENSION])
# Define the neural net.  Note that since I'm not trying to classify digits as in the tensorflow mnist
# tutorial, I have removed the softmax op.  My expectation is that 'net' will return a floating point
# value.
'''net = tf.matmul(x, W) + b'''#
n_nodes_hl1 = 3
n_nodes_hl2 = 3
n_nodes_hl3=3
n_classes   = 1
#batch_size = 32#use all d
#ata#BATCH_SIZE
#total_batches = int(1600000/batch_size)#use all data
#x = tf.placeholder('float')
#y = tf.placeholder('float')
hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([INPUT_DIMENSION, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}
output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])#Multiplies data,hidden_1_layer['weight']), #then add bias hidden_1_layer['bias']
    l1 = tf.nn.relu(l1)#convert l1 inton a Tensor Returns: A Tensor. Has the same type as features.
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])# summing
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])# summing
    l3 = tf.nn.relu(l3)
    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']
    return output

# Create a placeholder for the expected result during training
y = tf.placeholder(tf.float32, [None, OUTPUT_DIMENSION])
# training 
net=neural_network_model(x)
loss = tf.reduce_mean(tf.abs(y - net))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
sess = tf.Session()

init = tf.initialize_all_variables()
sess.run(init)

# Perform our training runs
for i in range( hm_epochs ):
  print ("trainin run: ", i,)
  batch_inputs, batch_outputs = generate_batch_data( BATCH_SIZE )
  # I've found that my weights and bias values are always zero after training, and I'm not sure why.
  sess.run( train_step, feed_dict={x: batch_inputs, y: batch_outputs})
  # Test our accuracy as we train...  I am defining my accuracy as the error between what I 
  # expected and the actual output of the neural net.
  #accuracy = tf.reduce_mean(tf.subtract( expected, net))  
  accuracy = tf.subtract( y, net) # using just subtract since I made my verification size 1 for debug
  # Uncomment this to debug
  #import pdb; pdb.set_trace()
  batch_inputs, batch_outputs = generate_batch_data( VERF_SIZE )
  result = sess.run(accuracy, feed_dict={x: batch_inputs, y: batch_outputs})
  print ("    progress: ", accuracy)
  print ("      inputs: ", batch_inputs)
  print ("      outputs:", batch_outputs)
  print ("      actual: ", result)