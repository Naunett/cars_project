#load weights from caffe_net
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import skimage
import skimage.transform
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

net = {}
net['input'] = InputLayer((None, 3, 224, 224))
net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
output_layer1 = net['fc7']
net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
output_layer2 = net['fc8']

#to download vgg cnn:
#!wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl
model = pickle.load(open('vgg_cnn_s.pkl'))
CLASSES = model['synset words']
MEAN_IMAGE = model['mean image']

lasagne.layers.set_all_param_values(output_layer2, model['values']) #[:14])

#save parameters for the layers we need
layer_w_b = {}
i = 0
for layer in net:
    print i, layer, net[layer].output_shape
    i+=1
    if layer[:4] != 'drop' and layer != 'input' and layer[:4] != 'pool' and layer[:4] != 'norm':
        layer_w_b[layer] = [net[layer].W.get_value(),net[layer].b.get_value()]

        
#imports for nolearn net
from nolearn.lasagne import NeuralNet

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer

from lasagne.updates import nesterov_momentum


#create layers for nolearn, load weights
layers0 = [
    # layer dealing with the input data
    (InputLayer, {'name':'input', 'shape': (None, 3, 224, 224)}),
    # first stage of our convolutional layers
    (ConvLayer, {'name':'conv1', 
                 'num_filters': 96, 
                 'filter_size': (7,7),
                 'stride':2,
                 'flip_filters':False,
                 'W':layer_w_b['conv1'][0],
                 'b':layer_w_b['conv1'][1]}),
    (NormLayer, {'name': 'norm1', 'alpha':0.0001 }),
    (PoolLayer, {'name':'pool1', 'pool_size': 3, 'stride':3, 'ignore_border':False}),
    (ConvLayer, {'name':'conv2', 
                 'num_filters': 256, 
                 'filter_size': (5,5),
                 'flip_filters':False,
                 'W':layer_w_b['conv2'][0],
                 'b':layer_w_b['conv2'][1]}),
    (PoolLayer, {'name':'pool2', 'pool_size': 2, 'stride':2, 'ignore_border':False}),
    (ConvLayer, {'name':'conv3', 
                 'num_filters': 512, 
                 'filter_size': (3,3), 
                 'pad':1, 
                 'flip_filters':False,
                 'W':layer_w_b['conv3'][0],
                 'b':layer_w_b['conv3'][1]}),
    (ConvLayer, {'name':'conv4', 
                 'num_filters': 512, 
                 'filter_size': (3,3), 
                 'pad':1, 
                 'flip_filters':False,
                 'W':layer_w_b['conv4'][0],
                 'b':layer_w_b['conv4'][1]}),
    (ConvLayer, {'name':'conv5', 
                 'num_filters': 512, 
                 'filter_size': (3,3), 
                 'pad':1, 
                 'flip_filters':False,
                 'W':layer_w_b['conv5'][0],
                 'b':layer_w_b['conv5'][1]}),
    (PoolLayer, {'name':'pool5', 'pool_size': 3, 'stride':3, 'ignore_border':False}), 
    (DenseLayer, {'name':'fc6',
                  'num_units': 4096, 
                  'W': layer_w_b['fc6'][0],
                  'b': layer_w_b['fc6'][1] }),
    (DropoutLayer, {'name': 'drop6', 'p': 0.5 }),
    (DenseLayer, {'name':'fc7',
                  'num_units': 4096, 
                  'W': layer_w_b['fc7'][0],
                  'b': layer_w_b['fc7'][1] })
]

net0 = NeuralNet(
    layers=layers0,
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
  #  regression=True,  # flag to indicate we're dealing with regression problem
  #  max_epochs=400,  # we want to train this many epochs
    verbose=1,
)

#initialize nolearn net
net0.initialize()

#save weights and biases to the file for future use
net0.save_params_to('nolearn_with_w_b.pkl')