from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import cPickle as pickle

layers0 = [
    # layer dealing with the input data
    (InputLayer, {'name':'input', 'shape': (None, 3, 224, 224)}),
    # first stage of our convolutional layers
    (ConvLayer, {'name':'conv1', 
                 'num_filters': 96, 
                 'filter_size': (7,7),
                 'stride':2,
                 'flip_filters':False}),
    (NormLayer, {'name': 'norm1', 'alpha':0.0001 }),
    (PoolLayer, {'name':'pool1', 'pool_size': 3, 'stride':3, 'ignore_border':False}),
    (ConvLayer, {'name':'conv2', 
                 'num_filters': 256, 
                 'filter_size': (5,5),
                 'flip_filters':False}),
    (PoolLayer, {'name':'pool2', 'pool_size': 2, 'stride':2, 'ignore_border':False}),
    (ConvLayer, {'name':'conv3', 
                 'num_filters': 512, 
                 'filter_size': (3,3), 
                 'pad':1, 
                 'flip_filters':False}),
    (ConvLayer, {'name':'conv4', 
                 'num_filters': 512, 
                 'filter_size': (3,3), 
                 'pad':1, 
                 'flip_filters':False}),
    (ConvLayer, {'name':'conv5', 
                 'num_filters': 512, 
                 'filter_size': (3,3), 
                 'pad':1, 
                 'flip_filters':False}),
    (PoolLayer, {'name':'pool5', 'pool_size': 3, 'stride':3, 'ignore_border':False}), 
    (DenseLayer, {'name':'fc6',
                  'num_units': 4096 }),
    (DropoutLayer, {'name': 'drop6', 'p': 0.5 }),
    (DenseLayer, {'name':'fc7',
                  'num_units': 4096 })
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

net0.load_params_from('nolearn_with_w_b.pkl')

with open('cars_net.pkl','wb') as f:
    pickle.dump(net0, f)