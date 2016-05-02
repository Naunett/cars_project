import numpy as np
import matplotlib.pyplot as plt
import io
import skimage
import skimage.transform
import urllib

import lasagne
from lasagne.utils import floatX

import cPickle as pickle


#image processing implemented outside of the class for parallel image preprocessing
def parallel_img_preprocess(url):
    '''
    Reshapes picture and prepare it for passing to the neural net input
    '''
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]
    
    # returns only preprocessed image
    return floatX(im[np.newaxis])


class ImgFeaturizer(object):
    
    def __init__(self, file_with_nn = 'cars_net.pkl'):
        '''
        Class takes one image, preprocess it and runs through neural network to featurize
        '''
        self.net0 = self.unpicke_net(file_with_nn)
    
    def unpicke_net(self, fname):
        '''
        downloads neural network with weights and biases
        '''
        net0 = pickle.load(open(fname))
        return net0
    
    def featurize_one_car(self, user_url):
        '''
        preprocess and featurize one car by passing it through the neural network
        '''
        im = parallel_img_preprocess(user_url)
        user_feat = self.net0.predict_proba(im)
        return user_feat
    
    def featurize_one_car_for_db_parallel(self, img_preprocessed):
        '''
        featurize one car by passing it through the neural network. takes preprocessed image as an input
        '''
        user_feat = self.net0.predict_proba(img_preprocessed)
        return user_feat

if __name__ == '__main__':
     featurizer = ImgFeaturizer()