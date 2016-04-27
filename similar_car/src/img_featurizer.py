import numpy as np
import matplotlib.pyplot as plt
import io
import skimage
import skimage.transform
import urllib

import lasagne
from lasagne.utils import floatX

import cPickle as pickle


class ImgFeaturizer(object):
    
    def __init__(self, file_with_nn = '/home/ubuntu/similar_car/src/cars_net.pkl'):
        '''
        Class takes timages, runs them through neural network and featurize
        '''
        self.net0 = self.unpicke_net(file_with_nn)
        self.MEAN_IMAGE = self.unpickle_mean_img()
        
    def img_preprocess(self, url):
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

        rawim = np.copy(im).astype('uint8')

        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Convert to BGR
        im = im[::-1, :, :]

        im = im - self.MEAN_IMAGE
        return rawim, floatX(im[np.newaxis])
    
    def unpickle_mean_img(self, fname ='/home/ubuntu/similar_car/src/mean_image.pkl'):
        '''
        helper for img_preprocess function
        '''
        MEAN_IMAGE = pickle.load(open(fname))
        return MEAN_IMAGE
    
    def unpicke_net(self, fname):
        '''
        downloads neural network with weights and biases
        '''
        net0 = pickle.load(open(fname))
        return net0
    
    def featurize_one_car(self, user_url):
        '''
        featurize one car by passing it through the neural network
        '''
        _, im = self.img_preprocess(user_url)
        user_feat = self.net0.predict_proba(im)
        return user_feat
    
#if __name__ == '__main__':
#     featurizer = ImgFeaturizer('/home/ubuntu/similar_car/code/cars_net.pkl')