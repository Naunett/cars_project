from img_featurizer_parallel import *
import cPickle as pickle
import pandas


#block of the functions outside of the class for parallel computing
import multiprocessing
from functools import partial
from img_featurizer_parallel import parallel_img_preprocess
import itertools


#run multiprocessing outside of the class
multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=6)

def prep_n_images(part_of_img_list):
    output = pool.map(parallel_img_preprocess, part_of_img_list)
    return output

def prep_n_images_all_img(part_of_img_list):
    output = map(partial(map, parallel_img_preprocess), part_of_img_list)
    return output



class DBFeaturizer(object):
    
    def __init__(self):
        '''
        Class takes timages, runs them through neural network and featurize
        '''    
        #create an instance of the featurizer for one image
        self.featurizer = ImgFeaturizer()
        
    def _unpicke_db(self, db_fname = '../data/scraped.pkl'):
        '''
        downloads scraped db with columns 'link' and 'image'
        '''
        with open(db_fname) as f:
            cars_df = pickle.load(f)
        return cars_df
    
    def create_small_db(self, df):
        '''
        creates smaller df that contains only links to the cars web pages and images
        '''
        df = df[['link', 'img']].copy()
        return df
    
    #functions that delete or add rows or columns don't have return, they explicitly modify DF that we are working with
    def _delete_rows_with_nan(self):
        '''
        deletes rows that has Nan values in important for future processing columns: link and img
        '''
        self.cars_img_df = self.cars_img_df[self.cars_img_df['img'].isnull() != True]
        self.cars_img_df = self.cars_img_df[self.cars_img_df['link'].isnull() != True]
    
    def _leave_only_one_img(self, df):
        '''
        this is to look for cars based on only their main image on website. dropping other links to the images
        '''
        df['img'] = df['img'].apply(lambda x: x[0])
        return df
    
    def _add_features_column(self):
        '''
        add column to keep featurized image data
        '''
        self.cars_img_df['featurized'] = np.nan
        self.cars_img_df['featurized'] = self.cars_img_df['featurized'].astype(object)

    def _add_preprocessed_column(self):
        '''
        add column to keep featurized image data
        '''
        self.cars_img_df['preprocessed'] = np.nan
        self.cars_img_df['preprocessed'] = self.cars_img_df['preprocessed'].astype(object)
 
       
    def featurize_db_main_car(self, db_fname = '../data/scraped_1500_042916.pkl', nrows = None):
        '''
        featurizes all the car images in the dataset
        nrows is optional parameter that allows to use small part of the dataframe (to test).
        '''
        cars_df = self._unpicke_db(db_fname).ix[:nrows]
        self.cars_img_df = self.create_small_db(cars_df)
        self._delete_rows_with_nan()
        self.cars_img_df = self._leave_only_one_img(self.cars_img_df)
        self._add_features_column()
        
        for k, img_url in enumerate(self.cars_img_df.img):
            try:
                prob = self.featurizer.featurize_one_car(img_url)
                print prob
                self.cars_img_df.set_value(k, 'featurized', prob)
            except IOError:
                print('bad url: ' + url)
        return self.cars_img_df
    
    
    
    #using parallel computations
    
    #helper functions for parallel
    def create_img_list(self, df):
        img_links = df.img.values.tolist()
        return img_links

    def split_list(self, img_links, n=500):
        img_links_splitted = [img_links[i:i+n] for i in range(0, len(img_links), n)]
        return img_links_splitted   
    
    
    #parallel featurization. here preprocess and featurization are implemented in one function to avoid saving preprocessed
    #images, since they are much more space consuming than featurized.
    
    def preprocess_and_featurize_all_images_parallel(self, db_fname, nrows = None):
        '''
        function preprocess and featurize all the images for every car. takes as an input scraped data. 
        '''
        cars_df = self._unpicke_db(db_fname).ix[:nrows]
        self.cars_img_df = self.create_small_db(cars_df)
        self._delete_rows_with_nan()
        self._add_features_column()
        
        img_links = self.create_img_list(cars_df)
        
        #!! change the split number to bigger one if necessary
        img_links_splitted = self.split_list(img_links, n = 10)
        all_img_featurized = []
        for i in range(len(img_links_splitted)):
            imgs_preprocessed = prep_n_images_all_img(img_links_splitted[i])
            chunk_featurized_imgs = []
            for k, imgs in enumerate(imgs_preprocessed):
                featurized_imgs = []
                for img in imgs:
                    try:
                        prob = self.featurizer.featurize_one_car_for_db_parallel(img)
                        featurized_imgs.append(prob)
                    except IOError:
                        print('bad url: ' + url)
                        prob = 0
                        featurized_imgs.append(prob)  
                chunk_featurized_imgs.append(featurized_imgs)
            all_img_featurized.append(chunk_featurized_imgs) 
        self.total_img_merged = list(itertools.chain.from_iterable(all_img_featurized))
        return self.total_img_merged
    
    def add_featurized_imgs_to_df(self, df, featurized_imgs):
        '''
        function adds featurized images to the corresponding column in dataframe
        '''
        for k, _ in enumerate(df.link):
            df.set_value(k, 'featurized', featurized_imgs[k])
        return df
    
    def featurize_db_all_imgs(self, db_fname = '../data/scraped.pkl'):
        '''
        runs image preprocess and featurization and saves the result into corresponding column
        '''
        self.preprocess_and_featurize_all_images_parallel(db_fname)
        self.cars_img_df = self.add_featurized_imgs_to_df(self.cars_img_df, self.total_img_merged)
        return self.cars_img_df
    
    def pickle_featurized(self, df, fname):
        with open(fname, 'w') as f:
            pickle.dump(df, f)
    
if __name__ == '__main__':
    db_featurizer = DBFeaturizer()
    #set the flag to False to run the process for one image of the car or True for all the images
    featurize_all_images = True
    if featurize_all_images:
        featurized_1500 = db_featurizer.featurize_db_all_imgs(db_fname = '../data/scraped.pkl')
    else:
        featurized_1500 = db_featurizer.featurize_db_main_car(db_fname = '../data/scraped.pkl')
    db_featurizer.pickle_featurized(featurized_1500, '../data/featurized.pkl')