from code.img_featurizer import *
import cPickle as pickle
import pandas

class DBFeaturizer(object):
    
    def __init__(self):
        '''
        Class takes timages, runs them through neural network and featurize
        '''    
        #create featurizer 
        self.featurizer = ImgFeaturizer()
        
    def _unpicke_db(self, db_fname = 'scraped_1500.pkl'):
        '''
        downloads scraped db with columns 'link' and 'image'
        '''
        with open(db_fname) as f:
            cars_df = pickle.load(f)
        return cars_df
    
    # the next two functions don't have return, they explicitly modify DF that we are working with
    
    def create_small_db(self, df):
        '''
        creates smaller df that contains only links to the cars web pages and images
        '''
        self.cars_img_df = df[['link', 'img']].copy()
    
    def _delete_rows_with_nan(self):
        self.cars_img_df = self.cars_img_df[self.cars_img_df['img'].isnull() != True]
        self.cars_img_df = self.cars_img_df[self.cars_img_df['link'].isnull() != True]
        
    #this function has return because leaving one img is optional and depends on whether we want to use 1 img
    #for one car or multiple images
    
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
    
    def featurize_db_main_car(self, db_fname = 'scraped_1500.pkl', nrows = None):
        '''
        featurizes all the car images in the dataset
        nrows is optional parameter that allows to use small part of the dataframe (to test).
        '''
        cars_df = self._unpicke_db(db_fname).ix[:nrows]
        self.create_small_db(cars_df)
        self._delete_rows_with_nan()
        #delete the next row if want to choose based on all the images of the car, not only main one
        self.cars_img_df = self._leave_only_one_img(self.cars_img_df)
        self._add_features_column()
        
        for k, img_url in enumerate(self.cars_img_df.img):
            try:
                prob = self.featurizer.featurize_one_car(img_url)
                self.cars_img_df.set_value(k, 'featurized', prob)
            except IOError:
                print('bad url: ' + url)
        return self.cars_img_df
    
    def pickle_featurized(self, df, fname):
        with open(fname, 'w') as f:
            pickle.dump(df, f)
    
if __name__ == '__main__':
    db_featurizer = DBFeaturizer()
    featurized_1500 = db_featurizer.featurize_db_main_car(db_fname = 'scraped_1500.pkl')
    pickle_featurized(featurized_1500, 'featurized_1500_t.pkl')