import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs4
import requests
import urllib
import re
import os, sys
import cPickle as pickle

def find_prices(results):
    '''
    helper function for search results. finds end adds prices, if available
    '''
    prices = []
    for rw in results:
        price = rw.find('span', {'class': 'price'})
        if price is not None:
            price = float(price.text.strip('$'))
        else:
            price = np.nan
        prices.append(price)
    return prices

def scrape_search_results(areas):
    '''
    scrapes search page, collects information about the cars available for sale.
    '''
    results = []  
    search_indices = np.arange(0, 300, 100)
    for area in areas:
        print area
        for i in search_indices:
            url = 'http://sfbay.craigslist.org/search/{0}/cta'.format(area)
            resp = requests.get(url, params={'hasPic': 1, 's': i})
            txt = bs4(resp.text, 'html.parser')
            cars = txt.findAll(attrs={'class': "row"})
            tags=txt.findAll('img')
            img_tags = "\n".join(set(tag['src'] for tag in tags))
            title = [rw.find('a', attrs={'class': 'hdrlnk'}).text
                          for rw in cars]
            links_raw = [rw.find('a', attrs={'class': 'hdrlnk'})['href']
                     for rw in cars]
            links = ['http://sfbay.craigslist.org'+car_link for car_link in links_raw]
            # find the time and the price
            time = [pd.to_datetime(rw.find('time')['datetime']) for rw in cars]
            price = find_prices(cars)

            # create a dataframe to store all the data
            data = np.array([time, price, title, links])
            col_names = ['time', 'price', 'title', 'link']
            df = pd.DataFrame(data.T, columns=col_names)

            # add the location variable to all entries
            df['loc'] = area
            results.append(df)

    # concatenate all the search results
    results = pd.concat(results, axis=0)
    return results

def add_columns_cars_df(df):
    '''
    add columns - placeholders - for every feature. 
    final scraped data will be pretty sparse, very few ads have all the information.
    '''
    df['img'] = np.nan
    df['img'] = df['img'].astype(object)
    df['model_year'] = np.nan
    df['make_and_model'] = np.nan
    df['VIN'] = np.nan
    df['odometer'] = np.nan
    df['condition'] = np.nan
    df['cylinders'] = np.nan
    df['drive'] = np.nan
    df['fuel'] = np.nan
    df['paint color'] = np.nan
    df['size'] = np.nan
    df['title status'] = np.nan
    df['transmission'] = np.nan
    df['type'] = np.nan
    return None
    
def scrape_car(df):
    '''
    scrapes all the necessary information about ecery car in df and stores it in corresponding cells
    '''
    for k,link in enumerate(df.link):
        resp_car = requests.get(link)
        txt_car = bs4(resp_car.text, 'html.parser')
        features = []
        for child in txt_car.findAll('p', {'class': 'attrgroup'}):
            for i in child.findAll('span'):
                features.append(i.text)
        if len(features) != 0:
            #get images from java script object - slider
            imgs_script = txt_car.findAll("script")[2]
            js_text = imgs_script.get_text()
            img_tags = re.findall('url":"([^"]*)\.jpg', js_text)
            img_tags = [url+'.jpg' for url in img_tags]

            features_splitted = [[item.strip() for item in str(features[i].encode("utf8")).split(':')] for i in xrange(len(features))]
            #add year, make and model to the results table
            m_model = features_splitted[0][0].split()
            if len(m_model[0]) == 4:
                year = m_model[0]
                make_and_model = ' '.join(m_model[1:])
            else: 
                make_and_model = ' '.join(m_model)

            df['model_year'][df.link == link] = int(year)
            df['make_and_model'][df.link == link] = make_and_model

            #add all the other available features
            for i in features_splitted[1:]:
                if (len(i)) > 1:
                    #df.set_value(k, i[0], i[1])
                    df[i[0]][df.link == link] = i[1]

            #add image tags
            df.set_value(k, 'img', img_tags)
            #doesn't return anything - changes df that is passed
    return None

def save_images_to_files(fpath, df):
    '''
    optional function: can download all the pictures of the car to the drive.
    it saves pictures to the separate directories for every car.
    '''
    for i, link in enumerate(df.link):
        path = fpath+'/'+str(link).split('/')[-1]
        links_imgs_one_car = df['img'].ix[i]
        if type(links_imgs_one_car) == list and len(links_imgs_one_car) != 0:
            if not os.path.exists(path):
                os.mkdir(path)
            for j in xrange(len(links_imgs_one_car)):
                url = links_imgs_one_car[j]
                im_name = links_imgs_one_car[j].split('/')[-1]
                img = urllib.urlretrieve(str(url), ('{}/{}.jpg'.format(path, str(im_name))))
    return None


#scraping itself
def scrape_all_info(areas = ['eby', 'nby', 'sfc', 'sby', 'scz']):
    '''scraping itself'''    
    #for SF areas = ['eby', 'nby', 'sfc', 'sby', 'scz']
    scraped_df = scrape_search_results(areas)
    add_columns_cars_df(scraped_df)
    scraped_df = scraped_df.reset_index()
    del scraped_df['index']
    print scraped_df.head(3)
    scrape_car(scraped_df)
    return scraped_df

if __name__ == '__main__':
    scraped_df = scrape_all_info(areas = ['eby', 'nby', 'sfc', 'sby', 'scz'])
    with open('data/scraped.pkl', 'w') as f:
        pickle.dump(scraped_df, f)