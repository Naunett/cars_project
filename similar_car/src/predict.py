from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
'''
def user_similarities(user_car_feat, df_cars_feat, n_predict):
    cosines = []
    for car in df_cars_feat.featurized:
        cosines.append(cosine_similarity(user_car_feat, car)[0][0])
    cosines = np.array(cosines)
    indexes = cosines.argsort()[::-1]
    car_links = df_cars_feat.link.values
    car_img_links = df_cars_feat.img.values
    cars_link_sort = car_links[indexes]
    car_img_links_sort = car_img_links[indexes]
    return zip(cars_link_sort[:n_predict],car_img_links_sort[:n_predict])
'''

def user_similarities(user_car_feat, df_cars_feat, df_cars_scraped, n_predict):
    cosines = []
    for car in df_cars_feat.featurized:
        cosines.append(cosine_similarity(user_car_feat, car)[0][0])
    cosines = np.array(cosines)
    indexes = cosines.argsort()[::-1]
    df_cars_top = df_cars_feat.ix[indexes][:n_predict]
    df_cars_top = pd.merge(df_cars_top,df_cars_scraped, on='link')
    car_links = df_cars_top.link.values
    car_img_links = df_cars_top.img_x.values
    car_model_year = df_cars_top.model_year.values.astype(int)
    car_make_and_model = df_cars_top.make_and_model.values
    car_price = df_cars_top.price.values
    car_price_clean = []
    for price in car_price:
        if np.isnan(price) == True:
            car_price_clean.append(0.0)
        else:
            car_price_clean.append(price)
    car_price = car_price_clean
    return zip(car_links, car_img_links, car_model_year, car_make_and_model, car_price)


def user_similarities_one_to_many(user_car_feat, df_cars_feat, df_cars_scraped, n_predict):
    cosines = []
    # parallelize if takes long
    for car_all in df_cars_feat.featurized:
        cosines_all_one_car = []
        for car in car_all:
            cosines_all_one_car.append(cosine_similarity(user_car_feat, car)[0][0])
        cosines.append(max(cosines_all_one_car))
    cosines = np.array(cosines)
    indexes = cosines.argsort()[::-1]
    df_cars_top = df_cars_feat.ix[indexes][:n_predict]
    df_cars_top = pd.merge(df_cars_top,df_cars_scraped, on='link')
    car_links = df_cars_top.link.values
    car_img_links = df_cars_top.img_x.apply(lambda x: x[0]).values
    car_model_year = df_cars_top.model_year.values.astype(int)
    car_make_and_model = df_cars_top.make_and_model.values
    car_price = df_cars_top.price.values
    car_price_clean = []
    for price in car_price:
        if np.isnan(price) == True:
            car_price_clean.append(0.0)
        else:
            car_price_clean.append(price)
    car_price = car_price_clean
    result = zip(car_links, car_img_links, car_model_year, car_make_and_model, car_price)
    return result


#if __name__ == '__main__':
#    user_similarities()
