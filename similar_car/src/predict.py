from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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



#if __name__ == '__main__':
#    user_similarities()
