from flask import Flask
from flask import (request,
                   redirect,
                   url_for,
                   session,
                   render_template)
import cPickle as pickle

#my imports
from src.predict import *
from src.img_featurizer import *

app = Flask(__name__)
app.secret_key = 'adjfkdsfjvjkdfsjgksdfjgkoppo'


@app.route('/')
@app.route('/index')
def submit():
    return render_template('submit.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    one_to_many = False
    if request.method == 'POST':
        user_link = str(request.form['user_car'])
        user_car_feat = featurizer.featurize_one_car(user_link)
        try:
            if one_to_many:
                predicted = user_similarities_one_to_many(user_car_feat, df_cars_feat_all_img, df_cars_scraped, 21)
            else:
                predicted = user_similarities(user_car_feat, df_cars_feat_one_img, df_cars_scraped, 21)
        except:
            print 'sorry, couldnt predict'
            predicted = [(user_link, user_link, user_link, user_link, user_link)]
        session['prediction'] = predicted
        return redirect(url_for('predict'))
    else:
        if 'prediction' in session:
            return render_template('predict.html', predicted = session['prediction'])
        return redirect(url_for('submit'))

if __name__ == '__main__':
    #loads two sets of featurized cars: one with all the featurized images for every car from craigslist, another only main img.
    #and loads neural net - when app starts
    with open("data/featurized_all_imgs.pkl") as f_un:
        df_cars_feat_all_img = pickle.load(f_un)
    with open("data/featurized_main_img.pkl") as f_un:
        df_cars_feat_one_img = pickle.load(f_un)
    with open("data/scraped.pkl") as f_un:
        df_cars_scraped = pickle.load(f_un)
    featurizer = ImgFeaturizer('src/cars_net.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)