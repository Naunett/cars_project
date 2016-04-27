from flask import Flask
from flask import (request,
                   redirect,
                   url_for,
                   session,
                   render_template)
import cPickle as pickle

#my imports
from src.predict import user_similarities
from src.img_featurizer import *

app = Flask(__name__)
app.secret_key = 'adjfkdsfjvjkdfsjgksdfjgkoppo'


@app.route('/')
def submit():
    return render_template('img_submit.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        user_link = str(request.form['user_car'])
        user_car_feat = featurizer.featurize_one_car(user_link)
        try:
            predicted = user_similarities(user_car_feat, df_cars_feat, 10)
        except:
            print 'sorry, couldnt predict'
            predicted = [(user_link,user_link)]
        session['prediction'] = predicted
        return redirect(url_for('predict'))
    else:
        if 'prediction' in session:
            return render_template('cars_predicted.html', predicted = session['prediction'])
        return redirect(url_for('submit'))

if __name__ == '__main__':
    #loads featurized cars and neural net - when app starts
    with open("data/featurized_1498.pkl") as f_un:
        df_cars_feat = pickle.load(f_un)
    featurizer = ImgFeaturizer('/home/ubuntu/similar_car/src/cars_net.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)