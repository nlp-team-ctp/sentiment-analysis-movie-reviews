import flask
import os
import pickle
import pandas as pd
import skimage
from nltk.tokenize import RegexpTokenizer


app = flask.Flask(__name__, template_folder='templates')

path_to_model = 'models/MultiNB_model_90_accu.pkl'
path_to_vectorizer = 'models/trigram_vectorizer.pkl'

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)


with open(path_to_model, 'rb') as f:
    model = pickle.load(f)


def preprocess(document):

    # convert to lower case
    document = document.lower()

    # tokenize document
    tk = RegexpTokenizer(r'[a-z\'\-\_]+')
    tokens = [token for token in tk.tokenize(document)]
    tokens = [token for token in tokens if token != 'br']

    return ' '.join(tokens)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        user_input_text = flask.request.form['user_input_text']

        X = vectorizer.transform([preprocess(user_input_text)])

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        result = True if y_pred == 1 else False
        confidance = max(y_proba.item(0), y_proba.item(1))
        confidance = round(confidance, 3) * 100
        confidance_percent = "%.1f" % confidance

        return flask.render_template('main.html',
                                     input_text=user_input_text,
                                     result=result,
                                     confidance=confidance_percent)


@app.route('/moviescore/')
def moviescore():
    return flask.render_template('moviescore.html')


if __name__ == '__main__':
    app.run(debug=True)
