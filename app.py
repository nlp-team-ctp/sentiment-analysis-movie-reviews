import flask
import os
import pickle
import pandas as pd
import skimage


app = flask.Flask(__name__, template_folder='templates')

path_to_sentiment_AMR = 'models/MultiNB_model_89_accu.pkl'
path_to_vectorizer = 'models/trigram_vectorizer.pkl'

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)


with open(path_to_sentiment_AMR, 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_text']

        # Turn the text into numbers using our vectorizer
        X = vectorizer.transform([user_input_text])

        # Make a prediction
        y_pred = model.predict(X)

        result = 'Positive' if y_pred == 1 else 'Negative'

        return flask.render_template('main.html',
                                     input_text=user_input_text,
                                     result=result)


@app.route('/bootstrap/')
def bootstrap():
    return flask.render_template('bootstrap.html')


if __name__ == '__main__':
    app.run(debug=True)
