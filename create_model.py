import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('data/imdb_dataset.csv')
# Define our X
X = df['clean_data']
# Define our y
y = df['sentiment']

# Initalize our vectorizer
vectorizer = TfidfVectorizer()

# Fit our vectorizer
vectorizer.fit(X)

# Transform your X data using your fitted vectorizer. 
X = vectorizer.transform(X)

model = MultinomialNB(alpha=.05)
# Fit our model with our training data.
model.fit(X, y)

# Save our vectorizer and model.
pickle.dump(model, open('models/sentiment_analysis_movie_review.pkl', 'wb'))

