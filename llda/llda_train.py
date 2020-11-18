from nltk.corpus import reuters
from llda import LldaModel
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


df = pd.read_csv(
    "C:\\Users\\Terolli\\Desktop\\sentiment-analysis-movie-reviews\\llda\\IMDB Dataset.csv")
df = df.drop_duplicates()


def preprocess(document):
    document = document.lower()
    tk = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = [token for token in tk.tokenize(document)]

    # stoplist = set(stopwords.words('english'))
    # tokens = [token for token in tokens if token not in stoplist]

    # porter = PorterStemmer()
    # tokens = [porter.stem(token) for token in tokens]

    tokens = [token for token in tokens if token != 'br']
    return ' '.join(tokens)


df['clean_review'] = df['review'].apply(preprocess)


X = df['clean_review'].tolist()
y = df['sentiment'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# labeled_docs have the following format:
# [
#   ('reuters.words', [reuters.categories]),
#   ('reuters.words', [reuters.categories]),
#   ...
# ]
labeled_docs_train = [(x, y.split())
                      for x, y in zip(X_train, y_train)]


print("initialize LLDA model")
# initialize LLDA model
llda_model = LldaModel(
    labeled_documents=labeled_docs_train, alpha_vector=0.01)
print(llda_model)

print("TRAINING...")
llda_model.training(iteration=5, log=True)

'''
# train llda_model
# early stop
while True:
    print("iteration %s sampling..." % (llda_model.iteration + 1))

    # training llda_model
    llda_model.training(1)

    print("after iteration: %s, perplexity: %s" %
          (llda_model.iteration, llda_model.perplexity()))
    print("delta beta: %s" % llda_model.delta_beta)
    if llda_model.is_convergent(method="beta", delta=0.01):
        break
'''

pickle.dump(llda_model, open('llda_model.pkl', 'wb'))
