from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS


# Create training corpus. Must be a sequence of sentences (e.g. an iterable or a generator).
sentences = Text8Corpus(datapath('testcorpus.txt'))
# Each sentence must be a list of string tokens:
first_sentence = next(iter(sentences))
print(first_sentence[:10])

# Train a toy phrase model on our training corpus.
phrase_model = Phrases(sentences, min_count=1, threshold=1,
                       connector_words=ENGLISH_CONNECTOR_WORDS)

# Apply the trained phrases model to a new, unseen sentence.
new_sentence = ['trees', 'graph', 'minors']
phrase_model[new_sentence]

for sent in phrase_model[sentences]:
    pass

phrase_model.add_vocab([["hello", "world"], ["meow"]])

# Export the trained model = use less RAM, faster processing. Model updates no longer possible.
frozen_model = phrase_model.freeze()
# Apply the frozen model; same results as before:
frozen_model[new_sentence]

# Save / load models.
frozen_model.save("/tmp/my_phrase_model.pkl")
model_reloaded = Phrases.load("/tmp/my_phrase_model.pkl")
# apply the reloaded model to a sentence
model_reloaded[['trees', 'graph', 'minors']]
