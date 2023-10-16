import pandas as pd
import string
import spacy
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from spacy.lang.it.stop_words import STOP_WORDS
from spacy.lang.it import Italian
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn import neural_network
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import json
import collections

#spacy.cli.download("it_core_news_sm")
#init spaCy
punctuations = string.punctuation
nlp = spacy.load("it_core_news_sm")
stop_words = spacy.lang.it.stop_words.STOP_WORDS
parser = Italian()

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

# Tokenizer function
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.text for word in mytokens ]
    # remove stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    # return preprocessed list of tokens
    return mytokens

ft = open("cpv_train.json","r")
X = []
y = []
for line in ft:
    jo = json.loads(line)
    X.append(jo["source"])
    y.append(jo["target"][0:10])
ft.close()

#BoW with word count
#vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
#BoW with TF-IDF
vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

print("DIVISIONI")
# divisioni
y = [x[0:2] for x in y]

classifier = svm.LinearSVC(C=0.5)
pipe1 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
print("SVC model fit...")
pipe1.fit(X, y)

ft = open("cpv_test.json","r")
X_test = []
y_test = []
for line in ft:
    jo = json.loads(line)
    X_test.append(jo["source"])
    y_test.append(jo["target"][0:10])
ft.close()
y_test = [x[0:2] for x in y_test]

predicted = pipe1.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("LinearSVC Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

classifier = MultinomialNB(alpha = 0.3, fit_prior = False)
pipe2 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
print("NB model fit...")
pipe2.fit(X, y)
predicted = pipe2.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("MultinomialNB Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")