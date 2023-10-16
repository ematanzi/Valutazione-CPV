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

import collections

df1 = pd.read_csv('cpv_clean_40.csv', sep=",", names=["cig","oggetto_bando","oggetto_lotto","oggetto_principale","cpv","desc_cpv"], encoding='utf-8')
cpv1 = df1["cpv"]
#BoW with word count
#vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
#BoW with TF-IDF
vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

#df1 = df1.iloc[:100000,:]       #todo per utilizzare 100000 bandi
df1 = df1[df1.duplicated('cpv',keep=False)] # rimuovo i cpv con un solo bando associato
X = df1['oggetto_bando']
y = df1['cpv'] # the labels

print("DIVISIONI")
# divisioni
y = [x[0:2] for x in df1['cpv']]

X_main, X_tesi, y_main, y_tesi = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  

X_exp, X_hype, y_exp, y_hype = train_test_split(X_tesi, y_tesi, test_size=0.2, random_state=42, stratify=y_tesi) 

X_train, X_test, y_train, y_test = train_test_split(X_exp, y_exp, test_size=0.2, random_state=42, stratify=y_exp) 

classifier = svm.LinearSVC(C=0.5)
pipe1 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
pipe1.fit(X_train, y_train)
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
pipe2.fit(X_train, y_train)
predicted = pipe2.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("MultinomialNB Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

#classifier = MLPClassifier(alpha=0.01, hidden_layer_sizes=(150, 150, 150), max_iter=20)
#pipe3 = Pipeline([("cleaner", predictors()),
#                  ('vectorizer', vector),
#                  ('classifier', classifier)])
#pipe3.fit(X_train, y_train)
#predicted = pipe3.predict(X_test)
#acc = metrics.accuracy_score(y_test, predicted)
#print("MLPClassifier Accuracy:", acc)
#precision = metrics.precision_score(y_test, predicted, average='macro')
#recall = metrics.recall_score(y_test, predicted, average='macro')
#fm = metrics.f1_score(y_test, predicted, average='macro')
#print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
#print("======================================================")

# gruppi
print("GRUPPI")
y = [x[0:3] for x in df1['cpv']]

X_main, X_tesi, y_main, y_tesi = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  

X_exp, X_hype, y_exp, y_hype = train_test_split(X_tesi, y_tesi, test_size=0.2, random_state=42, stratify=y_tesi) 

X_train, X_test, y_train, y_test = train_test_split(X_exp, y_exp, test_size=0.2, random_state=42, stratify=y_exp) 

classifier = svm.LinearSVC(C=0.5)
pipe1 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
pipe1.fit(X_train, y_train)
predicted = pipe1.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("LinearSVC Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

classifier = MultinomialNB(alpha = 0.1, fit_prior = False)
pipe2 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
pipe2.fit(X_train, y_train)
predicted = pipe2.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("MultinomialNB Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

#classifier = MLPClassifier(alpha=0.01, hidden_layer_sizes=(150, 150, 150), max_iter=20)
#pipe3 = Pipeline([("cleaner", predictors()),
#                  ('vectorizer', vector),
#                  ('classifier', classifier)])
#pipe3.fit(X_train, y_train)
#predicted = pipe3.predict(X_test)
#acc = metrics.accuracy_score(y_test, predicted)
#print("MLPClassifier Accuracy:", acc)
#precision = metrics.precision_score(y_test, predicted, average='macro')
#recall = metrics.recall_score(y_test, predicted, average='macro')
#fm = metrics.f1_score(y_test, predicted, average='macro')
#print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
#print("======================================================")

# classi
print("CLASSI")
y = [x[0:4] for x in df1['cpv']]

X_main, X_tesi, y_main, y_tesi = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  

X_exp, X_hype, y_exp, y_hype = train_test_split(X_tesi, y_tesi, test_size=0.2, random_state=42, stratify=y_tesi) 

X_train, X_test, y_train, y_test = train_test_split(X_exp, y_exp, test_size=0.2, random_state=42, stratify=y_exp) 

classifier = svm.LinearSVC(C=0.5)
pipe1 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
pipe1.fit(X_train, y_train)
predicted = pipe1.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("LinearSVC Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

classifier = MultinomialNB(alpha = 0.1, fit_prior = False)
pipe2 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
pipe2.fit(X_train, y_train)
predicted = pipe2.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("MultinomialNB Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

#classifier = MLPClassifier(alpha=0.01, hidden_layer_sizes=(150, 150, 150), max_iter=20)
#pipe3 = Pipeline([("cleaner", predictors()),
#                  ('vectorizer', vector),
#                  ('classifier', classifier)])
#pipe3.fit(X_train, y_train)
#predicted = pipe3.predict(X_test)
#acc = metrics.accuracy_score(y_test, predicted)
#print("MLPClassifier Accuracy:", acc)
#precision = metrics.precision_score(y_test, predicted, average='macro')
#recall = metrics.recall_score(y_test, predicted, average='macro')
#fm = metrics.f1_score(y_test, predicted, average='macro')
#print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
#print("======================================================")

# categorie
print("CATEGORIE")
y = [x[0:5] for x in df1['cpv']]

X_main, X_tesi, y_main, y_tesi = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  

X_exp, X_hype, y_exp, y_hype = train_test_split(X_tesi, y_tesi, test_size=0.2, random_state=42, stratify=y_tesi) 

X_train, X_test, y_train, y_test = train_test_split(X_exp, y_exp, test_size=0.2, random_state=42, stratify=y_exp) 

classifier = svm.LinearSVC(C=0.5)
pipe1 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
pipe1.fit(X_train, y_train)
predicted = pipe1.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("LinearSVC Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

classifier = MultinomialNB(alpha = 0.1, fit_prior = False)
pipe2 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
pipe2.fit(X_train, y_train)
predicted = pipe2.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("MultinomialNB Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

#classifier = MLPClassifier(alpha=0.01, hidden_layer_sizes=(150, 150, 150), max_iter=10)
#pipe3 = Pipeline([("cleaner", predictors()),
#                  ('vectorizer', vector),
#                  ('classifier', classifier)])
#pipe3.fit(X_train, y_train)
#predicted = pipe3.predict(X_test)
#acc = metrics.accuracy_score(y_test, predicted)
#print("MLPClassifier Accuracy:", acc)
#precision = metrics.precision_score(y_test, predicted, average='macro')
#recall = metrics.recall_score(y_test, predicted, average='macro')
#fm = metrics.f1_score(y_test, predicted, average='macro')
#print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
#print("======================================================")

# CPV
print("CPV")
y = [x[0:10] for x in df1['cpv']]

X_main, X_tesi, y_main, y_tesi = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)  

X_exp, X_hype, y_exp, y_hype = train_test_split(X_tesi, y_tesi, test_size=0.2, random_state=42, stratify=y_tesi) 

X_train, X_test, y_train, y_test = train_test_split(X_exp, y_exp, test_size=0.2, random_state=42, stratify=y_exp) 

classifier = svm.LinearSVC(C=0.5)
pipe1 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
pipe1.fit(X_train, y_train)
predicted = pipe1.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("LinearSVC Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

classifier = MultinomialNB(alpha = 0.1, fit_prior = False)
pipe2 = Pipeline([("cleaner", predictors()),
                  ('vectorizer', vector),
                  ('classifier', classifier)])
pipe2.fit(X_train, y_train)
predicted = pipe2.predict(X_test)
acc = metrics.accuracy_score(y_test, predicted)
print("MultinomialNB Accuracy:", acc)
precision = metrics.precision_score(y_test, predicted, average='macro')
recall = metrics.recall_score(y_test, predicted, average='macro')
fm = metrics.f1_score(y_test, predicted, average='macro')
print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
print("======================================================")

#classifier = MLPClassifier(alpha=0.01, hidden_layer_sizes=(150, 150, 150), max_iter=10)
#pipe3 = Pipeline([("cleaner", predictors()),
#                  ('vectorizer', vector),
#                  ('classifier', classifier)])
#pipe3.fit(X_train, y_train)
#predicted = pipe3.predict(X_test)
#acc = metrics.accuracy_score(y_test, predicted)
#print("MLPClassifier Accuracy:", acc)
#precision = metrics.precision_score(y_test, predicted, average='macro')
#recall = metrics.recall_score(y_test, predicted, average='macro')
#fm = metrics.f1_score(y_test, predicted, average='macro')
#print("P={0}, R={1}, F1={2}".format(precision, recall, fm))
#print("======================================================")
