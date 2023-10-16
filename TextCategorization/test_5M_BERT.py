import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import trange
import random
import collections
import json
import random

from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow import keras  # todo add
from keras.utils import custom_object_scope
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
#######################################
### --------- Setup BERT ---------- ###

from transformers import AutoModel, AutoTokenizer, BertConfig, TFBertModel

model_name = "dbmdz/bert-base-italian-uncased"

# Max length of tokens
max_length = 128

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)

# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config=config, from_pt=True)

#######################################
### --------- Import data --------- ###

ft = open("ds_5M/cpv_train.json","r")
data = []
labels = []
le = LabelEncoder()
for line in ft:
    jo = json.loads(line)
    labels.append(jo["target"][0:2])
ft.close()
le.fit(labels)
print("Number of classes: ", len(le.classes_))

#######################################
### ------- Build the model ------- ###

# TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model

import tensorflow as tf
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = {'cpv': CategoricalCrossentropy(from_logits = True)}
metric = {'cpv': CategoricalAccuracy('accuracy')}

with custom_object_scope({'f1_m': f1_m},{'precision_m': precision_m},{'recall_m': recall_m}):
    reconstructed_model = keras.models.load_model('bert-base-cpv-5M-64-e5.h5')
    reconstructed_model.compile(optimizer=optimizer, loss=loss, metrics=['acc', f1_m, precision_m, recall_m])


data_test = []
ft = open("ds_5M/cpv_test.json","r")
for line in ft:
    jo = json.loads(line)
    data_test.append((jo["source"], jo["target"][0:2]))
ft.close()

dataSplits = list(divide_chunks(data_test, 10000))

#######################################
### ----- Evaluate the model ------ ###

y_transform = []
y_pred_numeric = []
for split in dataSplits:
    X_test = []
    y_test = []
    for t in split:
        X_test.append(t[0])
        y_test.append(t[1])
    # Ready output data for the model
    y_transform.append(le.transform(y_test))
    #test_y_cpv = to_categorical(y_transform, num_classes=len(le.classes_))

    # Ready test data
    test_x = tokenizer(
        text=X_test,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = False,
        verbose = True)
    print("Test..")
    predict = reconstructed_model.predict(x={'input_ids': test_x['input_ids']})
    predicted = predict['cpv']
    y_pred_numeric.append(np.argmax(predicted, axis=-1))

print(metrics.classification_report(y_transform, y_pred_numeric, digits=4))
