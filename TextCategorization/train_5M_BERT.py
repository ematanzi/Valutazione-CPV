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
#model_name="dlicari/Italian-Legal-BERT"

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
#X_train = []
#y_train = []
for line in ft:
    jo = json.loads(line)
    #X_train.append(jo["source"])
    #y_train.append(jo["target"][0:10])
    data.append((jo["source"], jo["target"][0:2]))
    labels.append(jo["target"][0:2])
ft.close()

#y_train = [x[0:2] for x in y_train]
random.shuffle(data)

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

# Load the MainLayer
bert = transformer_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
# attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32')
# inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Then build your model output
le.fit(labels)
print("Number of classes: ", len(le.classes_))
cpv = Dense(units=len(le.classes_), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='cpv')(pooled_output)
outputs = {'cpv': cpv}

# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')

# Take a look at the model
model.summary()

#######################################
### ------- Train the model ------- ###

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = {'cpv': CategoricalCrossentropy(from_logits = True)}
metric = {'cpv': CategoricalAccuracy('accuracy')}

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['acc', f1_m, precision_m, recall_m])

def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]

dataSplits = list(divide_chunks(data, 128000))

for split in dataSplits:
    X_train = []
    y_train = []
    for t in split:
        X_train.append(t[0])
        y_train.append(t[1])
    # Ready output data for the model
    y_transform = le.transform(y_train)
    y_cpv = to_categorical(y_transform, num_classes=len(le.classes_))

    # Tokenize the input (takes some time)
    x = tokenizer(
        text=X_train,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True)

    print("Fit...")
    # Fit the model
    history = model.fit(
        # x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
        x={'input_ids': x['input_ids']},
        y={'cpv': y_cpv},
        validation_split=0.2,
        batch_size=16,
        epochs=5)
    
print("Save...")
model.save('bert-base-cpv-5M-64-e5.h5')

data_test = []
ft = open("ds_5M/cpv_test.json","r")
for line in ft:
    jo = json.loads(line)
    data_test.append((jo["source"], jo["target"][0:2]))
ft.close()

dataSplits = list(divide_chunks(data_test, 10000))

#with custom_object_scope({'f1_m': f1_m},{'precision_m': precision_m},{'recall_m': recall_m}):
#    reconstructed_model = keras.models.load_model("bert-base-500000-16.h5")
#    reconstructed_model.compile(optimizer=optimizer, loss=loss, metrics=['acc', f1_m, precision_m, recall_m])

#######################################
### ----- Evaluate the model ------ ###

for split in dataSplits:
    X_test = []
    y_test = []
    for t in split:
        X_test.append(t[0])
        y_test.append(t[1])
    # Ready output data for the model
    y_transform = le.transform(y_test)
    test_y_cpv = to_categorical(y_transform, num_classes=len(le.classes_))

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
    predict = model.predict(x={'input_ids': test_x['input_ids']})
    predicted = predict['cpv']
    y_pred_numeric = np.argmax(predicted, axis=-1)
    print(metrics.classification_report(y_transform, y_pred_numeric, target_names=categories, digits=4))
