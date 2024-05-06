import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from setup import WindowGenerator, create_ssm_lstm_single_model, create_ssm_lstm_model

from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam


def createModel(l1Nodes, l2Nodes, d1Nodes, d2Nodes, inputShape):
    # input layer
    lstm1 = LSTM(l1Nodes, input_shape=inputShape, return_sequences=True)
    lstm2 = LSTM(l2Nodes, return_sequences=True)
    flatten = Flatten()
    dense1 = Dense(d1Nodes)
    dense2 = Dense(d2Nodes)

    # output layer
    outL = Dense(1, activation='relu')
    # combine the layers
    layers = [lstm1, lstm2, flatten,  dense1, dense2, outL]
    # create the model
    model = Sequential(layers)
    opt = Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss='mse')
    return model

def convert_to_array(dataset):
    features_list = []
    labels_list = []
    for features, labels in dataset:
        features_list.append(features)
        labels_list.append(labels) 

    # Concatenate arrays
    features_array = tf.concat(features_list, axis=0).numpy()
    labels_array = tf.concat(labels_list, axis=0).numpy()

    return features_array, labels_array

    

input_width = 10
label_width = 1
df = pd.read_csv("training_sets_nudge/output1_1.csv")

time = pd.to_numeric(df.pop('time'))
features = df.columns

n = len(df)
train_df = df[0:int(n*0.8)]
test_df = df[int(n*0.8):]

window = WindowGenerator(input_width=input_width, label_width=label_width, shift=1, train_df=train_df, val_df=[], test_df=test_df, label_columns=['theta'])

print(window.train)

X_train, y_train = convert_to_array(window.train)
X_test, y_test = convert_to_array(window.test)


#convert y data into 1D array for some reason it doesn't work in the methods
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

model = createModel(32, 32, 8, 8, (X_train.shape[1], X_train.shape[2]))

model.fit(X_train, y_train, epochs=20, verbose=2)
model.save('my_model_test.h5')

#model = load_model('my_model_test.h5')
#print(model.summary())

import shap

explainer = shap.DeepExplainer(model, X_train)

print(explainer.expected_value[0])
shap_values = explainer.shap_values(X_test)
len(shap_values)

shap.initjs()

