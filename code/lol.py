import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from setup import WindowGenerator, create_ssm_lstm_single_model, create_ssm_lstm_model

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

model = create_ssm_lstm_single_model(32, 32, 8, 8, (X_train.shape[1], X_train.shape[2]))

#model.fit(X_train, y_train, epochs=20, verbose=2)
#model.save('my_model_test.h5')

model = load_model('my_model_test.h5')
#print(model.summary())

import shap

explainer = shap.GradientExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

shap.initjs()

print(X_test.shape)

i = 0
j = 0
x_test_df = pd.DataFrame(data=X_test[i][j].reshape(1,32), columns = features)
shap.force_plot(explainer.expected_value[0], shap_values[0][i][j], x_test_df)
