import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, compile_and_fit

file_array = os.listdir('training_datasets')
records = {}

def build(model):
    model_record = {}
    for i in range(len(file_array)):
        print(f'Run {i}')
        df = pd.read_csv("training_datasets/"+file_array[i])

        time = df.pop('time')
        #splitting data by 70% training, 20% validating, 10% testing
        n = len(df)
        train_df = df[0:int(n*0.7)]
        val_df = df[int(n*0.7):int(n*0.9)]
        test_df = df[int(n*0.9):]

        #normalize data: subtract the mean and divide by the standard deviation of each feature.
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        df_std = (df - train_mean) / train_std
        df_std = df_std.melt(var_name='Column', value_name='Normalized')

        OUT_STEPS = 200
        multi_window = WindowGenerator(input_width=200, label_width=OUT_STEPS, shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)
        history = compile_and_fit(model, multi_window)
        model_record[i] = [model.evaluate(multi_window.val), model.evaluate(multi_window.test, verbose=0)]
    
    return model_record

linear = tf.keras.models.load_model('models/msm/linear')
records['Linear'] = build(linear)

dense = tf.keras.models.load_model('models/msm/dense')
records['Dense'] = build(dense)

conv = tf.keras.models.load_model('models/msm/conv')
records['Conv'] = build(conv)

lstm = tf.keras.models.load_model('models/msm/lstm')
records['LSTM'] = build(lstm)

feedback = tf.keras.models.load_model('models/msm/feedback')
records['Feedback LTSM'] = build(feedback)

print(records.items())

for name, item in records.items():
  print(f'Run {name}: {item[4][0][2]}, {item[4][1][2]}')