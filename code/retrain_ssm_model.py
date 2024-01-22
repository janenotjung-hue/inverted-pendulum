import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, compile_and_fit, compile_and_fit_checkpoints

file_array = os.listdir('training_datasets')
records = {}
def build(model, path_name):
    model_record = {}
    for i in range(len(file_array)-1):
        print(f'Run {i+1}')
        df = pd.read_csv("training_datasets/"+file_array[i])

        time = pd.to_numeric(df.pop('time'))

        plot_cols = ['theta', 'thetadot', 'x', 'xdot']
        plot_features = df[plot_cols]

        column_indices = {name: i for i, name in enumerate(df.columns)}

        #splitting data by 70% training, 20% validating, 10% testing
        n = len(df)
        train_df = df[0:int(n*0.7)]
        val_df = df[int(n*0.7):int(n*0.9)]
        test_df = df[int(n*0.9):]

        num_features = df.shape[1]

        #normalize data: subtract the mean and divide by the standard deviation of each feature.
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        df_std = (df - train_mean) / train_std
        df_std = df_std.melt(var_name='Column', value_name='Normalized')

        window = WindowGenerator(input_width=100, label_width=100, shift=1, train_df=train_df, val_df=val_df, test_df=test_df)
        compile_and_fit_checkpoints(model, window, checkpoint_path=f'checkpoints/ssm/{path_name}')
    return model_record

dense = tf.keras.models.load_model('checkpoints/ssm/dense')
records['Dense'] = build(dense, 'dense')

lstm = tf.keras.models.load_model('checkpoints/ssm/lstm')
records['LSTM'] = build(lstm, 'lstm')

residual_lstm = tf.keras.models.load_model('checkpoints/ssm/residual')
records['Residual LSTM'] = build(residual_lstm, 'residual')

print(records.items())