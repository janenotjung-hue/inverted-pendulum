import os
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, compile, fit_save

file_array = os.listdir('../training_datasets')
def getData(address):
    df = pd.read_csvs(f"../training_datasets/output_{address}.csv")

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

    return train_df, val_df, test_df, num_features

df= pd.read_csv('../training_datasets/output_1.csv')
train_df, val_df, test_df, num_features = getData(1)

#define training window
OUT_STEPS = 200
multi_window = WindowGenerator(input_width=200, label_width=OUT_STEPS, shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)

linear_path = "training_linear/cp.ckpt"
linear = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
compile(linear)
fit_save(linear, multi_window, linear_path)
linear.save('../')
