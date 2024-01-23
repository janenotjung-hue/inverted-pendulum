import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, compile
import os

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df= pd.read_csv('training_datasets/output_5.csv')

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

#define training window
cp_shortcut = 'checkpoints/msm'
OUT_STEPS = 200
window = WindowGenerator(input_width=200, label_width=OUT_STEPS, shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)
performance = {}

def create_linear_model():
    model =  tf.keras.Sequential([
       tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
       tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),
       tf.keras.layers.Reshape([OUT_STEPS, num_features])])
    return compile(model)

def create_dense_model():
    model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    return compile(model)

def create_conv_model():
   CONV_WIDTH = 3
   model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
   return compile(model)

def create_lstm_model():
   model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])])
   return compile(model)

linear_untrained = create_linear_model()
performance['Linear untrained'] = linear_untrained.evaluate(window.test, verbose=2)

linear_cp = tf.train.latest_checkpoint('checkpoints/msm/linear_2/')
linear_trained = create_linear_model()
linear_trained.load_weights(linear_cp).expect_partial()
performance['Linear trained'] = linear_trained.evaluate(window.test, verbose=2)


dense_untrained = create_dense_model()
performance['Dense untrained'] = dense_untrained.evaluate(window.test, verbose=2)

dense_cp = tf.train.latest_checkpoint('checkpoints/msm/dense_2/')
dense_trained = create_dense_model()
dense_trained.load_weights(dense_cp).expect_partial()
performance['Dense trained'] = dense_trained.evaluate(window.test, verbose=2)

lstm_untrained = create_lstm_model()
performance['LSTM untrained'] = lstm_untrained.evaluate(window.test, verbose=2)

lstm_cp = tf.train.latest_checkpoint('checkpoints/msm/lstm_2/')
lstm_trained = create_lstm_model()
lstm_trained.load_weights(lstm_cp).expect_partial()
performance['LSTM trained'] = lstm_trained.evaluate(window.test, verbose=2)

conv_untrained = create_conv_model()
performance['Conv untrained'] = conv_untrained.evaluate(window.test, verbose=2)

conv_cp = tf.train.latest_checkpoint('checkpoints/msm/conv_2/')
conv_trained = create_conv_model()
conv_trained.load_weights(conv_cp).expect_partial()
performance['Conv trained'] = conv_trained.evaluate(window.test, verbose=2)


print('Loss, MAE, MAPE for all models')
for name, value in performance.items():
  print(f'{name:15s}: {value[0]:0.5e}, {value[1]:0.5f}, {value[2]:0.2f}%')
