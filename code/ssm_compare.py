import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, compile
import os

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)
    return inputs + delta

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
cp_shortcut = 'checkpoints/ssm'
window = WindowGenerator(input_width=100, label_width=100, shift=1, train_df=train_df, val_df=val_df, test_df=test_df)
val_performance = {}
performance = {}

def create_dense_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=num_features)])
    return compile(model)

def create_lstm_model():
   model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=num_features)])
   return compile(model)

def create_residual_model():
   model = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        kernel_initializer=tf.initializers.zeros())]))
   return compile(model)


dense_untrained = create_dense_model()
performance['Dense untrained'] = dense_untrained.evaluate(window.test, verbose=2)

dense_cp = tf.train.latest_checkpoint('checkpoints/ssm/dense_2/')
dense_trained = create_dense_model()
dense_trained.load_weights(dense_cp).expect_partial()
performance['Dense trained'] = dense_trained.evaluate(window.test, verbose=2)

lstm_untrained = create_lstm_model()
performance['LSTM untrained'] = lstm_untrained.evaluate(window.test, verbose=2)

lstm_cp = tf.train.latest_checkpoint('checkpoints/ssm/lstm_2/')
lstm_trained = create_lstm_model()
lstm_trained.load_weights(lstm_cp).expect_partial()
performance['LSTM trained'] = lstm_trained.evaluate(window.test, verbose=2)

residual_untrained = create_residual_model()
performance['Residual untrained'] = residual_untrained.evaluate(window.test, verbose=2)

residual_cp = tf.train.latest_checkpoint('checkpoints/ssm/residual_2/')
residual_trained = create_residual_model()
residual_trained.load_weights(residual_cp).expect_partial()
performance['Residual trained'] = residual_trained.evaluate(window.test, verbose=2)


print('Loss, MAE, MAPE for all models')
for name, value in performance.items():
  print(f'{name:15s}: {value[0]:0.5e}, {value[1]:0.5f}, {value[2]:0.2f}%')
