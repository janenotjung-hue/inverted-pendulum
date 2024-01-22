import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, compile_and_fit, compile_and_fit_checkpoints

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df= pd.read_csv('training_datasets/output_1.csv')

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

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

baseline = Baseline()
baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])

val_performance['Baseline'] = baseline.evaluate(window.val)
performance['Baseline'] = baseline.evaluate(window.test, verbose=0)

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=num_features)
])
history = compile_and_fit_checkpoints(dense, window, checkpoint_path=f'{cp_shortcut}/dense/cp.ckpt')

val_performance['Dense'] = dense.evaluate(window.val)
performance['Dense'] = dense.evaluate(window.test, verbose=0)
print()


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

history = compile_and_fit_checkpoints(lstm_model, window, checkpoint_path=f'{cp_shortcut}/lstm/cp.ckpt')

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate( window.val)
performance['LSTM'] = lstm_model.evaluate( window.test, verbose=0)

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)
    return inputs + delta

residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        kernel_initializer=tf.initializers.zeros())
]))

history = compile_and_fit_checkpoints(residual_lstm, window, checkpoint_path=f'{cp_shortcut}/residual/cp.ckpt')

val_performance['Residual LSTM'] = residual_lstm.evaluate(window.val)
performance['Residual LSTM'] = residual_lstm.evaluate(window.test, verbose=0)

print()

x = np.arange(len(performance))
width = 0.3

print('Loss, MAE, MAPE for all models')
for name, value in performance.items():
  print(f'{name:15s}: {value[0]:0.5e}, {value[1]:0.5f}, {value[2]:0.2f}%')

metric_name = 'mean_absolute_percentage_error'
metric_index = lstm_model.metrics_names.index(metric_name)
val_mape = [v[metric_index] for v in val_performance.values()]
test_mape = [v[metric_index] for v in performance.values()]

rects = plt.bar(x - 0.17, val_mape, width, label='Validation')
plt.bar_label(rects, fmt='{:.2f}%', padding=3)

rects = plt.bar(x + 0.17, test_mape, width, label='Test')
plt.bar_label(rects, fmt='{:.2f}%', padding=3)

plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
plt.ylabel('MAPE %')
_ = plt.legend()