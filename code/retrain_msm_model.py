import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, compile_and_fit, compile, fit_save

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df= pd.read_csv('../training_datasets/output_1.csv')

time = pd.to_numeric(df.pop('time'))

plot_cols = ['theta', 'thetadot', 'x', 'xdot']
plot_features = df[plot_cols]
#plot_features.index = time
#_ = plot_features.plot(subplots=True)

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

def create_linear(): 
    model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
    model = compile(model)
    return model

#define training window
OUT_STEPS = 200
multi_window = WindowGenerator(input_width=200, label_width=OUT_STEPS, shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)
val_performance = {}
performance = {}

linear_path = "training_linear/cp.ckpt"
linear = create_linear()
fit_save(linear, multi_window, linear_path)

linear = linear = create_linear()
linear.load_weights(linear_path)
loss, mae, mape = linear.evaluate(multi_window.test, verbose=0)
print(f"Untrained model, inaccuracy: {mape:5.3f}%")
