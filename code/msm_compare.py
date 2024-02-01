import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, fit_checkpoints, create_msm_linear_model, create_msm_dense_model, create_msm_conv_model, create_msm_lstm_model, create_msm_feedback_model

OUT_STEPS = 200
file_array = os.listdir('training_sets')
performance = []
def build(model, name):
   for i in range(len(file_array)):
      print(f'Run {i+1}')
      df = pd.read_csv("training_sets/"+file_array[i])

      time = pd.to_numeric(df.pop('time'))

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

      window = WindowGenerator(input_width=200, label_width=OUT_STEPS, shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)
      loss, mae, mape = model.evaluate(window.test, verbose=2)
      performance.append([name, i+1, loss, mae, mape])

linear_untrained = create_msm_linear_model()
build(linear_untrained, 'Linear Untrained')

linear_cp = tf.train.latest_checkpoint('model_versions/msm/linear/')
linear_trained = create_msm_linear_model()
linear_trained.load_weights(linear_cp).expect_partial()
build(linear_trained, 'Linear Trained')

dense_untrained = create_msm_dense_model()
build(dense_untrained,'Dense Untrained')

dense_cp = tf.train.latest_checkpoint('model_versions/msm/dense/')
dense_trained = create_msm_dense_model()
dense_trained.load_weights(dense_cp).expect_partial()
build(dense_trained, 'Dense Trained')

conv_untrained = create_msm_conv_model()
build(conv_untrained,'Conv Untrained')

conv_cp = tf.train.latest_checkpoint('model_versions/msm/conv/')
conv_trained = create_msm_conv_model()
conv_trained.load_weights(conv_cp).expect_partial()
build(conv_trained,'Conv Trained')

lstm_untrained = create_msm_lstm_model()
build(lstm_untrained,'LSTM Untrained')

lstm_cp = tf.train.latest_checkpoint('model_versions/msm/lstm/')
lstm_trained = create_msm_lstm_model()
lstm_trained.load_weights(lstm_cp).expect_partial()
build(lstm_trained,'LSTM Trained')

feedback_untrained = create_msm_feedback_model()
build(feedback_untrained,'Feedback Untrained')

feedback_cp = tf.train.latest_checkpoint('model_versions/msm/feedback/')
feedback_trained = create_msm_feedback_model()
feedback_trained.load_weights(feedback_cp).expect_partial()
build(feedback_trained, 'Feedback Trained')

performance = np.array(performance)
data = pd.DataFrame(performance, columns=["Name", "Test #", "Loss", "Mean Absolute Error (MAE)", "Mean Absolute Percentange Error (MAPE) %"])
print(data)
data.to_csv('msm_performance.csv', index=False)