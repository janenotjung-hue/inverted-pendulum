import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, fit_checkpoints, create_msm_linear_model, create_msm_dense_model, create_msm_conv_model, create_msm_lstm_model, create_msm_feedback_model
from keras.models import load_model

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

      window = WindowGenerator(input_width=200, label_width=OUT_STEPS, shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)
      loss, mae, mape = model.evaluate(window.val, verbose=2)
      performance.append([name, i+1, loss, mae, mape])

#linear_untrained = create_msm_linear_model()
#build(linear_untrained, 'Linear Untrained')
#
#linear_cp = tf.train.latest_checkpoint('model_versions_without_normalizing/msm/linear/')
#linear_trained = create_msm_linear_model()
#linear_trained.load_weights(linear_cp).expect_partial()
#build(linear_trained, 'Linear Trained')

#dense_untrained = create_msm_dense_model()
#build(dense_untrained,'Dense Untrained')
#
dense_cp = load_model('model_versions_new_format/msm/dense/model.keras', safe_mode=False)
#build(dense_cp, 'Dense Trained')
#
#conv_untrained = create_msm_conv_model()
#build(conv_untrained,'Conv Untrained')
#
conv_cp = load_model('model_versions_new_format/msm/conv/model.keras', safe_mode=False)
#build(conv_cp,'Conv Trained')
#
#lstm_untrained = create_msm_lstm_model()
#build(lstm_untrained,'LSTM Untrained')
#
lstm_cp = load_model('model_versions_new_format/msm/lstm/model.keras', safe_mode=False)
#build(lstm_cp,'LSTM Trained')
#
#feedback_untrained = create_msm_feedback_model()
#build(feedback_untrained,'Feedback Untrained')
#
feedback_cp = load_model('model_versions_new_format/msm/feedback/model.keras', safe_mode=False)
#build(feedback_cp, 'Feedback Trained')
#
print(dense_cp.summary())
print(conv_cp.summary())
print(lstm_cp.summary())
print(feedback_cp.summary())
#performance = np.array(performance)
#data = pd.DataFrame(performance, columns=["Name", "Test #", "Loss", "Mean Absolute Error (MAE)", "Mean Absolute Percentange Error (MAPE) %"])
#print(data)
#data.to_csv('results/final/msm_val_performance.csv', index=False)

tf.keras.utils.plot_model(dense_cp, to_file="diagrams\msm_dense.png", show_shapes=True, show_layer_names=True, rankdir="TB")
tf.keras.utils.plot_model(conv_cp, to_file="diagrams\msm_conv.png", show_shapes=True, show_layer_names=True, rankdir="TB")
tf.keras.utils.plot_model(lstm_cp, to_file="diagrams\msm_lstm.png", show_shapes=True, show_layer_names=True, rankdir="TB")
tf.keras.utils.plot_model(feedback_cp, to_file="diagrams\msm_feedback.png", show_shapes=True, show_layer_names=True, rankdir="TB")