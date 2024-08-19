import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, fit_checkpoints, create_msm_linear_model, create_msm_dense_model, create_msm_conv_model, create_msm_lstm_model, create_msm_feedback_model
from keras.models import load_model

OUT_STEPS = 200
file_array = os.listdir('training_data')
performance = []
def build(model, name):
   for i in range(len(file_array)):
      print(f'Run {i+1}')
      df = pd.read_csv("training_data/"+file_array[i])

      time = pd.to_numeric(df.pop('time'))

      #splitting data by 70% training, 20% validating, 10% testing
      n = len(df)
      train_df = df[0:int(n*0.7)]
      val_df = df[int(n*0.7):int(n*0.9)]
      test_df = df[int(n*0.9):]

      window = WindowGenerator(input_width=200, label_width=OUT_STEPS, shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)
      loss, mae, mape = model.evaluate(window.test, verbose=2)
      performance.append([name, i+1, loss, mae, mape])

dense_untrained = create_msm_dense_model()
build(dense_untrained,'Dense Untrained')

dense_cp = load_model('trained_models/msm/dense/model.keras')
build(dense_cp, 'Dense Trained')

conv_untrained = create_msm_conv_model()
build(conv_untrained,'Conv Untrained')

conv_cp = load_model('trained_models/msm/conv/model.keras')
build(conv_cp,'Conv Trained')

lstm_untrained = create_msm_lstm_model()
build(lstm_untrained,'LSTM Untrained')

lstm_cp = load_model('trained_models/msm/lstm/model.keras')
build(lstm_cp,'LSTM Trained')

feedback_untrained = create_msm_feedback_model()
build(feedback_untrained,'Feedback Untrained')

feedback_cp = load_model('trained_models/msm/feedback/model.keras')
build(feedback_cp, 'Feedback Trained')

performance = np.array(performance)
data = pd.DataFrame(performance, columns=["Name", "Test #", "Loss", "Mean Absolute Error (MAE)", "Mean Absolute Percentange Error (MAPE) %"])
print(data)
data.to_csv('results/final_nudges/msm_test_performance.csv', index=False)