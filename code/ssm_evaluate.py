import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from setup import WindowGenerator, fit_checkpoints, create_ssm_dense_model, create_ssm_conv_model, create_ssm_lstm_model, create_ssm_residual_model

MAX_EPOCHS = 20
file_array = os.listdir('training_sets')
performance = []
def build(model, window, name):
    input_width = 1
    label_width = 1
    match window:
        case 'conv':
            input_width = 3
            label_width = 1
        case 'wide':
            label_width = 10
            input_width = 10
    for i in range(len(file_array)):
        print(f'Run {i+1}')
        df = pd.read_csv("training_sets/"+file_array[i])

        time = pd.to_numeric(df.pop('time'))

        n = len(df)
        train_df = df[0:int(n*0.7)]
        val_df = df[int(n*0.7):int(n*0.9)]
        test_df = df[int(n*0.9):]

        window = WindowGenerator(input_width=input_width, label_width=label_width, shift=1, train_df=train_df, val_df=val_df, test_df=test_df)
        #try record validation error
        loss, mae, mape = model.evaluate(window.test, verbose=2)
        performance.append([name, i+1, loss, mae, mape])
        

dense_untrained = create_ssm_dense_model()
build(dense_untrained, 'basic', 'Dense Untrained')

dense_cp = load_model('model_versions_new_format/ssm/dense/model.keras')
build(dense_cp, 'basic', 'Dense Trained')

#conv_untrained = create_ssm_conv_model()
#build(conv_untrained, 'conv', 'Conv Untrained')
#
#conv_cp = tf.train.latest_checkpoint('model_versions_without_normalizing/ssm/conv/')
#conv_trained = create_ssm_conv_model()
#conv_trained.load_weights(conv_cp).expect_partial()
#build(conv_trained, 'conv', 'Conv Trained')
#
#lstm_untrained = create_ssm_lstm_model()
#build(lstm_untrained, 'wide', 'LSTM Untrained')
#
#lstm_cp = tf.train.latest_checkpoint('model_versions_without_normalizing/ssm/lstm/')
#lstm_trained = create_ssm_lstm_model()
#lstm_trained.load_weights(lstm_cp).expect_partial()
#build(lstm_trained, 'wide', 'LSTM Trained')
#
#residual_untrained = create_ssm_residual_model()
#build(residual_untrained, 'wide', 'Residual Untrained')
#
#residual_cp = tf.train.latest_checkpoint('model_versions_without_normalizing/ssm/residual/')
#residual_trained = create_ssm_residual_model()
#residual_trained.load_weights(residual_cp).expect_partial()
#build(residual_trained, 'wide', 'Residual Trained')
#
#performance = np.array(performance)
#data = pd.DataFrame(performance, columns=["Name", "Test #", "Loss", "Mean Absolute Error (MAE)", "Mean Absolute Percentange Error (MAPE) %"])
#print(data)
#data.to_csv('ssm_performance_without_normalizing.csv', index=False)