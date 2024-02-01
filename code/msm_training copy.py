import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, fit_checkpoints, create_msm_linear_model, create_msm_dense_model, create_msm_conv_model, create_msm_lstm_model, create_msm_feedback_model

OUT_STEPS = 200
file_array = os.listdir('more_training')
def build(model, path_name):
    for i in range(len(file_array)):
        print(f'Run {i+1}')
        df = pd.read_csv("more_training/"+file_array[i])

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
        checkpoint_path = path_name+'/cp-{epoch:04d}.ckpt'
        history = fit_checkpoints(model, window, checkpoint_path=checkpoint_path)
    return history

linear_cp = tf.train.latest_checkpoint('model_versions/msm/linear/')
linear = create_msm_linear_model()
linear.load_weights(linear_cp).expect_partial()
build(linear, 'model_versions/msm/linear/')

#dense_cp = tf.train.latest_checkpoint('model_versions/msm/dense/')
#dense_trained = create_msm_dense_model()
#dense_trained.load_weights(dense_cp).expect_partial()
#build(dense_trained, 'model_versions/msm/dense/')
#
#conv_cp = tf.train.latest_checkpoint('model_versions/msm/conv/')
#conv = create_msm_conv_model()
#conv.load_weights(conv_cp).expect_partial()
#build(conv, 'model_versions/msm/conv/')
#
#lstm_cp = tf.train.latest_checkpoint('model_versions/msm/lstm/')
#lstm = create_msm_lstm_model()
#lstm.load_weights(lstm_cp).expect_partial()
#build(lstm, 'model_versions/msm/lstm/')

#feedback_cp = tf.train.latest_checkpoint('model_versions/msm/feedback/')
#feedback = create_msm_feedback_model()
#feedback.load_weights(feedback_cp).expect_partial()
#build(feedback, 'model_versions/msm/feedback/')

#linear = create_msm_linear_model()
#build(linear, 'model_versions/msm/linear')

#dense = create_msm_dense_model()
#history = build(dense, 'model_versions/msm/dense')

#conv = create_msm_conv_model()
#build(conv, 'model_versions/msm/conv')

lstm = create_msm_lstm_model()
build(lstm, 'model_versions/msm/lstm')

#feedback = create_msm_feedback_model()
#build(feedback, 'model_versions/msm/feedback')