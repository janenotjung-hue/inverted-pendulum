import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, fit_checkpoints, create_ssm_dense_model, create_ssm_conv_model, create_ssm_lstm_model, create_ssm_residual_model

MAX_EPOCHS = 20
file_array = os.listdir('more_training')
def build(model, window, path_name):
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

        window = WindowGenerator(input_width=input_width, label_width=label_width, shift=1, train_df=train_df, val_df=val_df, test_df=test_df)
        checkpoint_path = path_name+'/cp-{epoch:04d}.ckpt'
        history = fit_checkpoints(model, window, checkpoint_path=checkpoint_path)
    return history

dense_cp = tf.train.load_checkpoint('model_versions/ssm/dense/')
dense_trained = create_ssm_dense_model()
dense_trained.load_weights(dense_cp).expect_partial()
build(dense_trained, 'basic', 'model_versions/ssm/dense')

conv_cp = tf.train.load_checkpoint('model_versions/ssm/conv/')
conv_trained = create_ssm_conv_model()
conv_trained.load_weights(conv_cp).expect_partial()
build(conv_trained, 'conv', 'model_versions/ssm/conv/')

lstm_cp = tf.train.load_checkpoint('model_versions/ssm/lstm/')
lstm_trained = create_ssm_lstm_model()
lstm_trained.load_weights(lstm_cp).expect_partial()
build(lstm_trained, 'wide', 'model_versions/ssm/lstm/')

residual_cp = tf.train.load_checkpoint('model_versions/ssm/residual/')
residual_trained = create_ssm_residual_model()
residual_trained.load_weights(residual_cp).expect_partial()
build(residual_trained, 'wide', 'model_versions/ssm/residual/')

#conv = create_ssm_conv_model()
#build(conv, 'conv', 'model_versions/ssm/conv')

#lstm = create_ssm_lstm_model()
#build(lstm, 'wide', 'model_versions/ssm/lstm')

#residual = create_ssm_residual_model()
#build(residual, 'wide', 'model_versions/ssm/residual')
