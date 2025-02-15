import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, fit_checkpoints, create_ssm_dense_model, create_ssm_conv_model, create_ssm_lstm_model, create_ssm_residual_model

file_array = os.listdir('training_data')
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
        df = pd.read_csv("training_data/"+file_array[i])

        time = df.pop('time')

        #splitting data by 70% training, 20% validating, 10% testing
        n = len(df)
        train_df = df[0:int(n*0.7)]
        val_df = df[int(n*0.7):int(n*0.9)]
        test_df = df[int(n*0.9):]

        window = WindowGenerator(input_width=input_width, label_width=label_width, shift=1, train_df=train_df, val_df=val_df, test_df=test_df)
        checkpoint_path = ('%s/cp-{epoch:02d}-{loss:.4f}.keras' % (path_name))
        history = fit_checkpoints(model, window, checkpoint_path=checkpoint_path)
    model.save(f'{path_name}/model.keras')
    return history

dense = create_ssm_dense_model()
build(dense, 'basic', 'trained_models/ssm/dense')
#
conv = create_ssm_conv_model()
build(conv, 'conv', 'trained_models/ssm/conv')
#
lstm = create_ssm_lstm_model()
build(lstm, 'wide', 'trained_models/ssm/lstm')

residual = create_ssm_residual_model()
build(residual, 'wide', 'trained_models/ssm/residual')
