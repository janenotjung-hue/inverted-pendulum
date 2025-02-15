import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from setup import WindowGenerator, fit_checkpoints, create_msm_linear_model, create_msm_dense_model, create_msm_conv_model, create_msm_lstm_model, create_msm_feedback_model

OUT_STEPS = 200
file_array = os.listdir('training_data')
def build(model, path_name):
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
        checkpoint_path = ('%s/cp-{epoch:02d}-{loss:.4f}.keras' % (path_name))
        # !! record history into a csv to check loss and val_loss !!
        history = fit_checkpoints(model, window, checkpoint_path=checkpoint_path)
    model.save(f'{path_name}/model.keras')
    return history

dense = create_msm_dense_model()
history = build(dense, 'trained_models/msm/dense')

conv = create_msm_conv_model()
build(conv, 'trained_models/msm/conv')

lstm = create_msm_lstm_model()
build(lstm, 'trained_models/msm/lstm')

feedback = create_msm_feedback_model()
build(feedback, 'trained_models/msm/feedback')