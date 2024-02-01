import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_array = os.listdir('training_sets')
data_info=[]
for i in range(len(file_array)):
    print(f'Run {i+1}')
    df = pd.read_csv("training_sets/"+file_array[i])

    time = pd.to_numeric(df.pop('time'))

    #splitting data by 70% training, 20% validating, 10% testing
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()
    test_df = (test_df - train_mean) / train_std
    data_info.append([file_array[i], train_mean, train_std])

data = pd.DataFrame(data_info, columns=["Name", "Mean", "STD"])
pd.set_option("display.max_rows", None, "display.max_columns", None)

data.to_csv('dataset_info.csv', index=False)