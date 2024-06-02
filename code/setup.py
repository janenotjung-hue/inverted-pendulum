import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def get_datasets(dropCol):
    file_array = os.listdir('../training_sets_nudge')
    X_trains, X_tests, y_trains, y_tests = [],[],[],[]
    for i in range(len(file_array)):
        df = pd.read_csv("../training_sets_nudge/"+file_array[i])
        df = df.drop(columns=["time"])
        if(dropCol != None): df = df.drop(columns=[dropCol])
        n = len(df)
        label_df = []
        data_df = df.iloc[:-1]
        
        temp_df = df['x'].iloc[1:]
        
        for i in range(n-1):
            if data_df['x'][i] < temp_df.get(i+1):
                label_df.insert(i, 1) #right 
            else:
                label_df.insert(i, 0) #left

        X = data_df
        y = pd.Series(data=label_df, name="direction")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
        
    
    return X_trains, X_tests, y_trains, y_tests


"""
    predictions = values returned after model.predict(X_test) was called.
    actual = y_test.
    
    Returns:
        - cm (Confusion Matrix)
        - acc (Accuracy) = number of times direction outcome was prediction correctly.
        - sen (Sensitivity) = number of times going left was predicted correctly.
        - spe (Specificity) = number of times going right was predicted correctly.
"""
def get_results(predictions: np.array, actual: np.array):
    predictions_nominal = [ 0 if x < 0.5 else 1 for x in predictions]
    cm = confusion_matrix(actual, predictions_nominal)
    #print(cm)
    #print("Accuracy: ", round(np.sum(np.diagonal(cm))/np.sum(cm),3))
    #print("Sensitivity: ", round(cm[1,1]/np.sum(cm[1,:]),3))
    #print("Specificity: ", round(cm[0,0]/np.sum(cm[0,:]),3))
    acc = np.sum(np.diagonal(cm))/np.sum(cm)
    sen = cm[1,1]/np.sum(cm[1,:])
    spe = cm[0,0]/np.sum(cm[0,:])
    
    return cm, acc, sen, spe
    
    