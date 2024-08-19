import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from joblib import dump, load

def get_datasets(dropCol):
    file_array = os.listdir('../training_sets_nudge')
    X_trains, X_tests, y_trains, y_tests = [],[],[],[]
    for i in range(len(file_array)):
        df = pd.read_csv("../training_sets_nudge/"+file_array[i])
        df = df.drop(columns=["time"])
        
        n = len(df)
        label_df = []
        data_df = df.iloc[:-1]
        
        temp_df = df['x'].iloc[1:]
        
        for i in range(n-1):
            if data_df['x'][i] < temp_df.get(i+1):
                label_df.insert(i, 1) #right 
            else:
                label_df.insert(i, 0) #left

        if(dropCol != None): data_df = data_df.drop(columns=dropCol)
        X = data_df
        y = pd.Series(data=label_df, name="direction")

        train_size = int(len(y) * 0.9)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
        
    
    return X_trains, X_tests, y_trains, y_tests
    
X_trains, X_tests, y_trains, y_tests = get_datasets(None)
X_train_combined = pd.concat(X_trains, axis=0).reset_index(drop=True)
X_test_combined = pd.concat(X_tests, axis=0).reset_index(drop=True)
y_train_combined = pd.concat(y_trains, axis=0).reset_index(drop=True)
y_test_combined = pd.concat(y_tests, axis=0).reset_index(drop=True)

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
    if np.array_equal(predictions, actual):
        return [], 1, 1, 1
    cm = confusion_matrix(actual, predictions_nominal)
    #print(cm)
    
    #print("Accuracy: ", round(np.sum(np.diagonal(cm))/np.sum(cm),3))
    #print("Sensitivity: ", round(cm[1,1]/np.sum(cm[1,:]),3))
    #print("Specificity: ", round(cm[0,0]/np.sum(cm[0,:]),3))
    acc = np.sum(np.diagonal(cm))/np.sum(cm)
    sen = cm[1,1]/np.sum(cm[1,:])
    spe = cm[0,0]/np.sum(cm[0,:])
    
    return cm, acc, sen, spe
    

# mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500, verbose=True)
# # for i in range(len(X_trains)):  
# #     mlp.fit(X_trains[i], y_trains[i])

# mlp.fit(X_train_combined, y_train_combined)
# dump(mlp, 'code/xai_models/mlp_model.joblib')

# tree = ExtraTreesClassifier()
# # for i in range(len(X_trains)):  
# #     classifier.fit(X_trains[i], y_trains[i])
# tree.fit(X_train_combined, y_train_combined)
# dump(tree, 'code/xai_models/tree_class_model.joblib')

