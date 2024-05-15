import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from setup import WindowGenerator

def create_ssm_dense_model(n : int):
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=n)])
    return compile(model)

input_width = 10
label_width = 1
df = pd.read_csv("../training_sets_nudge/output1_1.csv")

features = df.columns

n = len(df)
label_df = []
data_df = df.iloc[:-1]

#TODO: change index of temp_df to start from 0
temp_df = df['x'].iloc[1:]

for i in range(n-1):
    if data_df['x'][i] < temp_df.get(i+1):
        label_df.insert(i, 1) #right 
    else:
        label_df.insert(i, 0) #left

X = data_df
y = label_df

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=50)
df = pd.concat([X, pd.DataFrame(y)], axis=1)
df.columns = np.concatenate((features, ['label']))

print("Shape of data =", df.shape)
features = np.array(df.columns)

from sklearn.ensemble import ExtraTreesRegressor
reg = ExtraTreesRegressor(random_state=50)

reg.fit(X_train, y_train)
print('R2 score for the model on test set =', reg.score(X_test, y_test))

X_train = np.array(X_train)
from lime import lime_tabular
explainer_lime = lime_tabular.LimeTabularExplainer(X_train,
                                                   feature_names=features,
                                                   verbose=True, 
                                                   mode='regression')



# Index corresponding to the test vector
i = 10
 
# Number denoting the top features
k = 5
 
# Calling the explain_instance method by passing in the:
#    1) ith test vector
#    2) prediction function used by our prediction model('reg' in this case)
#    3) the top features which we want to see, denoted by k
X_test = np.array(X_test)
exp_lime = explainer_lime.explain_instance(
    X_test[i], reg.predict, num_features=k)
 
# Finally visualizing the explanations
exp_lime.show_in_notebook()