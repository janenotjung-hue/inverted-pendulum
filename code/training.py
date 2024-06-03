from setup import get_datasets, get_results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from joblib import dump, load

X_trains, X_tests, y_trains, y_tests = get_datasets(None)

"""reg = ExtraTreesRegressor()
for i in range(len(X_trains)):  
    reg.fit(X_trains[i], y_trains[i])
dump(reg, 'code/xai_models/reg_model.joblib')

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500, verbose=True)
for i in range(len(X_trains)):  
    mlp.fit(X_trains[i], y_trains[i])
dump(mlp, 'code/xai_models/mlp_model.joblib')"""

classifier = ExtraTreesClassifier()
for i in range(len(X_trains)):  
    classifier.fit(X_trains[i], y_trains[i])
dump(classifier, 'xai_models/tree_class_model.joblib')
