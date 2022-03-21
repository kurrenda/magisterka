import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import pandas as pd

data = pd.read_csv('./data/positions/coordinates_2022_03_20_09_22_07_854349.csv', delimiter=';')

X = data.to_numpy()
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1).fit(X)

joblib.dump(clf, './data/models/one_class_model.pkl', compress=9)