import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest


data = pd.read_csv(r'C:\Users\Rafal\Documents\adv_pyth\magisterka\data\positions\train_2022_06_01_12_58_09_027892.csv', delimiter=';')

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

neigh = KNeighborsClassifier(n_neighbors=3)
clf = neigh.fit(X.values, y.values)

joblib.dump(clf, './data/models/one_class_model.pkl', compress=9)