# Importação de Bibliotecas 
from arquivo_preprocessado import preprocessing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings ("ignore")

X_train, y_train, X_test, y_test, X_test_final , y_test_final = preprocessing()
      
def modelo (x_train , y_train):
   param_grid = {
    'bootstrap': [True],
    'max_depth': [None],
    'max_features': [1],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'n_estimators': [100],
    'random_state': [42],
   }
   rf = RandomForestClassifier()
   grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                           cv = 10, n_jobs = -1, verbose = 3,scoring='recall',return_train_score=True)
   grid_search.fit(x_train, y_train)
   return grid_search

def metrica(y_test):
    yhat = modelo_1.predict(X_test_final)
    Acuracia = accuracy_score(y_test,yhat)
    Matrix = confusion_matrix(y_test,yhat)
    Report = classification_report(y_test,yhat)
    return Acuracia , Matrix , Report

modelo_1 = modelo (X_train,y_train)

with open('agro_model.pkl', 'wb') as file:
     pickle.dump(modelo_1, file)

Acuracia , Matrix, Report = metrica (y_test_final)
