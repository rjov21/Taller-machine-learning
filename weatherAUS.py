# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:11:30 2022

@author: ABC
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

url = 'weatherAUS.csv'
data = pd.read_csv(url)

#tratamiento de data

data.drop(['Date', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
           'WindGustSpeed', 'Pressure9am', 'Humidity9am', 'Humidity3pm', 
           'Pressure3pm', 'Temp9am', 'Temp3pm'], axis= 1, inplace = True)
data.drop(['Date'], axis=1, inplace=True)
data['Location'].replace(['Canberra', 'Sydney', 'Perth', 'Darwin', 'Hobart',
                          'Brisbane', 'Adelaide', 'Bendigo', 'Townsville', 'AliceSprings', 
                          'MountGambier', 'Ballarat', 'Launceston', 'Albany', 'Albury',
                          'MelbourneAirport', 'PerthAirport', 'Mildura', 'SydneyAirport', 
                          'Nuriootpa', 'Sale', 'Watsonia', 'Tuggeranong', 'Portland',
                          'Woomera', 'Cobar', 'Cairns', 'Wollongong', 'GoldCoast',
                          'WaggaWagga', 'NorfolkIsland', 'Penrith', 'SalmonGums', 'Newcastle',
                          'CoffsHarbour', 'Witchcliffe', 'Richmond', 'Dartmoor', 
                          'NorahHead', 'BadgerysCreek', 'MountGinini', 'Moree', 'Walpole',
                          'PearceRAAF', 'Williamtown', 'Melbourne', 'Nhil', 'Katherine',
                          'Uluru'],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                          19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                          35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], inplace=True)


#partir la data en dos

data_train = data[:206]
data_test = data[206:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)



# Regresión Logística

# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)


# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')