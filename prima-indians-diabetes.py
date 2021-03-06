# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 23:09:34 2022

@author: ABC
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

url = 'diabetes.csv'
data = pd.read_csv(url)


#tratamiento de la data

data.Age.replace(np.nan, 33, inplace=True) # promedio de 33 de la columna edad 
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangos, labels=nombres)
glucose_rangos = [0, 25, 50, 75, 100, 125, 150, 175, 200]
glucose_nombres = ['1', '2', '3', '4', '5', '6', '7', '8']
data.Glucose = pd.cut(data.Glucose, glucose_rangos, labels=glucose_nombres)
insulin_rangos = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
insulin_nombres = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
data.Insulin = pd.cut(data.Insulin, insulin_rangos, labels=insulin_nombres)
diabetes_rangos = [0, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5]
diabetes_nombres = ['1', '2', '3', '4', '5', '6', '7', '8']
data.DiabetesPedigreeFunction = pd.cut(data.DiabetesPedigreeFunction, diabetes_rangos, labels=diabetes_nombres)
data.dropna(axis=0,how='any', inplace=True)


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
