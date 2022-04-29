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

