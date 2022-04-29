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