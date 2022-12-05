import numpy as np
import pandas as pd

from google.colab import files
upload = files.upload()

data = pd.read_csv('iris.csv')
data

data.info()

## Train Test Split

from sklearn.model_selection import train_test_split
## train - 70%
## test - 30%

# input data
X = data.drop(columns=['variety'])
# output data
Y = data['variety']
# split the data for train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

## Model Training

# logistic regression 
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression()

# model training
train_model_log = model_log.fit(x_train, y_train)

# print  to get performance
print("Accuracy: ",model_log.score(x_test, y_test) * 100)

from sklearn import model_selection
# knn - k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier()

train_model_knn = model_knn.fit(x_train, y_train)

# print to get performance
print("Accuracy: ",model_knn.score(x_test, y_test) * 100)

# decision tree
from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()
model_tree.fit(x_train, y_train)
# print to get performance
print("Accuracy: ",model_tree.score(x_test, y_test) * 100)
