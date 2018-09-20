import numpy as np
import pandas as pd
from sklearn import datasets

data = pd.read_csv('H:/learning-area/iris-dataset/iris.csv')
data.info()
data.isnull().sum()
data.describe()

from sklearn import model_selection
target = data['Species']
data.drop('Species', axis=1, inplace=True)
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, random_state=0)
x_train.head()

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)

dtree.score(x_test, y_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predictions'])

dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=20)
dt.fit(x_train, y_train)
y_p = dt.predict(x_test)
pd.crosstab(y_test, y_p, rownames=['Actual'], colnames=['Predicted'])

