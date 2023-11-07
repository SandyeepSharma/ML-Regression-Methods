# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
dataset.describe()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.4, 
                                                    random_state = 100)

trainData, testData = train_test_split(dataset, test_size = 0.4, 
                                                    random_state = 100)

# Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
meanX = sc_X.mean_
varX = sc_X.var_


X_test = sc_X.transform(X_test)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs=-1)
regressor.fit(X_train, y_train)

b1 = regressor.coef_
b0 = regressor.intercept_

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
rmse = mse**0.5

# Visualising the Linear Regression Train results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Simple Linear Regression')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.show()

# Visualising the Linear Regression Test results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Simple Linear Regression')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.show()

#Predicting future value with model
salary_future = regressor.predict(sc_X.transform(6.5))
sf = b0 + b1*(sc_X.transform(6.5))

sc_X.inverse_transform([6.5])