import pandas as pd 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = np.linspace(-1, 1, 60)
y = X ** 3 + 1
    
"""
Q1 a) Plot function
"""
def function():
    
    print(y)
    
    plt.scatter(X, y, color='red')
    plt.xlabel('x')
    
    plt.show()
    
'''
Q1 b) Take a training set and introduce noise
'''
def noise_train_set():
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
    random_noise_x = (np.random.random_sample((30,)) - np.random.random_sample((30,))) / 2
    # print(X_train)
    X_train = X_train + random_noise_x
    
    plt.scatter(X_train, y_train, color='red')
    plt.scatter(X_test, y_test, color='blue')
    plt.show()

'''
Q1 c) perform a linear regression on the dataset, observe the SSE
'''
def lin_regr():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
    random_noise_x = (np.random.normal(size=30))
    X_train = X_train + random_noise_x
    
    # print(type(X_train))
    
    reg = LinearRegression().fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    
    y_predict = reg.predict(X_test.reshape(-1, 1))
    
    #Determine the sum of squared error between predicted and test
    print(f"SSE: {(y_predict - y_test)**2}")
    
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_test, y_predict, color='blue')
    
    plt.show()

def polynomial_regr():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
    random_noise_x = (np.random.normal(size=30))
    X_train = X_train + random_noise_x
    X_train = pd.DataFrame(data=X_train) 
    X_train['x2'] = (X_train[0]) ** 2
    
    print(X_train.head())
    
    reg = LinearRegression().fit((X_train), y_train)
    
    y_predict = reg.predict(X_test.reshape(-1, 1))
    
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_test, y_predict, color='blue')
    
    plt.show()
    
    
polynomial_regr()