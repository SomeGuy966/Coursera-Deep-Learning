"""
NOTE: This code is heavily modified for simplicity
For original source code of the solution, go to the Jupyter notebook contained in this folder
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s
def propagate(w, b, X, Y):
    # Forward propagation
    m = X.shape[1]
    Z = np.dot(w.T, X) + b # w.T is transposed row vector of weights. np.dot(w.T, X) returns row vector with each element [i] as w.T multiplied by column vector of training example X[i]
    A = sigmoid(Z)

    # Backward propagation
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum((A - Y))

    gradients = {"dw": dw,
             "db": db}

    return gradients
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    #w = copy.deepcopy(w)
    #b = copy.deepcopy(b)

    for i in range(num_iterations):
        gradients = propagate(w, b, X, Y)

        # Retrieve derivatives from gradients
        dw = gradients["dw"]
        db = gradients["db"]

        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db


    params = {"w": w,
              "b": b}

    gradients = {"dw": dw,
             "db": db}

    return params, gradients
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0



    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w = np.zeros((X_train.shape[0], 1))
    b = 0.0

    params, gradients = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d




# Main
# Loaing all the prerequisite data
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

"""
Dissects above code
Info written by me:
1. train and test datasets are Numpy Arrays (Four dimensional)
2. .shape returns dimensions. Shape of train and test datasets are: 
    (number of training examples, number of pixels width, number of pixels height, number of color channels) 
    (209, 64, 64, 3)
"""

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

"""
Dissects above code
Info written by me:
Reshapes train and test datasets into 2D arrays
train_set_x_orig.shape[0] = m (m = number of training examples)
-1 = tells the code to "infer" the column size (in this case, it sets the column size as the number of pixels 64*64*3 = 12288)

Reshapes data into (nx, m)
12288 pixels stacked vertically; each example placed side by side
"""

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

"""
Dissects above code
Info written by me:
Divides all values by 255
"""


# Command that runs the main function
logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)



