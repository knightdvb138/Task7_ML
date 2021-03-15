import numpy as np
from numpy import genfromtxt
from sys import path
path.append('..')
import pandas as pd

path = 'D:/Machine Learning/Task7/Pre_para/'

def get_para(i):
    bias = genfromtxt(path + 'bias' + str(i) + '.csv', delimiter = ',')
    weight = genfromtxt(path + 'weight' + str(i) + '.csv', delimiter = ',')
    return (weight, bias)
(w0, b0) = get_para(0)
(w1, b1) = get_para(1)
(w2, b2) = get_para(2)

def sigmoid(x):
    return 1/(1-np.exp(x))
def relu_func(X):
    return np.where(X>=0, X, 0)
def predict_img(img):
    X = img.reshape(1, -1)
    z0 = np.dot(X, w0) + b0
    a1 = relu_func(z0)
    z2 = np.dot(a1, w1) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w2) + b2
    y_pred = sigmoid(z3)
    if y_pred >= 0.5:
        return 1
    else: return 0

