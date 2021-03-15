# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # Sigmoid Function
# def sigmoid(x):
#     return 1/(1+np.exp(x))

# # Derivative Sigmoid Function
# def sigmoid_der(x):
#     return x*(1-x)

# # Class Neural Network
# class NeuralNetwork:
#     def __init__(self, layers, alpha=0.1):
#         self.layers = layers
#         # learning rate
#         self.alpha = alpha
#         # Weight and bias
#         self.W = []
#         self.b = []
#         # Initial paramaters in layer
#         for i in range(0, len(layers)-1):
#             w_ = 2*np.random.randn(layers[i], layers[i+1]) - 1
#             b_ = np.zeros((layers[i+1], 1))
#             self.W.append(w_)
#             self.b.append(b_)
    
#     # Train model with data
#     def fit_partial(self, x, y):
#         A = [x]
#         # Feedforward
#         out = A[-1]
#         for i in range(0, len(self.layers)-1):
#             out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
#             A.append(out)
#         # Backpropagation
#         y = y.reshape(-1, 1)
#         dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
#         dW = []
#         db = []
#         for i in reversed(range(0, len(self.layers)-1)):
#             dw_ = np.dot((A[i]).T, dA[-1]*sigmoid_der(A[i+1]))
#             db_ = (np.sum(dA[-1]*sigmoid_der(A[i+1]), 0)).reshape(-1, 1)
#             dA_ = np.dot(dA[-1]*sigmoid_der(A[i+1]), self.W[i].T)
#             dW.append(dw_)
#             db.append(db_)
#             dA.append(dA_)

#         # Reverse dW, db
#         dW = dW[::-1]
#         db = db[::-1]

#         # Gradient descent
#         for i in range(0, len(self.layers)-1):
#             self.W[i] = self.W[i] - self.alpha * dW[i]
#             self.b[i] = self.b[i] - self.alpha * db[i]

#     def fit(self, X, y, epochs=20, verbose=10):
#         for epoch in range(0, epochs):
#             self.fit_partial(X, y)
#             if epoch%verbose == 0:
#                 loss = self.calculate_loss(X, y)
#                 print("Epoch {}, loss {}".format(epoch, loss))
#                 accurary = self.cal_accuracy(X, y)
#                 print("Accuracy: {}".format(accurary))
#         return self.W, self.b
    
#     # Predict
#     def predict(self, X):
#         for i in range(0, len(self.layers) - 1):
#             X = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
#         return X
    
#     # Calculate Loss Function
#     def calculate_loss(self, X, y):
#         y_predict = self.predict(X)
#         m = y.shape[0]
#         return 1/m*(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict)))
    
#     # Calculate Accuracy
#     def cal_accuracy(self, X, y):
#         m = y.shape[0]
#         pred = self.predict(X)
#         pred = pred.reshape(y.shape)
#         error = np.sum(np.abs(pred - y))
#         return (m - error)/m * 100

# data = pd.read_csv('D:\\Machine Learning\\Task5\\data\\data.csv')
# # N, d = data.shape
# # X = data[:, 0:d-1].reshape(-1, d-1)
# # y = data[:, -1].reshape(-1, 1)
# length = len(data)
# dataFrame = data.to_numpy()
# np.random.shuffle(dataFrame)
# data_train = dataFrame[:int(length*0.7)]
# data_test = dataFrame[int(length*0.7):length]
# X_train = data_train[:,:-1]
# Y_train = data_train[:, -1].reshape(-1, 1)
# X_test = data_test[:,:-1]
# Y_test = data_test[:, -1].reshape(-1, 1)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# p = NeuralNetwork([X_train.shape[1], 5, 1], 0.01)
# W = []
# b = []
# W, b = p.fit(X_train, Y_train, 1000, 10)
# test_loss = p.calculate_loss(X_test, Y_test)
# test_acc = p.cal_accuracy(X_test, Y_test)
# print(test_acc, test_loss)



# 1/03/2021

import numpy as np
# from scipy.special import expit

# sigmoid function
def sigmoid():
    return lambda X: 1 / (1+np.exp(-X))

# Sigmoid function derivative
def sigmoid_der():
    return lambda X: sigmoid()(X) * (1 - sigmoid()(X))

# reLu function
def relu():
    return lambda X: np.where(X>0, X, 0)

# reLu function derivative
def relu_der():
    def _(X):
        X[X<=0] = 0
        X[X>0] = 1
        return X
    return _

# softmax fuction
def softmax():
    def _(X):
        exps = np.exp(X)
        summ = np.sum(X, axis=0)
        return np.divide(exps, summ)
    return _

# softmax derivative
# def softmax_der():
#       pass

def no_func():
    return lambda X: X

def no_func_der():
    return lambda X : 1

def get_activation(activation):
    activation = activation.lower()
    if activation == 'sigmoid':
        return sigmoid(), sigmoid_der()
    elif activation == 'relu':
        return relu(), relu_der()
    elif activation == 'no_func':
        return no_func(), no_func_der()
    # default
    return no_func(), no_func_der()


# Class Layerrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
import math

"""
    Layer: The hidden layer of neural network
    .....

    Attributes
    ----------
        shape: type -> INT
            is the number of neurons in this layer
        activation: type -> STRING 
            is the activation function of this layer
            default -> 'sigmoid'
"""

class Layer:
    def __init__(self, shape, activation='sigmoid'):
        self._act_funcion, self._act_function_der = get_activation(activation)
        self.shape = (shape,)

        # setup the hidden layer
        # config shape, weights, biases and initialize them
        def _setup(self, pre_layer):
            self.shape = (pre_layer.shape[0], ) + self.shape
            self.weight = np.random.randn(pre_layer.shape[1], self.shape[1]) / self._get_spec_number(pre_layer)
            self.bias = np.random.randn(1, self.shape[1]) / self._get_spec_number(pre_layer)
            self.values = np.zeros(self.shape)
        
        def _get_spec_number(self, pre_layer):
            return self.shape[1] * pre_layer.shape[1]

        def _forward(self, pre_layer):
            if isinstance(pre_layer, np.ndarray): #first hidden layer
                self.z = np.dot(pre_layer, self.weight) + self.bias
            else:
                self.z = np.dot(pre_layer.values, self.weight) + self.bias
            self.values = self._act_fuction(self.z)

        def _backward(self, delta, pre_layer, learning_rate):
            delta = delta * self._act_function_der(self.z)
            delta_bias = (1/ self.shape[0]) * (np.sum(delta, axis=0).reshape(1, -1))
            # No regularization
            if isinstance(pre_layer, np.ndarray):
                weight_der = (1/self.shape[0]) * (np.dot(pre_layer.T, delta))
            else:
                weight_der = (1/self.shape[0])*(np.dot(pre_layer.values.T, delta))
            # Adding regularization
            regular_param = 0.1
            if isinstance(pre_layer, np.ndarray):
                weight_der = (1/self.shape[0]) * (np.dot(pre_layer.T, delta)) + (regular_param/self.shape[0])*self.weight
            else:
                weight_der = (1/self.shape[0]) * (np.dot(pre_layer.values.T, delta)) + (regular_param/self.shape[0])*self.weight

            self.bias -= learning_rate * delta_bias
            delta = np.dot(delta, self.weight.T)
            self.weight -= learning_rate + weight_der
            return delta

        # backward_Adam
        def _backwardAdam(self, delta, pre_layer, learning_rate, VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, iter_num):
            delta = delta * self._act_function_der(self.z)
            delta_bias = (1/ self.shape[0]) * (np.sum(delta, axis=0).reshape(1, -1))
            # No regularization
            if isinstance(pre_layer, np.ndarray):
                weight_der = (1/self.shape[0]) * (np.dot(pre_layer.T, delta))
            else:
                weight_der = (1/self.shape[0])*(np.dot(pre_layer.values.T, delta))
            
            VdW = beta1*VdW + (1-beta1) * weight_der
            Vdb = beta1*Vdb + (1-beta1) * delta_bias
            SdW = beta2*SdW + (1 - beta2) * np.power(weight_der, 2)
            Sdb = beta2*Sdb + (1 - beta2) * np.power(delta_bias, 2)
            VdW_corrected = VdW / (1 - math.pow(beta1, iter_num + 1))
            Vdb_corrected = Vdb / (1 - math.pow(beta1, iter_num + 1))
            SdW_corrected = SdW / (1 - math.pow(beta2, iter_num + 1))
            Sdb_corrected = Sdb / (1 - math.pow(beta2, iter_num + 1))

            self.bias -= learning_rate * \
                (Vdb_corrected/(np.sqrt(Sdb_corrected) + epsilon))
            delta = np.dot(delta, self.weight.T)
            self.weight -= learning_rate * \
                (VdW_corrected/(np.sqrt(SdW_corrected) + epsilon))

            return delta

# Class Neural Networkkkkkkkkkkkkkkkkkkkkkkkkkkkk
"""
    NN: is a simple neural network model for classification & regression problems
    ......

    Attributes
    ----------
        X: type -> ndarray
            the input data
        Y: type -> ndarray
            the target data
        output_activation: type -> string
            The activation function of the last layer, the output layer
            default -> 'sigmoid'
"""

class NN:
    def __init__(self, X, Y, output_activation='sigmoid'):
        self._X = X
        self._Y = Y
        self._layers = []
        self._output_activation = output_activation

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise Exception("Invalid Type", type(layer), " != <class 'Layer' >")
        self._layers.append(layer)

    # Train data
    def fit(self, learning_rate=0.01, iteration=1000):
        self._setup()
        self._costs = []
        self._learning_rate = learning_rate
        self._iteration = iteration
        # Adam parameters
        # VdW = 0
        # Vdb = 0
        # SdW = 0
        # Sdb = 0
        # beta1 = 0.9
        # beta2 = 0.999
        # epsilon = math.pow(10, -8)
        for i in range(iteration):
            self._forwardPropagation()
            self._backPropagation()

            # Adam Optimizer
            # self._forwardPropagation()
            # self._backPropagationAdam(VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, i)
            print(self._calc_cost(self.layers[len(self._layers)-1].values))
            if (i%100==0):
                self._costs.append(self._calc_cost(self._layers[len(self._layers)-1].values))
        # Save weights and bias
        path_weights = "Pre_para/weight"
        path_biases = "Pre_para/bias"
        for j in range(len(self._layers)):
            np.savetxt(path_weights + str(j) + ".csv",
                       self._layers[j].weight, delimiter=",")
            np.savetxt(path_biases + str(j) + ".csv",
                       self._layers[j].bias, delimiter=",")
    # Return the cost function
    def _calc_cost(self, Y_pred):
        # return np.sum(np.square(self._Y - Y_pred)/2)
        # No regularization
        return (-1/self._X.shape[0])*(np.sum(self._Y.T @ (np.log(Y_pred)) + (1 - self._Y).T @ (np.log(1 - Y_pred))))
        # Add regularization
        regular_part = 0
        for layer in self._layers:
            regular_part += np.sum(np.power(layer.weight, 2))
        regular_para = 0.1
        regular_part = (regular_para/(2*self._X.shape[0])) * regular_para
        return (-1/self._X.shape[0])*(np.sum(self._Y.T@ (np.log(Y_pred)) + (1 - self._Y).T @ (np.log(1 - Y_pred)))) + regular_part
    
    # configuration the shape,
    # weight and bias of each layer
    # add output layer
    def _setup(self):
        for index, layer in enumerate(self._layers):
            if (index == 0): # First hidden layer
                layer._setup(self._X)
            else:
                layer._setup(self._layers[index-1])
        # setup and add output layer
        output_layer = Layer(self._Y.shape[1], activation=self._output_activation)
        output_layer._setup(self._layers[len(self._layers) - 1])
        self.add_layer(output_layer)
    
    # Forward Propagation
    def _forwardPropagation(self):
        for index, layer in enumerate(self._layers):
            if (index == 0): # the first hidden layer
                layer._forward(self._X)
            else:
                layer._forward(self._layers[index - 1])
    
    # BackPropagation
    def _backPropagation(self):
        # delta = self._Y - self._layers[len(self._layers)-1].values
        delta = (- self._Y / self._layers[len(self._layers)-1].values) + ((1 - self._Y) / (1 - self._layers[len(self._layers)-1].values))
        for i in range(len(self._layers)-1, -1, -1):
            if (i == 0): # first hidden layer
                delta = self._layers[i]._backward(delta, self._X, self._learning_rate)
            else:
                delta = self._layers[i]._backward(delta, self._layers[i-1], self._learning_rate)

    # BACKPROPAGATION ADAM
    def _backPropagationAdam(self, VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, iter_num):
        delta = (-self._Y / self._layers[len(self._layers) - 1].values) + (
            (1-self._Y) / (1 - self._layers[len(self._layers) - 1].values))
        for i in range(len(self._layers)-1, -1, -1):
            if (i == 0):  # first hidden layer
                delta = self._layers[i]._backward_Adam(
                    delta, self._X, self._learning_rate, VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, iter_num)
            else:
                delta = self._layers[i]._backward_Adam(
                    delta, self._layers[i-1], self._learning_rate, VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, iter_num)

    def predict(self, X_test):
        for index, layer in enumerate(self._layers):
            if(index == 0):
                layer._foward(X_test)
            else:
                layer._foward(self._layers[index-1])
        if self._is_continues():  # if target labels is continues
            return self._layers[len(self._layers)-1].values
        if self._is_multiclass():  # if target labels is multiclass
            return self._threshold_multiclass(self._layers[len(self._layers)-1])
        # binary classification
        return self._threshold(self._layers[len(self._layers)-1], 0.5)

    # set the 'predict.value' > 'value' [treshhold] to '1' others to '0'

    def _threshold(self, target, value):
        predict = target.values
        predict[predict < value] = 0
        predict[predict >= value] = 1
        return predict

    # set the max 'predict.value' to '1' others to '0'
    def _threshold_multiclass(self, target):
        predict = target.values
        predict = np.where(predict == np.max(
            predict, keepdims=True, axis=1), 1, 0)
        # predict[] = 1 | 0
        return predict

    # check if it's a multiclassfication problem
    def _is_multiclass(self):
        return len(np.unique(self._Y)) > 2

    # check if it's a regression problem
    def _is_continues(self):
        return len(np.unique(self._Y)) > (self._Y.shape[0] / 3)

    # setup pretrained weights
    def _setup_pretrained_weights(self):
        path_weights = "Pre_para/weight"
        path_biases = "Pre_para/bias"
        for index in range(len(self._layers)):
            temp1 = np.genfromtxt(path_biases + str(index) + ".csv", delimiter=",")
            self._layers[index].bias = np.array([temp1]).reshape(1, self._layers[index].shape[1])
            # output layer
            if index == 0:
                if (self._X.shape[1] == 1 or self._layers[0].shape[1] == 1):
                    temp = np.genfromtxt(
                        path_weights + str(index) + ".csv", delimiter=",")
                    self._layers[index].weight = np.array([temp]).reshape(
                        self._layers[index - 1].shape[1], self._layers[index].shape[1])
                else:
                    self._layers[index].weight = np.genfromtxt(
                        path_weights + str(index) + ".csv", delimiter=",")
            else:
                if ((self._layers[index - 1].shape[1] == 1) or (self._layers[index].shape[1] == 1)):
                    temp = np.genfromtxt(
                        path_weights + str(index) + ".csv", delimiter=",")
                    self._layers[index].weight = np.array([temp]).reshape(
                        self._layers[index - 1].shape[1], self._layers[index].shape[1])
                else:
                    self._layers[index].weight = np.genfromtxt(
                        path_weights + str(index) + ".csv", delimiter=",")

    # predict with pretrained weights

    def predict_pretrained_weights(self, X_test):
        self._setup()
        self._setup_pretrained_weights()
        return self.predict(X_test)