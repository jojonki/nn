import numpy as np
import math
import random
from matplotlib import pyplot

class Neural:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.random_sample((hidden_dim, input_dim+1))
        self.W2 = np.random.random_sample((output_dim, hidden_dim+1))
        self.h_momentum = np.zeros((hidden_dim, input_dim+1))
        self.y_momentum = np.zeros((output_dim, hidden_dim+1))

    def train(self, X, T, epsilon, mu, epoch):
        self.error = np.zeros(epoch)
        N = X.shape[0]
        for ep in range(epoch):
            for i in range(N):
                x = X[i, :]
                t = T[i, :]

                self.__update_weight(x, t, epsilon, mu)
            self.error[ep] = self.__calc_error(X, T)

    def predict(self, X):
        N = X.shape[0]
        C = np.zeros(N).astype('int') # class
        Y = np.zeros((N, X.shape[1]))
        for i in range(N):
            x = X[i, :]
            h, y = self.__forward(x)
            Y[i] = y
            C[i] = y.argmax()
        return (C, Y)

    def plot_error(self):
        pyplot.ylim(0.0, 2.0)
        pyplot.plot(range(0, self.error.shape[0]), self.error)        
        pyplot.show()
    
    def __sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __forward(self, x):
        input =  np.r_[np.array([1]), x]
        h = self.__sigmoid(np.dot(self.W1, input))
        h_dash = np.r_[np.array([1]), h]
        y = self.__sigmoid(np.dot(self.W2, h_dash))
        return (h, y)

    def __update_weight(self, x, t, epsilon, mu):
        h, y = self.__forward(x)

        # update W2
        output_delta = (y - t) * y * (1.0 - y) 
        _W2 = self.W2
        self.W2 -= epsilon * output_delta.reshape((-1, 1)) * np.r_[np.array([1]), h] - mu * self.y_momentum
        self.y_momentum = self.W2 - _W2 # 重みの差分がmomentum

        # update W1
        hidden_delta = (self.W2[:, 1:].T.dot(output_delta)) * h * (1.0 - h)
        _W1 = self.W1
        self.W1 -= epsilon * hidden_delta.reshape((-1, 1)) * np.r_[np.array([1]), x]
        self.h_momentum = self.W1 - _W1

    def __calc_error(self, X, T):
        N = X.shape[0]
        err = 0.0
        for i in range(N):
            x = X[i, :]
            t = T[i, :]
            h, y = self.__forward(x)
            err += (y-t).dot((y-t).reshape((-1, 1))) / 2.0
        return err
