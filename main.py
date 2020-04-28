import sys
sys.path.append("scripts")
from read_mnist import *
from functions import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        return

    def predict(self, x):
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, t)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        grads = {}
        grads['W1'] = numerical_gradient(self.predict(x), self.params['W1'])
        grads['b1'] = numerical_gradient(self.predict(x), self.params['b1'])
        grads['W2'] = numerical_gradient(self.predict(x), self.params['W2'])
        grads['b2'] = numerical_gradient(self.predict(x), self.params['b2'])

        return grads
