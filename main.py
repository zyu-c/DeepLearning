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
