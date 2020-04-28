import sys
sys.path.append("scripts")
from read_mnist import *
from functions import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.a = 0
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
        self.a = self.a + 1
        print(self.a)
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, t)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        return np.sum(y == t) / float(x.shape[0])

    def gradient(self, x, t):
        grads = {}
        grads['W1'] = self.numerical_gradient(self.loss, self.params['W1'], x, t)
        #grads['b1'] = self.numerical_gradient(self.loss, self.params['b1'], x, t)
        #grads['W2'] = self.numerical_gradient(self.loss, self.params['W2'], x, t)
        #grads['b2'] = self.numerical_gradient(self.loss, self.params['b2'], x, t)

        return grads

    def numerical_gradient(self, f, a, x, t):
        h = 1e-4
        grad = np.zeros_like(a)

        it = np.nditer(a, flags = ['multi_index'])
        while not it.finished:
            tmp_val = a[it.multi_index]
            a[it.multi_index] = tmp_val + h
            fxh1 = f(x, t)

            a[it.multi_index] = tmp_val - h
            fxh2 = f(x, t)

            a[it.multi_index] = tmp_val
            grad[it.multi_index] = (fxh1 - fxh2) / (2 * h)
            it.iternext()
        return grad

net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)
grads = net.gradient(x, t)
print(grads['W1'].shape)