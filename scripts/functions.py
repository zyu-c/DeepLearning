import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    a = np.max(x)
    exp_x = np.exp(x - a)
    return exp_x / np.sum(exp_x)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, frags = ['multi_index'])
    while not it.finished:
        tmp_val = x[it.index]
        x[it.index] = tmp_val + h
        fxh1 = f(x)

        x[it.index] = tmp_val - h
        fxh2 = f(x)

        x[it.index] = tmp_val
        grad[it.index] = (fxh1 - fxh2) / (2 * h)
        it.iternext()

    return grad        
