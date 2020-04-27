import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    a = np.max(x)
    exp_x = np.exp(x - a)
    return exp_x / np.sum(exp_x)