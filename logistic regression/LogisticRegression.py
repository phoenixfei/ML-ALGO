import numpy as np
#定义sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
#对sigmoid函数求导
def derivate_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
