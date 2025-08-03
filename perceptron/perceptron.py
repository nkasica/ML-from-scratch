import numpy as np

class Perceptron:

    def __init__(self, d):
        self.weights = np.random.uniform(-1, 1, d)
        self.bias = 0

    def train(self, X: np.ndarray, Y: np.ndarray, iter: int):
        for _ in range(iter):
            for x, y in zip(X, Y): 
                a = x @ self.weights + self.bias

                if y * a <= 0:
                    self.weights += y * x
                    self.bias += y 

    def predict(self, xhat: np.ndarray):
        return np.sign(xhat @ self.weights + self.bias)