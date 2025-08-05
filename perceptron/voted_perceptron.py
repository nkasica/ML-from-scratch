import numpy as np

class VotedPerceptron:

    def __init__(self, d):
        self.weights = np.random.uniform(-1, 1, d)
        self.bias = 0
        self.hyperplanes = [] # ([w1, ..., wd], b, lifetime)

    def train(self, X: np.ndarray, Y: np.ndarray, iter: int):
        lifetime = 0

        for _ in range(iter):
            for x, y in zip(X, Y): 
                a = x @ self.weights + self.bias

                if y * a <= 0:
                    # store weights and lifetime
                    self.hyperplanes.append([self.weights.copy(), self.bias, lifetime])
                    lifetime = 0
                    self.weights += y * x
                    self.bias += y 
                else:
                    lifetime += 1

        self.hyperplanes.append([self.weights.copy(), self.bias, lifetime])

    def predict(self, xhat: np.ndarray):
        if not self.hyperplanes:
            raise ValueError("Model must be trained before you can make predictions")

        weights = np.array([h[0] for h in self.hyperplanes])
        biases = np.array([h[1] for h in self.hyperplanes])
        lifetimes = np.array([h[2] for h in self.hyperplanes]) 

        return np.sign(np.sum(lifetimes * np.sign(xhat @ weights.T + biases)))



                       
                       
