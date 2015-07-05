import collections
import copy

from scipy.special import expit as sigmoid


class OnlineLogisticRegression(object):

    def __init__(self):
        self.weights = {}

    def fit(self, data, alpha=0.01):

        field_weights_template = collections.defaultdict(lambda: 0)
        self.weights = collections.defaultdict(lambda: copy.copy(field_weights_template))
        self.intercept_weight = 0

        for i, (x, y) in enumerate(data):
            y_hat = self.predict(x)
            error = y_hat - y

            for (field, index, value) in x:
                self.weights[field][index] -= alpha * error * value

            if i % 100000 == 0:
                print('Processed {} rows'.format(i))

    def predict(self, x):
        z = 0

        for (field, index, value) in x:
            z += self.weights[field][index] * value

        p = sigmoid(z)

        return p
