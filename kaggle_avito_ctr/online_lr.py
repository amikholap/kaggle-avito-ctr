import collections

from scipy.special import expit as sigmoid


class OnlineLogisticRegression(object):

    def __init__(self):
        self.weights = {}

    @property
    def weights_flat(self):
        weights = []

        for field, subweights in self.weights.items():
            for index, value in subweights.items():
                weights.append((field, index, value))

        weights = sorted(weights, key=lambda x: abs(x[2]))

        return weights

    def fit(self, data, alpha=0.01):

        self.weights = collections.defaultdict(self._weights_template)

        for i, (x, y) in enumerate(data, 1):
            y_hat = self.predict(x)
            error = y_hat - y

            for (field, index, value) in x:
                self.weights[field][index] -= alpha * error * value

            if i % 100000 == 0:
                print('Processed {} rows'.format(i), end='\r')

        print('Processed {} rows'.format(i))

    def _weights_template(self):
        return collections.defaultdict(int)

    def predict(self, x):
        z = 0

        for (field, index, value) in x:
            z += self.weights[field][index] * value

        p = sigmoid(z)

        return p
