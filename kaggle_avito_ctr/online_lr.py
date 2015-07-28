import collections
import math

from pybloom import BloomFilter
from scipy.special import expit as sigmoid


class OnlineLogisticRegression(object):

    # Minimal number of times for a feature to occur to be
    # used with a nonzero weight.
    COUNT_THRESHOLD = -1

    def __init__(self):
        self.weights = None
        self._counters = None
        self._clicks = None
        self._not_clicks = None
        self._ad_impression_counts = None
        self._ad_click_counts = None
        self._user_impression_counts = None
        self._user_click_counts = None

    @property
    def weights_flat(self):
        weights = []

        for field, subweights in self.weights.items():
            for index, value in subweights.items():
                weights.append((field, index, value))

        weights = sorted(weights, key=lambda x: abs(x[2]))

        return weights

    def get_weight(self, field, index):
        weight = self.weights[field][index]
        return weight

    def fit(self, data, lambda1=0, lambda2=0):

        self.weights = collections.defaultdict(self._weights_template)
        self._counters = collections.defaultdict(self._counters_template)

        self._clicks = BloomFilter(capacity=5000000)
        self._not_clicks = BloomFilter(capacity=200000000)

        self._ad_impression_counts = collections.defaultdict(int)
        self._ad_click_counts = collections.defaultdict(int)
        self._user_impression_counts = collections.defaultdict(int)
        self._user_click_counts = collections.defaultdict(int)

        for i, (x, y) in enumerate(data, 1):
            self._process_online_features(x, y)

            y_hat = self.predict(x)
            error = y_hat - y

            for (field, index, value) in x:
                alpha = 1 / (10 + math.sqrt(self._counters[field][index]))

                # Logloss gradient.
                grad = error * value

                # w = self.get_weight(field, index)

                # # L1 regularization.
                # if w != 0:
                #     grad += math.copysign(lambda1, w)

                # # L2 regularization.
                # if field != 'intercept':
                #     grad += lambda2 * w

                self.weights[field][index] -= alpha * grad

                self._counters[field][index] += 1

            if i % 100000 == 0:
                print('Processed {} rows'.format(i), end='\r')

        print('Processed {} rows'.format(i))

    def _counters_template(self):
        return collections.defaultdict(int)

    def _weights_template(self):
        return collections.defaultdict(int)

    def _process_online_features(self, x, y):
        for i, (field, index, value) in enumerate(x):
            if field == 'ad_id':
                ad_id_index = i
                ad_id = value
            elif field == 'user_id':
                user_id_index = i
                user_id = value

        x.pop(ad_id_index)
        if user_id_index > ad_id_index:
            user_id_index -= 1
        x.pop(user_id_index)

        if user_id == -1:
            return

        combination = (user_id, ad_id)

        if combination in self._clicks:
            # User clicked the ad.
            x.append(('user_clicked_ad', 0, 1))
        elif combination in self._not_clicks:
            # User was shown but never clicked the ad.
            x.append(('user_not_clicked_ad', 0, 1))

        if y == 1:
            self._clicks.add(combination)
        elif y == 0:
            self._not_clicks.add(combination)

        n_ad_impressions = self._ad_impression_counts[ad_id]
        n_ad_clicks = self._ad_click_counts[ad_id]
        ad_online_ctr = n_ad_clicks / (10 + n_ad_impressions)

        n_user_impressions = self._user_impression_counts[user_id]
        n_user_clicks = self._user_click_counts[user_id]
        user_online_ctr = n_user_clicks / (10 + n_user_impressions)

        x.append(('user_online_ctr', 0, user_online_ctr))
        x.append(('ad_online_ctr', 0, ad_online_ctr))

        self._ad_impression_counts[ad_id] += 1
        self._user_impression_counts[user_id] += 1
        if y == 1:
            self._ad_click_counts[ad_id] += 1
            self._user_click_counts[user_id] += 1

    def predict(self, x):
        z = 0

        for (field, index, value) in x:
            w = self.get_weight(field, index)
            z += w * value

        p = sigmoid(z)

        return p
