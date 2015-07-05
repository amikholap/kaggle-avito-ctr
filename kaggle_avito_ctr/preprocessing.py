import numpy as np

from .globals import DATA


class Preprocessor(object):

    @property
    def agents1(self):
        """
        A list of the first pass agents.
        """
        return [
            self.price_discretizer,
        ]

    @property
    def agents2(self):
        """
        A list of the second pass agents.
        """
        return [
            self.one_hot_encoder,
        ]

    @property
    def agents(self):
        return self.agents1 + self.agents2

    def __init__(self):
        self.price_discretizer = QuantileDiscretizer('price', 20)
        self.one_hot_encoder = IterativeSparseOneHotEncoder(DATA['CATEGORICAL'])

    def fit(self, X_factory):
        """
        Extract feature properties from a dataset.

        Args:
            X_factory: a factory that provides data iterators with items
            formatted as [(field1, value1), (field2, value2), ...] lists.
        """

        fitted_agents = []

        for agents in (self.agents1, self.agents2):
            X = X_factory()

            for agent in agents:
                agent.prepare_fit()

            for i, row in enumerate(X):
                for fitted_agent in fitted_agents:
                    row = fitted_agent.transform(row)

                for agent in agents:
                    agent.fit_row(row)

            for agent in agents:
                agent.finish_fit()

            fitted_agents.extend(agents)

    def transform(self, row):
        for agent in self.agents:
            row = agent.transform(row)
        return row


class PreprocessorAgent(object):
    """Base class for transformers compatible with Preprocessor."""

    def prepare_fit(self):
        pass

    def fit_row(self, row):
        pass

    def finish_fit(self):
        pass

    def fit(self, X):
        self.prepare_fit()
        for row in X:
            self.fit_row(row)
        self.finish_fit()


class IterativeSparseOneHotEncoder(PreprocessorAgent):
    """One-hot encoder that operates on streams of sparse features."""

    def __init__(self, categorical_features):
        """
        Args:
            categorical_features: A list of categorical features field names.
        """
        self.categorical_features = set(categorical_features)
        self._feature_mapping = {}

    def prepare_fit(self):
        self._feature_mapping = {field: {} for field in self.categorical_features}

    def fit_row(self, row):
        """Collect unique feature values."""
        for field, value in row:
            if field in self.categorical_features:
                self._feature_mapping[field][value] = None

    def finish_fit(self):
        """Assign each feature value a position in one-hot encoded vector."""
        for mapping in self._feature_mapping.values():
            for i, value in enumerate(mapping.keys()):
                mapping[value] = i

    def transform(self, row):
        """
        Map [(field, value), ...] -> [(field, index, value), ...].

        Args:
            row: A list of (field, value) pairs.

        Returns:
            A list of (field, index, value) triplets.
            field is a feature name.
            index is feature-local position.
            value is the value of the feature.
        """

        transformed_row = []

        for (field, value) in row:
            if field in self._feature_mapping:
                # Categorical feature.
                index = self._feature_mapping[field].get(value, None)
                if index is None:
                    # Unknown categorical feature value.
                    # Skip it.
                    continue
                triplet = (field, index, 1)
            else:
                # Non-categorical feature.
                # Treat non-categorical features as having 0 index and original value.
                triplet = (field, 0, value)

            transformed_row.append(triplet)

        return transformed_row


class QuantileDiscretizer(PreprocessorAgent):

    def __init__(self, field, q):
        self.field = field
        self.transformed_field = '{}_percentile'.format(self.field)
        self.q = np.linspace(0, 100, q + 1)[1:-1]

    def prepare_fit(self):
        self._index = 0
        self._values = np.empty(shape=1000, dtype=np.float32)

    def fit_row(self, row):
        if self._values.size <= self._index:
            self._values.resize(int(self._values.size * 1.5))

        for (field,  value) in row:
            if field == self.field:
                self._values[self._index] = value
                self._index += 1
                break

    def finish_fit(self):
        self.percentiles = np.percentile(self._values[:self._index], self.q)[1:-1]
        del self._index
        del self._values

    def transform(self, row):
        for i, (field, value) in enumerate(row):
            if field == self.field:
                transformed_value = 0
                for j, p in enumerate(self.percentiles):
                    if value < p:
                        transformed_value = j
                        break
                    else:
                        transformed_value = len(self.percentiles)
                row[i] = (self.transformed_field, transformed_value)
                break
        return row
