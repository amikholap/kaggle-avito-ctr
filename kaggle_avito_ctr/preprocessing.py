import logging
import re
import string

import numpy as np
import pymorphy2

from .globals import DATA, session
from .models import Category


_logger = logging.getLogger(__name__)


class Preprocessor(object):

    @property
    def agents1(self):
        """
        A list of the first pass agents.
        """
        return [
            self.category_feature_extractor,
            self.price_discretizer,
            # self.text_feature_extractor,
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
        self.category_feature_extractor = CategoryFeatureExtractor()
        self.price_discretizer = QuantileDiscretizer('price', 20)
        self.text_feature_extractor = TextFeatureExtractor()

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

        msg = '{} summary:\n'.format(self.__class__.__name__)

        for field, mapping in sorted(self._feature_mapping.items(),
                                     key=lambda pair: len(pair[1])):
            for i, value in enumerate(mapping.keys()):
                mapping[value] = i

            msg += '{:25} | {} distinct values\n'.format(field, len(mapping))

        _logger.info(msg)

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


class CategoryFeatureExtractor(PreprocessorAgent):

    def __init__(self):
        self.categories = {c.category_id: c for c in session.query(Category)}

    def transform(self, row):
        for (field, value) in row:
            if field == 'search_cat_id':
                search_cat_id = value
            elif field == 'ad_cat_id':
                ad_cat_id = value

        search_cat = self.categories[search_cat_id]
        ad_cat = self.categories[ad_cat_id]

        if search_cat.category_id == ad_cat.category_id:
            row.append(('search_ad_same_cat', 1))
        if search_cat.parent_category_id == ad_cat.parent_category_id:
            row.append(('search_ad_same_parent_cat', 1))

        return row


class TextFeatureExtractor(PreprocessorAgent):

    analyzer = pymorphy2.MorphAnalyzer()
    number_pattern = re.compile('\d+')
    punctuation_pattern = re.compile(r'[{}]'.format(string.punctuation))

    def transform(self, row):
        search_query_idx = ad_title_idx = None
        search_query = ad_title = None

        for i, (field, value) in enumerate(row):
            if field == 'search_query':
                search_query_idx = i
            elif field == 'ad_title':
                ad_title_idx = i

            if search_query_idx is not None and ad_title_idx is not None:
                break

        ad_title = row.pop(ad_title_idx)[1]

        if search_query_idx is None:
            # Don't do anything if search query is missing.
            return row

        if ad_title_idx < search_query_idx:
            # Compensate for the pop.
            search_query_idx -= 1
        search_query = row.pop(search_query_idx)[1]

        search_query_tokens = self._tokenize(search_query)
        ad_title_tokens = self._tokenize(ad_title)

        search_query = ''.join(search_query_tokens)
        ad_title = ''.join(ad_title_tokens)

        row.extend([
            ('query_common_tokens', self._get_common_tokens_percentage(search_query_tokens, ad_title_tokens)),
            ('query_common_numbers', self._get_common_numbers_percentage(search_query, ad_title)),
            ('query_lcs', self._get_lcs_len_percentage(search_query, ad_title)),
        ])

        return row

    def _tokenize(self, s):
        s = self._remove_punctuation(s)
        s = self._normalize(s)
        return s

    def _remove_punctuation(self, s):
        return re.sub(self.punctuation_pattern, '', s)

    def _normalize(self, s):
        """Make lowercase and convert all words to their normal form."""
        return [self.analyzer.normal_forms(w)[0] for w in s.split()]

    def _get_common_tokens_percentage(self, tokens1, tokens2):
        """Calculate a proportion of common tokens with respect to the first argument."""
        s1 = set(tokens1)
        s2 = set(tokens2)
        p = len(s1 & s2) / len(s1) if s1 else 0
        return p

    def _get_lcs_len_percentage(self, s1, s2):
        """Calculate a proportion of a longest common substring length to the first string length."""
        lcs = pymorphy2.utils.longest_common_substring([s1, s2])
        return len(lcs) / len(s1) if s1 else 0

    def _get_common_numbers_percentage(self, s1, s2):
        """Calculate a proportion of common numbers to a total count of numbers in the first string."""
        ns1 = set(re.findall(self.number_pattern, s1))
        ns2 = set(re.findall(self.number_pattern, s2))
        p = len(ns1 & ns2) / len(ns1) if ns1 else 0
        return p
