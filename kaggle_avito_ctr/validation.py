import logging
import math

from .extraction import SparseDataset


_logger = logging.getLogger(__name__)


def evaluate(model, data):
    score = logloss(model, data)
    return score


def cv(clf, filename, n_folds=5, num_samples=None):
    scores = []

    if num_samples is not None:
        test_limit = num_samples / n_folds
        train_limit = num_samples - test_limit
    else:
        train_limit = test_limit = None

    with SparseDataset(filename) as train_ds:
        with SparseDataset(filename) as test_ds:
            for part in range(n_folds):
                train_iterator = train_ds.iterator(offset=part, limit=train_limit, skip_nth=n_folds)
                test_iterator = test_ds.iterator(offset=part, limit=test_limit, every_nth=n_folds)

                clf.fit(train_iterator)
                score = logloss(clf, test_iterator)

                scores.append(score)

                _logger.info('CV {}/{} score: {}'.format(part, n_folds, score))

    return scores


def validation_curve(clf, filename, train_sizes, test_proporion=0.2):
    scores = []
    nth = int(1 / test_proporion)

    with SparseDataset(filename) as train_ds:
        with SparseDataset(filename) as test_ds:
            for limit in train_sizes:
                train_iterator = train_ds.iterator(limit=limit, skip_nth=nth)
                clf.fit(train_iterator)

                train_iterator = train_ds.iterator(limit=limit, skip_nth=nth)
                test_iterator = test_ds.iterator(every_nth=nth)
                train_score = logloss(clf, train_iterator)
                test_score = logloss(clf, test_iterator)

                scores.append((train_score, test_score))

                _logger.info('{} samples train/test scores: {:.5}/{:.5}'
                             .format(limit, train_score, test_score))

    return scores


def sample_logloss(y_hat, y):
    margin = 1e-9
    y_hat = max(min(y_hat, 1 - margin), margin)
    return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))


def logloss(clf, data):
    n = 0
    loss = 0

    for x, y in data:
        y_hat = clf.predict(x)
        loss += sample_logloss(y_hat, y)
        n += 1

    loss /= n

    return loss
