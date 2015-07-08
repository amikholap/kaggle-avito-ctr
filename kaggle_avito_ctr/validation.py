import gzip
import math
import tempfile

from .utils import stream_encoded_dataset


def cv(clf, filename, n_folds=5):
    scores = []

    for part in range(1, n_folds + 1):
        with gzip.open(filename, 'rt') as src, \
                tempfile.NamedTemporaryFile(mode='w') as train_dst, \
                tempfile.NamedTemporaryFile(mode='w') as test_dst:

            with gzip.open(train_dst.name, 'wt') as train_dst_gz, \
                    gzip.open(test_dst.name, 'wt') as test_dst_gz:
                train_test_split(src, train_dst_gz, test_dst_gz, n_folds, part)

            clf.fit(stream_encoded_dataset(train_dst.name))
            score = logloss(clf, stream_encoded_dataset(test_dst.name))

        scores.append(score)

        print('CV {}/{} score: {}'.format(part, n_folds, score))

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


def train_test_split(src, train_dst, test_dst, n_folds, part=1):
    assert n_folds > 1
    assert 0 < part <= n_folds

    for i, line in enumerate(src):
        if (i + part) % n_folds == 0:
            test_dst.write(line)
        else:
            train_dst.write(line)

    train_dst.flush()
    test_dst.flush()
