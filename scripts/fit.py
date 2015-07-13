#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_avito_ctr.extraction import SparseDataset
from kaggle_avito_ctr.online_lr import OnlineLogisticRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Name of a file containing the dataset')
    parser.add_argument('dst', help='Name of a file to save fitted model')

    args = parser.parse_args()

    print('Begin training')

    with SparseDataset(args.dataset) as X:
        model = fit(X.iterator())

    print('Training succeded')

    with open(args.dst, 'wb') as f:
        pickle.dump(model, f)

    print_summary(model)


def fit(X):
    model = OnlineLogisticRegression()
    model.fit(X)
    return model


def print_summary(model):
    print('Top important features:')
    print('{:25} | {:5} | {:10}'.format('feature', 'index', 'weight'))
    print('-' * 50)
    for f, i, w in model.weights_flat[-10:]:
        print('{:25} | {:5} | {:10.5}'.format(f, i, w))


if __name__ == '__main__':
    main()
