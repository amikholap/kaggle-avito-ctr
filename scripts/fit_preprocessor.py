#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_avito_ctr.preprocessing import Preprocessor
from kaggle_avito_ctr.utils import stream_sparse_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file', help='TSV containing the dataset')
    parser.add_argument('target_file', help='Filename to store pickled preprocessor')

    args = parser.parse_args()

    preprocessor = fit_preprocessor(args.source_file)
    save_preprocessor(preprocessor, args.target_file)


def fit_preprocessor(source):
    preprocessor = Preprocessor()
    X_factory = lambda: stream_sparse_dataset(source)
    preprocessor.fit(X_factory)
    return preprocessor


def save_preprocessor(preprocessor, dst):
    with open(dst, 'wb') as f:
        pickle.dump(preprocessor, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
