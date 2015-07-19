#!/usr/bin/env python3
import argparse
import csv
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_avito_ctr.extraction import SparseDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Name of a file containing a model')
    parser.add_argument('dataset', help='Name of a file containing test dataset')
    parser.add_argument('dst', help='Name of a submission CSV file')

    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    make_submission(model, args.dataset, args.dst)


def make_submission(model, dataset_filename, dst):
    header = ['ID', 'IsClick']

    with open(dst, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with SparseDataset(dataset_filename) as dataset:
            for sample_id, prediction in _stream_predictions(model, dataset):
                row = (sample_id, prediction)
                writer.writerow(row)


def _stream_predictions(model, dataset):
    for row, sample_id in dataset.iterator():
        prediction = model.predict(row)
        yield sample_id, prediction


if __name__ == '__main__':
    main()
