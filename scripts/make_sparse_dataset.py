#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_avito_ctr.utils import stream_sparse_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file', help='TSV containing the dataset')
    parser.add_argument('target_file', help='Gzipped text file containing the result')
    parser.add_argument('preprocessor', help='Pickled preprocessor')

    args = parser.parse_args()

    with open(args.preprocessor, 'rb') as f:
        preprocessor = pickle.load(f)

    transform(args.source_file, args.target_file, preprocessor)


def transform(src, dst, preprocessor):
    with gzip.open(dst, 'wt') as dst_file:
        for i, row in enumerate(stream_sparse_dataset(src)):
            transformed_row = preprocessor.transform(row)
            transformed_row_json = json.dumps(transformed_row)
            dst_file.write(transformed_row_json)
            dst_file.write('\n')

            if i % 100000 == 0:
                print(i)


if __name__ == '__main__':
    main()
