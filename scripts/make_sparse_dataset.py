#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_avito_ctr.extraction import RawDataset, SparseDataset


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
    with RawDataset(src) as src_ds:
        with SparseDataset(dst, 'w') as dst_ds:
            for (i, row) in enumerate(src_ds.sparse_iterator()):
                transformed_row = preprocessor.transform(row)
                dst_ds.append(transformed_row)

                if i % 100000 == 0:
                    print('Processed {} rows'.format(i), end='\r')

    print()


if __name__ == '__main__':
    main()
