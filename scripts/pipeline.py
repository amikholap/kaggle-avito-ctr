#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_avito_ctr.extraction import RawDataset, SparseDataset, make_test_query, make_train_query
from kaggle_avito_ctr.preprocessing import Preprocessor


def main():
    parser = init_parser()
    args = parser.parse_args()

    do_export = do_fitpp = do_process = dofit = docv = False

    if args.export:
        do_export = True
    elif args.fitpp:
        do_fitpp = True
    elif args.process:
        do_process = True
    elif args.cv:
        do_cv = True
    else:
        do_export = do_fitpp = do_process = dofit = docv = True

        if args.noexport:
            do_export = False

        if args.nofitpp:
            do_fitpp = False

        if args.noprocess:
            do_process = False

        if args.nofit:
            do_fit = False

        if args.nocv:
            do_cv = False
        
    if do_export:
        print('Exporting dataset to {}'.format(args.raw_dataset))
        export(args.raw_dataset, args.format)
    else:
        print('Skipping dataset export')

    if do_fitpp:
        print('Fitting preprocessor to {}'.format(args.preprocessor))
        preprocessor = fit_preprocessor(args.raw_dataset)
        serialize(preprocessor, args.preprocessor)
    else:
        print('Skipping preprocessor fitting')
        preprocessor = deserialize(args.preprocessor)

    if do_process:
        print('Preprocessing raw dataset {} to {}'.format(args.raw_dataset, args.dataset))
        transform(args.raw_dataset, args.dataset, preprocessor, args.format)
    else:
        print('Skipping raw dataset preprocessing')


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('raw_dataset', help='Name of a file containing raw dataset')
    parser.add_argument('dataset', help='Name of a file containing preprocessed dataset')
    parser.add_argument('preprocessor', help='Name of a file containing pickled preprocessor')
    parser.add_argument('model', help='Name of a file containing pickled model')

    parser.add_argument('--format', choices=['train', 'test'], help='Train or test data format')

    parser.add_argument('--noexport', action='store_true', help='Skip DB export')
    parser.add_argument('--nofitpp', action='store_true', help='Skip preprocessor fitting')
    parser.add_argument('--noprocess', action='store_true', help='Skip dataset preprocessing')
    parser.add_argument('--nofit', action='store_true', help='Skip model training')
    parser.add_argument('--nocv', action='store_true', help='Skip model validation')

    parser.add_argument('--export', action='store_true', help='Perform only DB export')
    parser.add_argument('--fitpp', action='store_true', help='Perform only preprocessor fitting')
    parser.add_argument('--process', action='store_true', help='Perform only dataset preprocessing')
    parser.add_argument('--fit', action='store_true', help='Perform only model training')
    parser.add_argument('--cv', action='store_true', help='Perform only model validation')

    return parser


def serialize(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def deserialize(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def export(dst, part):

    with RawDataset(dst, 'w') as ds:
        if part == 'train':
            q = make_train_query()
        elif part == 'test':
            q = make_test_query()

        for row in q:
            ds.append(row)


def fit_preprocessor(dataset):
    X_factory = lambda: RawDataset(dataset).sparse_iterator('train')
    preprocessor = Preprocessor()
    preprocessor.fit(X_factory)
    return preprocessor


def transform(src, dst, preprocessor, part):
    with RawDataset(src) as src_ds:
        with SparseDataset(dst, 'w') as dst_ds:
            for (i, row) in enumerate(src_ds.sparse_iterator(part)):
                transformed_row = preprocessor.transform(row)
                dst_ds.append(transformed_row)

                if i % 100000 == 0:
                    print('Processed {} rows'.format(i), end='\r')

    print()


if __name__ == '__main__':
    main()
