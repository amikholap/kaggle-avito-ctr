#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_avito_ctr.extraction import RawDataset, make_test_query, make_train_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dst', help='Name of a file to write')
    parser.add_argument('--type', choices=['train', 'test'], default='train')
    parser.add_argument('--offset', help='Skip fist N samples')
    parser.add_argument('--limit', help='Max number of entries to fetch')

    args = parser.parse_args()

    export(args.dst, args.type, args.offset, args.limit)


def export(dst, part, offset, limit):

    with RawDataset(dst, 'w') as ds:
        if part == 'train':
            q = make_train_query()
        elif part == 'test':
            q = make_test_query()

        for row in q:
            ds.append(row)


if __name__ == '__main__':
    main()
