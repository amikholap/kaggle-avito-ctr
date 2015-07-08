#!/usr/bin/env python3
import argparse
import csv
import gzip
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_avito_ctr.extraction import extract_data, get_field_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dst', help='Name of a file to write')
    parser.add_argument('--offset', help='Skip fist N samples')
    parser.add_argument('--limit', help='Max number of entries to fetch')

    args = parser.parse_args()

    export(args.dst, args.offset, args.limit)


def export(dst, offset, limit):
    field_names = get_field_names()

    with gzip.open(dst, 'wt') as f:
        writer = csv.DictWriter(f, fieldnames=field_names, delimiter='\t')
        writer.writeheader()
        for row in extract_data(offset, limit):
            writer.writerow(row)


if __name__ == '__main__':
    main()
