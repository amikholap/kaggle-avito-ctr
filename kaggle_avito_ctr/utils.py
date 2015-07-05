import csv
import json

from .globals import DATA, DATA_TYPES


def stream_sparse_dataset(tsv):
    """
    Iterate through tsv converting rows to sparse representation.

    Target variable is always the first of a row.
    """

    with open(tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')

        for raw_row in reader:
            row = [None] * len(raw_row)

            y_fieldname = DATA['y']
            y_dtype = DATA_TYPES[y_fieldname]
            y = y_dtype(raw_row.pop(y_fieldname))
            row[0] = (y_fieldname, y)

            for i, (k, v) in enumerate(raw_row.items(), 1):
                if v:
                    dtype = DATA_TYPES[k]
                    typed_val = dtype(v)
                else:
                    typed_val = v
                row[i] = (k, typed_val)

            yield row


def stream_encoded_dataset(filename):
    """
    Iterate through encoded dataset yielding (x, y) pairs.

    x is a list of (field, index, value).
    y is an integer.
    """
    with open(filename) as f:
        for line in f:
            row = json.loads(line)
            y = row.pop(0)[2]
            yield row, y
