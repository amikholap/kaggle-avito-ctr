import gzip
import json

import sqlalchemy as sa

from .globals import session
from .models import AdInfo, Category, Location, SearchInfo, TrainSearchStream, UserInfo

SearchCategory = sa.orm.aliased(Category)
AdCategory = sa.orm.aliased(Category)


COLUMNS = [
    sa.cast(TrainSearchStream.is_click, sa.Integer).label('is_click'),
    sa.literal_column('1', type_=sa.Integer).label('intercept'),
    TrainSearchStream.ad_position,
    TrainSearchStream.hist_ctr,
    sa.cast(sa.func.extract('hour', sa.cast(SearchInfo.search_date, sa.DateTime)), sa.Integer).label('hour'),
    # SearchInfo.search_query,
    # SearchInfo.search_params.label('search_params'),
    AdInfo.price,
    # AdInfo.title.label('ad_title'),
    # AdInfo.params.label('ad_params'),
    Location.level.label('loc_level'),
    Location.region_id,
    Location.city_id,
    UserInfo.user_agent_id,
    UserInfo.user_agent_family_id,
    UserInfo.user_agent_osid,
    UserInfo.user_device_id,
    SearchCategory.category_id.label('search_cat_id'),
    SearchCategory.level.label('search_cat_level'),
    AdCategory.category_id.label('ad_cat_id'),
]


def _json_converter(v):
    return json.loads(v)


_special_converters = {
    'is_click': int,
    'params': _json_converter,
    'search_params': _json_converter,
}


class Dataset(object):
    """Generic file-based dataset."""

    def __init__(self, filename, mode='r'):
        self.file = self._open(filename, mode=mode)

    def _open(self, filename, *args, **kwargs):
        f = open(filename, *args, **kwargs)
        return f

    def _encode_row(self, row):
        return row

    def _decode_row(self, row):
        return row

    def append(self, row):
        line = self._encode_row(row)
        line += '\n'
        self.file.write(line)

    def iterator(self, offset=0, limit=None, skip_nth=None, every_nth=None, n_cycles=1):
        """
        Iterator for the dataset.

        Args:
            offset: Skip first N rows.
            limit: Return no more than N rows.
            skip_nth: Skip every N-th row.
            every_nth: Return only every N-th row.
            n_cycles: Iterate N times.
        """

        n_yielded = 0

        for _ in range(n_cycles):

            self.reset()

            for i, line in enumerate(self.file):

                if i < offset:
                    continue

                if limit is not None and n_yielded >= limit:
                    break

                # +1 to avoid cycling.
                # This doesn't change the semantics.
                if skip_nth is not None and (i + offset + 1) % skip_nth == 0:
                    continue

                if every_nth is not None and (i + offset + 1) % every_nth != 0:
                    continue

                line = line.rstrip('\n')
                row = self._decode_row(line)

                yield row

                n_yielded += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def flush(self):
        self.file.flush()

    def reset(self):
        """Set cursor position to the beginning of a file."""
        self.file.seek(0)


class GzipCompressorMixin(object):

    def _open(self, filename, mode='r', encoding='utf-8'):
        gzip_mode = mode + 't'
        f = gzip.open(filename, gzip_mode, encoding=encoding)
        return f


class JsonFormatMixin(object):

    def _encode_row(self, row):
        encoded_row = json.dumps(row, ensure_ascii=False)
        return encoded_row

    def _decode_row(self, row):
        decoded_row = json.loads(row)
        return decoded_row


class RawDataset(JsonFormatMixin, GzipCompressorMixin, Dataset):

    def sparse_iterator(self):
        """
        Iterate through a dataset converting values to (name, value) tuples.
        """
        field_names = get_field_names()
        for row in self.iterator():
            row = list(zip(field_names, row))
            yield row


class SparseDataset(JsonFormatMixin, GzipCompressorMixin, Dataset):

    def iterator(self, *args, **kwargs):
        """
        Iterate over (x, y) pairs.
        """
        it = super().iterator(*args, **kwargs)
        for row in it:
            y = row.pop(0)[2]
            yield row, y


def extract_data(offset=None, limit=None):
    query = _make_query(offset, limit)
    for row in query:
        yield row


def get_field_names():
    field_names = [c.name for c in COLUMNS]
    return field_names


def _make_query(offset=None, limit=None):

    query = (session.query(TrainSearchStream)
             .join(AdInfo, TrainSearchStream.ad_id == AdInfo.ad_id)
             .join(SearchInfo, TrainSearchStream.search_id == SearchInfo.search_id)
             .join(Location, SearchInfo.location_id == Location.location_id)
             .join(UserInfo, SearchInfo.user_id == UserInfo.user_id)
             .join(SearchCategory, SearchInfo.category_id == SearchCategory.category_id)
             .join(AdCategory, AdInfo.category_id == AdCategory.category_id)
             .with_entities(*COLUMNS)
             .filter(AdInfo.is_context == 1)
             .yield_per(1000))

    if offset:
        query = query.offset(offset)
    if limit:
        query = query.limit(limit)

    return query
