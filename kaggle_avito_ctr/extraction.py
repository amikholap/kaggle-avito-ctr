import gzip
import json

import sqlalchemy as sa

from .globals import session
from .models import (AdInfo, Category, Location, SearchInfo, TestSearchStream,
                     TrainSearchStream, ValSearchStream, UserInfo)


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

    def get_field_names(self, part):
        q = _make_query(part)
        field_names = [c.name for c in q.statement.columns]
        return field_names

    def sparse_iterator(self, part):
        """
        Iterate through a dataset converting values to (name, value) tuples.
        """

        assert part in ('train', 'eval', 'test')

        field_names = self.get_field_names(part)
        for row in self.iterator():
            row = list(zip(field_names, row))
            yield row


class SparseDataset(JsonFormatMixin, GzipCompressorMixin, Dataset):

    def iterator(self, *args, **kwargs):
        """
        Iterate over (x, label) pairs.

        label is either target for train or sample id for test.
        """

        it = super().iterator(*args, **kwargs)
        for row in it:
            label = row.pop(0)[2]
            yield row, label


def extract_data(offset=None, limit=None):
    query = _make_query(offset, limit)
    for row in query:
        yield row


def _make_query(part, offset=None, limit=None):

    AdCategory = sa.orm.aliased(Category)
    SearchCategory = sa.orm.aliased(Category)

    if part == 'train':
        SearchStream = TrainSearchStream
    elif part == 'eval':
        SearchStream = ValSearchStream
    elif part == 'test':
        SearchStream = TestSearchStream

    query = (session.query(SearchStream)
             .join(AdInfo, SearchStream.ad_id == AdInfo.ad_id)
             .outerjoin(AdCategory, AdInfo.category_id == AdCategory.category_id)
             .join(SearchInfo, SearchStream.search_id == SearchInfo.search_id)
             .join(SearchCategory, SearchInfo.category_id == SearchCategory.category_id)
             .join(Location, SearchInfo.location_id == Location.location_id)
             .outerjoin(UserInfo, SearchInfo.user_id == UserInfo.user_id)
             .filter(SearchStream.object_type == 3)
             .yield_per(1000))

    columns = []

    if hasattr(SearchStream, 'is_click'):
        columns.append(
            sa.cast(SearchStream.is_click, sa.Integer).label('is_click')
        )
    else:
        columns.append(SearchStream.id)

    columns.extend([
        sa.literal_column('1', type_=sa.Integer).label('intercept'),

        SearchStream.ad_position,
        # SearchStream.hist_ctr,

        sa.cast(sa.func.extract('hour', sa.cast(SearchInfo.search_date, sa.DateTime)), sa.Integer).label('hour'),
        # SearchInfo.search_query,
        # SearchInfo.search_params.label('search_params'),
        SearchCategory.category_id.label('search_cat_id'),
        SearchCategory.level.label('search_cat_level'),

        AdInfo.price,
        # AdInfo.title.label('ad_title'),
        AdInfo.params.label('ad_params'),
        sa.func.coalesce(AdCategory.category_id, -1).label('ad_cat_id'),
        sa.func.coalesce(AdInfo.n_impressions, 0).label('ad_n_impressions'),
        sa.func.coalesce(AdInfo.n_clicks, 0).label('ad_n_clicks'),

        sa.func.coalesce(UserInfo.user_agent_id, -1).label('user_agent_id'),
        sa.func.coalesce(UserInfo.user_agent_family_id, -1).label('user_agent_family_id'),
        sa.func.coalesce(UserInfo.user_agent_osid, -1).label('user_agent_osid'),
        sa.func.coalesce(UserInfo.user_device_id, -1).label('user_device_id'),
        sa.func.coalesce(UserInfo.n_context_impressions, 0).label('user_n_impressions'),
        sa.func.coalesce(UserInfo.n_context_clicks, 0).label('user_n_clicks'),

        Location.level.label('loc_level'),
        Location.region_id,
        Location.city_id,
    ])

    query = query.with_entities(*columns)

    if offset:
        query = query.offset(offset)
    if limit:
        query = query.limit(limit)

    return query


def make_train_query(*args, **kwargs):
    return _make_query('train', *args, **kwargs)


def make_test_query(*args, **kwargs):
    return _make_query('test', *args, **kwargs)


def make_val_query(*args, **kwargs):
    return _make_query('eval', *args, **kwargs)
