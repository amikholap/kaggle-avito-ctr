import sqlalchemy as sa

from .globals import session
from .models import AdInfo, Category, Location, SearchInfo, TrainSearchStream, UserInfo

SearchCategory = sa.orm.aliased(Category)
AdCategory = sa.orm.aliased(Category)


COLUMNS = [
    sa.literal_column('1', type_=sa.Integer).label('intercept'),
    TrainSearchStream.is_click,
    TrainSearchStream.ad_position,
    TrainSearchStream.hist_ctr,
    sa.cast(sa.func.extract('hour', sa.cast(SearchInfo.search_date, sa.DateTime)), sa.Integer).label('hour'),
    # SearchInfo.search_query,
    SearchInfo.search_params.label('search_params'),
    AdInfo.price,
    # AdInfo.title.label('ad_title'),
    AdInfo.params.label('ad_params'),
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
    AdCategory.level.label('ad_cat_level'),
]


def extract_data(offset=None, limit=None):
    query = _make_query(offset, limit)
    columns = [c.name for c in query.statement.columns]

    for row in query:
        features = {name: value for name, value in zip(columns, row)}
        yield features


def get_field_names():
    query = _make_query()
    field_names = [c.name for c in query.statement.columns]
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

    if limit:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)

    return query
