import sqlalchemy as sa

from .globals import session
from .models import AdInfo, Category, Location, SearchInfo, TrainSearchStream, UserInfo


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
    search_category = sa.orm.aliased(Category)
    ad_category = sa.orm.aliased(Category)

    query = (session.query(TrainSearchStream)
             .join(AdInfo, TrainSearchStream.ad_id == AdInfo.ad_id)
             .join(SearchInfo, TrainSearchStream.search_id == SearchInfo.search_id)
             .join(Location, SearchInfo.location_id == Location.location_id)
             .join(UserInfo, SearchInfo.user_id == UserInfo.user_id)
             .join(search_category, SearchInfo.category_id == search_category.category_id)
             .join(ad_category, AdInfo.category_id == ad_category.category_id)
             .with_entities(sa.literal_column('1').label('intercept'),
                            TrainSearchStream.is_click, TrainSearchStream.ad_position, TrainSearchStream.hist_ctr,
                            sa.cast(sa.func.extract('hour', sa.cast(SearchInfo.search_date, sa.DateTime)),
                                    sa.Integer).label('hour'),
                            AdInfo.price,
                            Location.level.label('loc_level'), Location.region_id, Location.city_id,
                            UserInfo.user_agent_id, UserInfo.user_agent_family_id,
                            UserInfo.user_agent_osid, UserInfo.user_device_id,
                            search_category.category_id.label('search_cat_id'),
                            search_category.level.label('search_cat_level'),
                            ad_category.category_id.label('ad_cat_id'), ad_category.level.label('ad_cat_level'))
             .yield_per(1000))

    if limit:
        query = query.limit(limit)
    if offset:
        query = query.offset(offset)

    return query
