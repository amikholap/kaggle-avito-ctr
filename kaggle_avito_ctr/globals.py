import sqlalchemy as sa
import sqlalchemy.orm  # noqa


engine = sa.create_engine('postgresql://postgres@localhost:5432/avito')
Session = sa.orm.sessionmaker(bind=engine)
session = Session()

DATA = {
    'y': 'is_click',
    'intercept': 'intercept',
    'CATEGORICAL': {
        'hour',
        'user_agent_id',
        'user_agent_family_id',
        'user_agent_osid',
        'user_device_id',
        'loc_level',
        'region_id',
        'city_id',
        'search_cat_id',
        'search_cat_level',
        'ad_cat_id',
        'ad_cat_level',
        'price_percentile',
    },
}

DATA_TYPES = {
    'intercept': int,
    'is_click': int,
    'ad_position': int,
    'hist_ctr': float,
    'price': float,
    'price_percentile': int,
    'hour': int,
    'user_agent_id': int,
    'user_agent_family_id': int,
    'user_agent_osid': int,
    'user_device_id': int,
    'loc_level': int,
    'region_id': int,
    'city_id': int,
    'search_query': str,
    'query_common_tokens': float,
    'query_common_numbers': float,
    'query_lcs': float,
    'search_cat_id': int,
    'search_cat_level': int,
    'ad_title': str,
    'ad_cat_id': int,
    'ad_cat_level': int,
}
