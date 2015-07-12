import logging

import sqlalchemy as sa
import sqlalchemy.orm  # noqa


DATA = {
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

engine = sa.create_engine('postgresql://postgres@localhost:5432/avito')
Session = sa.orm.sessionmaker(bind=engine)
session = Session()


root_logger = logging.getLogger()
root_logger.addHandler(logging.StreamHandler())
root_logger.setLevel(logging.INFO)
