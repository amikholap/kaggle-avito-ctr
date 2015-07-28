import datetime
import logging

import sqlalchemy as sa
import sqlalchemy.orm  # noqa


DATA = {
    # First day of usable ciunters.
    'COUNTER_START_DATE': datetime.datetime(2015, 5, 3).timestamp(),

    'CONTINUOUS': {
        'intercept',
        'user_n_visits',
        'user_n_phone_requests',
        'ad_ctr_scaled',
        'ad_ctr_root_scaled',
        'ad_ctr_pow2_scaled',
        'ad_ctr_pow3_scaled',
        'user_ctr_scaled',
        'user_ctr_root_scaled',
        'user_ctr_pow2_scaled',
        'user_ctr_pow3_scaled',
    },
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
        'ad_parameter',
        'price_percentile',
        'hist_ctr_percentile',
    },
}

engine = sa.create_engine('postgresql://postgres@localhost:5432/avito')
Session = sa.orm.sessionmaker(bind=engine)
session = Session()


root_logger = logging.getLogger()
root_logger.addHandler(logging.StreamHandler())
root_logger.setLevel(logging.INFO)
