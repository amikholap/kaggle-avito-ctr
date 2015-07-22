from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, SmallInteger, String
from sqlalchemy.dialects.postgresql import BIT, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship


#class TableNames(object):
    #ad_info = 'ads_info_sample'
    #category = 'category'
    #location = 'location'
    #phone_request = 'phone_requests_stream'
    #search_info = 'search_info_sample'
    #train_search_stream = 'train_search_stream_sample'
    #val_search_stream = 'eval_search_stream'
    #test_search_stream = 'test_search_stream'
    #user_info = 'user_info_sample'
    #visit = 'visits_stream'


class TableNames(object):
    ad_info = 'ads_info'
    category = 'category'
    location = 'location'
    phone_request = 'phone_requests_stream'
    search_info = 'search_info'
    train_search_stream = 'train_search_stream_last_week'
    val_search_stream = 'eval_search_stream'
    test_search_stream = 'test_search_stream'
    user_info = 'user_info'
    visit = 'visits_stream'


Base = declarative_base()


class AdInfo(Base):
    __tablename__ = TableNames.ad_info

    ad_id = Column(Integer, primary_key=True)
    location_id = Column(Integer, ForeignKey('{}.location_id'.format(TableNames.location)))
    category_id = Column(Integer, ForeignKey('{}.category_id'.format(TableNames.category)))
    params = Column(JSON)
    price = Column(Float)
    title = Column(String(140))
    is_context = Column(Integer)
    n_impressions = Column(Integer)
    n_clicks = Column(Integer)

    location = relationship('Location', backref=backref('ads'))
    category = relationship('Category', backref=backref('ads'))


class Category(Base):
    __tablename__ = TableNames.category

    category_id = Column(Integer, primary_key=True)
    level = Column(SmallInteger)
    parent_category_id = Column(SmallInteger)
    subcategory_id = Column(SmallInteger)


class Location(Base):
    __tablename__ = TableNames.location

    location_id = Column(Integer, primary_key=True)
    level = Column(SmallInteger)
    region_id = Column(SmallInteger)
    city_id = Column(SmallInteger)


class PhoneRequest(Base):
    __tablename__ = TableNames.phone_request

    user_id = Column(Integer, ForeignKey('{}.user_id'.format(TableNames.user_info)), primary_key=True)
    ip_id = Column(Integer, primary_key=True)
    ad_id = Column(Integer, ForeignKey('{}.ad_id'.format(TableNames.ad_info)), primary_key=True)
    phone_request_date = Column(DateTime, primary_key=True)

    user = relationship('UserInfo', backref=backref('phone_requests'))
    ad = relationship('AdInfo', backref=backref('phone_requests'))


class SearchInfo(Base):
    __tablename__ = TableNames.search_info

    search_id = Column(Integer, primary_key=True)
    search_date = Column(DateTime)
    ip_id = Column(Integer)
    user_id = Column(Integer, ForeignKey('{}.user_id'.format(TableNames.user_info)))
    is_user_logged_on = Column(BIT)
    search_query = Column(String(760))
    location_id = Column(Integer, ForeignKey('{}.location_id'.format(TableNames.location)))
    category_id = Column(Integer, ForeignKey('{}.category_id'.format(TableNames.category)))
    search_params = Column(JSON)

    user = relationship('UserInfo', backref=backref('searches'))
    location = relationship('Location', backref=backref('searches'))
    category = relationship('Category', backref=backref('searches'))


class SearchStream(Base):
    __abstract__ = True

    ad_position = Column(SmallInteger)
    object_type = Column(SmallInteger)
    hist_ctr = Column(Float)


class LabeledSearchStream(SearchStream):
    __abstract__ = True

    is_click = Column(BIT)


class TrainSearchStream(LabeledSearchStream):
    __tablename__ = TableNames.train_search_stream

    search_id = Column(Integer, ForeignKey('{}.search_id'.format(TableNames.search_info)), primary_key=True)
    ad_id = Column(Integer, ForeignKey('{}.ad_id'.format(TableNames.ad_info)), primary_key=True)

    search = relationship('SearchInfo', backref=backref('impressions_train'))
    ad = relationship('AdInfo', backref=backref('impressions_train'))


class ValSearchStream(LabeledSearchStream):
    __tablename__ = TableNames.val_search_stream

    search_id = Column(Integer, ForeignKey('{}.search_id'.format(TableNames.search_info)), primary_key=True)
    ad_id = Column(Integer, ForeignKey('{}.ad_id'.format(TableNames.ad_info)), primary_key=True)

    search = relationship('SearchInfo', backref=backref('impressions_val'))
    ad = relationship('AdInfo', backref=backref('impressions_val'))


class TestSearchStream(SearchStream):
    __tablename__ = TableNames.test_search_stream

    id = Column(Integer)
    search_id = Column(Integer, ForeignKey('{}.search_id'.format(TableNames.search_info)), primary_key=True)
    ad_id = Column(Integer, ForeignKey('{}.ad_id'.format(TableNames.ad_info)), primary_key=True)

    search = relationship('SearchInfo', backref=backref('impressions_test'))
    ad = relationship('AdInfo', backref=backref('impressions_test'))


class UserInfo(Base):
    __tablename__ = TableNames.user_info

    user_id = Column(Integer, primary_key=True)
    user_agent_id = Column(Integer)
    user_agent_osid = Column(SmallInteger)
    user_device_id = Column(SmallInteger)
    user_agent_family_id = Column(SmallInteger)

    n_context_impressions = Column(Integer)
    n_context_clicks = Column(Integer)


class Visit(Base):
    __tablename__ = TableNames.visit

    user_id = Column(Integer, ForeignKey('{}.user_id'.format(TableNames.user_info)), primary_key=True)
    ip_id = Column(Integer, primary_key=True)
    ad_id = Column(Integer, ForeignKey('{}.ad_id'.format(TableNames.ad_info)), primary_key=True)
    view_date = Column(DateTime, primary_key=True)

    user = relationship('UserInfo', backref=backref('visits'))
    ad = relationship('AdInfo', backref=backref('visits'))
