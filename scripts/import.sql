--------------------------------------------------------
--   Date   | Version  |     Who    | Description      |
-- 20150605 |   0.1    | MPekalski  | Initial Version  |
--------------------------------------------------------
--
--
-- 01 | Train Search Stream
DROP TABLE IF EXISTS train_search_stream;
CREATE TABLE train_search_stream
(
  search_id integer,
  ad_id integer,
  ad_position smallint,
  object_type smallint,
  hist_ctr real,
  is_click bit(1)    -- 0/1/NULL
);
COPY train_search_stream (search_id, ad_id , ad_position, object_type, hist_ctr, is_click) FROM '/avito/data/trainSearchStream.tsv' CSV DELIMITER E'\t' HEADER;
-- Query returned successfully: 392 356 948 rows affected, 394 093 ms execution time.

ALTER TABLE train_search_stream ADD CONSTRAINT pk_train_search_stream PRIMARY KEY (search_id, ad_id);
CLUSTER train_search_stream USING pk_train_search_stream;
ANALYZE train_search_stream;
--
--
-- 02 | User Info
DROP TABLE IF EXISTS user_info;
CREATE TABLE user_info
(
	user_id INTEGER PRIMARY KEY   -- max 4 339 861
,	user_agent_id INTEGER         -- max 64017
, user_agent_osid SMALLINT      -- max 51
, user_device_id SMALLINT       -- max 4213
, user_agent_family_id SMALLINT -- max 89
);

COPY user_info (user_id, user_agent_id, user_agent_osid, user_device_id, user_agent_family_id) FROM E'/avito/data/UserInfo.tsv' DELIMITER E'\t' CSV HEADER;
CLUSTER user_info USING user_info_pkey;
ANALYZE user_info;
--
--
-- 03 | Category
DROP TABLE IF EXISTS category;
CREATE TABLE category
(
	category_id INTEGER PRIMARY KEY, -- max 500 001
	level SMALLINT,  									-- max 3
	parent_category_id SMALLINT, 			-- max 12
	subcategory_id SMALLINT						-- max 57
);

COPY category (category_id, level, parent_category_id, subcategory_id) FROM '/avito/data/Category.tsv' CSV DELIMITER E'\t' HEADER;
CLUSTER category USING category_pkey;
ANALYZE category;
--
--
-- 04 | Location
DROP TABLE IF EXISTS location;
CREATE TABLE location
(
	location_id INTEGER PRIMARY KEY, 	-- max 1 250 001
	level SMALLINT,  								 	-- 1/2/3
	region_id SMALLINT,								-- < 100
	city_id SMALLINT 							  	-- < 9999
);

COPY location (location_id, level, region_id, city_id) FROM '/avito/data/Location.tsv' CSV DELIMITER E'\t' HEADER;
CLUSTER location USING location_pkey;
ANALYZE location;
--
--
-- 05 | AdsInfo
DROP TABLE IF EXISTS ads_info;
CREATE TABLE ads_info
(
	ad_id INTEGER PRIMARY KEY,
	location_id INTEGER,
	category_id INTEGER,
	params VARCHAR(560),
	price REAL,
	title VARCHAR(140),
	is_context INTEGER
);

COPY ads_info (ad_id, location_id, category_id, params, price, title, is_context) FROM '/avito/data/AdsInfo.tsv' CSV DELIMITER E'\t' HEADER;

CLUSTER ads_info USING ads_info_pkey;
CREATE INDEX ads_info_location_id ON ads_info (location_id) WITH (FILLFACTOR = 100);
CREATE INDEX ads_info_category_id ON ads_info (category_id) WITH (FILLFACTOR = 100);
ANALYZE ads_info;
--
--
-- 06 | testSearchStream
DROP TABLE IF EXISTS test_search_stream;
CREATE TABLE test_search_stream
(
  id INTEGER, 
  search_id INTEGER,
  ad_id INTEGER,
  ad_position SMALLINT, 
  object_type SMALLINT,
  hist_ctr REAL
);
COPY test_search_stream (id, search_id, ad_id , ad_position, object_type, hist_ctr) FROM '/avito/data/testSearchStream.tsv' CSV DELIMITER E'\t' HEADER;
ALTER TABLE test_search_stream ADD CONSTRAINT pk_test_search_stream PRIMARY KEY (search_id, ad_id);
CLUSTER test_search_stream USING pk_test_search_stream;
ANALYZE test_search_stream;
--
--
-- 07 | SearchInfo
DROP TABLE IF EXISTS search_info;
CREATE TABLE search_info
(
	search_id INTEGER, --not a primary key b/c: TSV_no_header/SearchInfo.tsv:1735958: INSERT failed: UNIQUE constraint failed: SearchInfo.SearchID
	search_date TIMESTAMP,
	ip_id INTEGER,
	user_id INTEGER,
	is_user_logged_on BIT(1),
	search_query VARCHAR(760),
	location_id INTEGER,
	category_id INTEGER,
	search_params VARCHAR(175)
);
COPY search_info (search_id, search_date, ip_id , user_id, is_user_logged_on, search_query, location_id, category_id, search_params) FROM '/avito/data/SearchInfo.tsv' CSV DELIMITER E'\t' HEADER;

ALTER TABLE search_info ADD CONSTRAINT pk_search_info PRIMARY KEY (search_id);
CLUSTER search_info USING pk_search_info;
ANALYZE search_info;
--
--
-- 08 | VisitsStream
DROP TABLE IF EXISTS visits_stream;
CREATE TABLE visits_stream
(
	user_id INTEGER,
	ip_id INTEGER,
	ad_id INTEGER,
	view_date TIMESTAMP
);
COPY visits_stream (user_id, ip_id, ad_id, view_date) FROM '/avito/data/VisitsStream.tsv' CSV DELIMITER E'\t' HEADER;
-- 286 821 375
CREATE UNIQUE INDEX visits_stream_all ON visits_stream(user_id, ad_id, ip_id, view_date) WITH (FILLFACTOR = 100) ;
CLUSTER visits_stream USING visits_stream_all;
DROP INDEX visits_stream_all;
CREATE INDEX visits_stream_user_id ON visits_stream (user_id)  	WITH (FILLFACTOR = 100);
CREATE INDEX visits_stream_ip_id ON visits_stream (ip_id) 			WITH (FILLFACTOR = 100);
CREATE INDEX visits_stream_ad_id ON visits_stream (ad_id)  			WITH (FILLFACTOR = 100);
ANALYZE visits_stream;
-- 09 | VisitsStream
DROP TABLE IF EXISTS phone_requests_stream;
CREATE TABLE phone_requests_stream
(
	user_id INTEGER,
	ip_id INTEGER,
	ad_id INTEGER,
	phone_request_date TIMESTAMP
);
COPY phone_requests_stream (user_id, ip_id, ad_id, phone_request_date) FROM '/avito/data/PhoneRequestsStream.tsv' CSV DELIMITER E'\t' HEADER;
-- 286 821 375
CREATE UNIQUE INDEX phone_requests_stream_all ON phone_requests_stream(user_id, ad_id, ip_id, phone_request_date) WITH (FILLFACTOR = 100) ;
CLUSTER phone_requests_stream USING phone_requests_stream_all;
DROP INDEX phone_requests_stream_all;
CREATE INDEX phone_requests_stream_user_id ON phone_requests_stream (user_id)  	WITH (FILLFACTOR = 100);
CREATE INDEX phone_requests_stream_ip_id ON phone_requests_stream (ip_id)  			WITH (FILLFACTOR = 100);
CREATE INDEX phone_requests_stream_ad_id ON phone_requests_stream (ad_id)  			WITH (FILLFACTOR = 100);
ANALYZE phone_requests_stream;
