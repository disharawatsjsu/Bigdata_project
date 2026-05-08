CREATE DATABASE IF NOT EXISTS supply_chain;
USE supply_chain;

CREATE EXTERNAL TABLE IF NOT EXISTS market_prices (
  price_date STRING,
  close_price DOUBLE,
  commodity STRING,
  symbol STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/opt/data/commodities_hive/'
TBLPROPERTIES ('skip.header.line.count'='1');
