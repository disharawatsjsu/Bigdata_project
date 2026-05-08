CREATE DATABASE IF NOT EXISTS supply_chain;
USE supply_chain;

CREATE EXTERNAL TABLE IF NOT EXISTS predictions_v2 (
  event_date TIMESTAMP,
  predicted_value DOUBLE,
  actual_value DOUBLE,
  naive_value DOUBLE
)
STORED AS PARQUET
LOCATION '/supply-chain/analytics/predictions_v2/'
TBLPROPERTIES ('mapred.input.dir.recursive'='true');

CREATE EXTERNAL TABLE IF NOT EXISTS shap_attribution_v2 (
  chokepoint STRING,
  commodity STRING,
  target_horizon STRING,
  mean_abs_shap DOUBLE,
  normalized_share DOUBLE
)
STORED AS PARQUET
LOCATION '/supply-chain/analytics/shap_attribution_v2/';

CREATE EXTERNAL TABLE IF NOT EXISTS regression_metrics (
  commodity STRING,
  target STRING,
  test_r2 DOUBLE,
  test_mae DOUBLE,
  naive_mae DOUBLE,
  improvement_pct DOUBLE,
  spearman DOUBLE,
  decision_flag STRING
)
STORED AS PARQUET
LOCATION '/supply-chain/analytics/regression_metrics/';
