# Airflow DAGs

DAGs land here in subsequent PRs:
- PR G: daily_ingest_dag
- PR H: historical_backfill_dag
- PR I: lifecycle_demote_dag
- PR M: feature_refresh_dag
- PR P: model_retrain_dag

Mounted into airflow-webserver and airflow-scheduler containers at
/opt/airflow/dags.

To trigger a DAG manually: open http://localhost:8090, log in with
admin/admin (or override via .env), find the DAG, click play.
