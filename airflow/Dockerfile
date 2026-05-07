# Airflow image with Docker CLI so DAG tasks can `docker exec` into spark-master / namenode.
# Requires /var/run/docker.sock mounted in docker-compose (see webserver + scheduler).
FROM apache/airflow:2.10.3

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
 && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir requests
