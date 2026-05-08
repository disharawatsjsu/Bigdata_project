# Stack Health Notes

Living document of known issues and workarounds for the docker compose
stack. Updated as services are diagnosed.

## streamlit

**Status:** Up (healthy after full `docker compose down` / `up -d` on 2026-05-06).

**Previous observation:** Container had been **Exited (0)** in an older session (likely after manual `docker compose down`, resource pressure, or a completed one-shot run that exited cleanly).

**Notes:** First boot runs `pip install` inside the container; that can take 1–2 minutes. Logs showed a normal Streamlit startup (`Uvicorn server started on 0.0.0.0:8501`).

**Workaround:** If Streamlit is down, run `docker compose up -d streamlit` (or recycle the full stack). Check logs: `docker compose logs streamlit --tail 100`.

## spark-worker-1 / spark-worker-2

**Status:** Up (registered with `spark://spark-master:7077` after full stack restart on 2026-05-06).

**Previous observation:** Workers had been **Exited (137)** — often **SIGKILL** (Docker OOM or manual stop) when the host was tight on memory or after an unclean shutdown.

**Notes:** Current logs show successful registration: `Successfully registered with master spark://spark-master:7077`. WARN about `NativeCodeLoader` is normal on some platforms.

**Workaround:** Ensure Docker Desktop has enough RAM (e.g. 13–16 GB for the full stack). Recreate workers: `docker compose up -d spark-worker-1 spark-worker-2`. See Spark master UI on host port **8080**.

## kafka

Default host port changed from **29092** → **39092** (compose default `${KAFKA_HOST_PORT:-39092}`) to reduce clashes with other local Kafka stacks that often use 29092.

If **39092** is still in use, set **`KAFKA_HOST_PORT`** in `.env` to a free port. Inside the Docker network, clients still use **`kafka:9092`**.

## airflow-init

**Expected** to exit with code **0** after DB migration and admin user creation. This is **not** a failure. Long-running Airflow processes are **airflow-webserver** and **airflow-scheduler**.

## hdfs replication / rebalance

HDFS replication is set to **1** (vs the Hadoop default 3) to reduce local disk usage during bulk historical backfills.

If you change replication (or add/remove datanodes) and want to force a re-replication sweep, run:

- `docker compose exec namenode hdfs dfsadmin -setrep -R -w 1 /supply-chain`

Optional: run the balancer to smooth block placement:

- `docker compose exec namenode hdfs balancer`
