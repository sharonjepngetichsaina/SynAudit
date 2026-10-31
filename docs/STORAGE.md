# Storage and dashboard (Alina)

InfluxDB stores every result; the Streamlit dashboard shows them live.

| Service | URL | Login |
|---|---|---|
| Streamlit dashboard | http://localhost:8501 | — |
| InfluxDB web UI | http://localhost:8086 | admin / synaudit123 |

Connection settings: `INFLUX_URL=http://influxdb:8086` inside Docker, `http://localhost:8086` on a laptop
(the default), `INFLUX_TOKEN=synaudit-dev-token`, `INFLUX_ORG=synaudit`, `INFLUX_BUCKET=synaudit`.
Datasets are named like the Kafka topics: `adult`, `credit-default`. `RUN_ID` labels an experiment.

## How each service saves its results

Each one keeps its own result format; `storage/db.py` takes it as it is.

| Who | What to add | Measurement |
|---|---|---|
| Diana (quality) | nothing: `quality/sinks.py` already writes to InfluxDB | `synaudit_quality`, `synaudit_column_quality` (see `INFLUX_SCHEMA.md`) |
| Grace (drift) | `db.write_drift_rows(window_tests(...))` | `drift_events` |
| Sharon (utility) | `db.write_utility_rows(df)`, df with the columns of `adult-utility.csv` | `utility` (+ `utility_ratio` = bal_acc_synthetic / bal_acc_real) |
| Sharon (privacy) | `db.write_privacy_rows(df)`, df with the columns of `adult-copies.csv` | `privacy` |
| Alina (sketches) | `python -m sketches.sketch_service --dataset adult` (see `sketches/README.md`) | `sketches`, `sketch_freq` |
| Matilda (DGIM) | `db.write_dgim_snapshot(payload)`, next to `write_snapshot` | `dgim` (one point per tracked condition) |

Add `from storage import db` at the top and `influxdb-client` to your requirements.

## Results already saved as files

```bash
python scripts/import_results.py results/utility/*.csv results/privacy/*.csv
```
The script recognises utility, privacy and drift CSVs and DGIM JSON snapshots by their columns.

## Demo data

```bash
docker compose -f docker/docker-compose.yml up -d influxdb dashboard
python scripts/seed_dashboard_demo.py
```
