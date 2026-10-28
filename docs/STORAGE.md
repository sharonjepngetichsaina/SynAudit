# Storage and dashboard (Alina)

InfluxDB stores every result the evaluators produce; the Streamlit dashboard shows them live.

| Service | URL | Login |
|---|---|---|
| Streamlit dashboard | http://localhost:8501 | — |
| InfluxDB web UI | http://localhost:8086 | admin / synaudit123 |

## Saving results from your service

```python
from storage import db

db.write_quality(batch_id=b, dataset="adult", metric="overall", score=0.87)                    # Diana
db.write_drift(window_id=w, dataset="adult", column="age", test="ks",
               statistic=0.21, p_value=0.003, alarm=True)                                       # Grace
db.write_utility(batch_id=b, dataset="adult", model="logreg",
                 accuracy=0.81, f1=0.62, roc_auc=0.86, utility_ratio=0.95)                      # Sharon
db.write_privacy(batch_id=b, dataset="adult", exact_copy_rate=0.0, median_dcr=1.4, mia_auc=0.51) # Sharon
```

Install `influxdb-client` in your service and set `INFLUX_URL=http://influxdb:8086` (plus
`INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET`, see `docker/storage.compose.snippet.yml`).
Running outside Docker, the defaults point to `http://localhost:8086`, so nothing needs to be set.
Set `RUN_ID` (e.g. `E1-baseline`) to tell experiments apart on the dashboard.

## Measurements

| Measurement | Tags (filters) | Fields (values) |
|---|---|---|
| quality | run_id, dataset, metric | score, batch_id |
| drift | run_id, dataset, column, test | statistic, p_value, alarm, window_id |
| utility | run_id, dataset, model | accuracy, f1, roc_auc, utility_ratio, batch_id |
| privacy | run_id, dataset | exact_copy_rate, median_dcr, dcr_train_share, mia_auc, batch_id |

## Demo data (dashboard without the pipeline)

```bash
docker compose -f docker/docker-compose.yml up -d influxdb dashboard
pip install influxdb-client
python scripts/seed_dashboard_demo.py
```
