# SynAudit: Real-Time Quality and Privacy Monitoring for Streaming Synthetic Data

**Course:** Text, Audio, Image and Stream Mining, ELTE
**Team:** Matilda Migwi, Diana Grigoryan, g-mbira, Sharon Jepngetich, Alina Amanbekova

---

## Abstract

SynAudit generates synthetic versions of two tabular datasets (UCI Adult and UCI Default of Credit Card Clients), streams real and synthetic records through Kafka, and monitors the synthetic stream live for statistical quality, distribution drift, machine-learning utility and privacy risk. Results are stored in InfluxDB and shown in a Streamlit dashboard. *(Final results summary to be added after the end-to-end run.)*

## Contents

1. Introduction and objectives
2. System architecture
3. Environment setup
4. Data ingestion (Matilda)
5. Synthetic data generation and quality evaluation (Diana)
6. Drift detection (g-mbira)
7. ML utility and privacy (Sharon)
8. Storage, sketches and dashboard (Alina)
9. End-to-end execution
10. Results
11. Known issues and limitations
12. Conclusion and future work
13. References

---

## 1. Introduction and objectives

Synthetic data is used to share or test with data that resembles real records without exposing them. The risk is that synthetic data can be too poor to be useful or too close to the real data to be private. SynAudit checks both continuously on a stream instead of once on a static file.

Objectives:

- Generate synthetic Adult and Credit Default data with SDV.
- Stream real and synthetic records through Kafka using one shared message contract.
- Measure quality (statistical similarity), drift, ML utility (TSTR/TRTR) and privacy (exact-copy detection) on the stream.
- Store and visualise all metrics in one dashboard.

## 2. System architecture

*(Diagram and data-flow description to be added.)*

```
UCI ──► download_data.py ──► data/raw ──► preprocess.py ──► data/processed ──┬─► generate_synthetic.py ──► data/synthetic
                                                                               │
                                                     producer.py (real + synthetic) ──► Kafka topics
                                                                                           │
                       quality · drift · utility · privacy · sketches · DGIM consumers ◄──┘
                                                                                           │
                                                                InfluxDB ──► Streamlit dashboard
```

## 3. Environment setup

**Requirements:** Python 3.11, Docker Desktop with Docker Compose, Git.

All Python dependencies are in one file, `requirements.txt`, at the repository root. The same file is used by the local virtual environment and by every Docker image (`quality`, `sketches`, `dashboard`, `jobs`). All images use `python:3.11-slim`, so Docker builds the dependency layer once and shares it.

```bash
git clone https://github.com/sharonjepngetichsaina/SynAudit.git
cd SynAudit
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -c "import pandas, sdv, sdmetrics, river, kafka, influxdb_client, streamlit, sklearn, matplotlib; print('OK')"
```

Expected output: `OK`.

Notes:

- `sdv` installs PyTorch, so the first install takes 15–30 minutes on Windows.
- Earlier versions had five separate requirements files with conflicting Kafka clients (`kafka-python` and `kafka-python-ng`) and no `sdv`. These were merged into one file.

## 4. Data ingestion (Matilda)

Run instructions: `docs/DATA_INGESTION.md`.

*(Results in progress.)*

## 5. Synthetic data generation and quality evaluation (Diana)

Run instructions: `docs/DATA_SYNTHESIS_AND_QUALITY.md`.

Each synthetic dataset has 40,000 generated rows with no duplicates. An earlier version held a 10,000-row batch twice plus 20,000 new rows, and some services read a different, stale synthetic file. Both problems were fixed before this run.

## 6. Drift detection (g-mbira)

## 7. ML utility and privacy (Sharon)

## 8. Storage, sketches and dashboard (Alina)

## 9. End-to-end execution

Run every command from the repository root with the virtual environment active.
The order follows the data dependencies, not team ownership: each phase consumes
the output of the one before it. Component details are in `docs/`.

| Phase | Owner | Step | Details |
|---|---|---|---|
| 1 | Matilda | Prepare real data | `docs/DATA_INGESTION.md` |
| 2 | Diana | Generate synthetic data, batch quality | `docs/DATA_SYNTHESIS_AND_QUALITY.md` |
| 3 | Alina | Start infrastructure and services | `docker/docker-compose.yml` |
| 4 | Sharon, g-mbira | Utility, privacy, drift (batch, automatic) | `utility/`, `privacy/`, `drift/` |
| 5 | Matilda | Stream real and synthetic records | `docs/DATA_INGESTION.md` |
| 6 | Diana, Alina, Matilda | Quality, sketches, DGIM (streaming, automatic) | `quality/`, `sketches/`, `ingestion/` |
| 7 | Alina | Dashboard | http://localhost:8501 |

**Phase 1: prepare real data**

```bash
python ingestion/download_data.py
python ingestion/preprocess.py
```

**Phase 2: generate synthetic data and run batch quality**

```bash
python scripts/generate_synthetic.py --dataset adult --num-rows 40000 --seed 42
python scripts/generate_synthetic.py --dataset credit-default --num-rows 40000 --seed 42
python scripts/evaluate_quality.py --dataset adult
python scripts/evaluate_quality.py --dataset credit-default
```

**Phases 3–4: start everything**

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml ps
```

The utility, privacy and drift jobs read the CSVs directly and exit when done.

**Phase 5: stream** (one terminal each)

```bash
python ingestion/producer.py --dataset adult --source real --rate 50
python ingestion/producer.py --dataset adult --source synthetic --rate 50
python ingestion/producer.py --dataset credit-default --source real --rate 50
python ingestion/producer.py --dataset credit-default --source synthetic --rate 50
```

**Phases 6–7: watch**

- Kafka UI: http://localhost:8081
- Dashboard: http://localhost:8501

**Stop and reset**

```bash
docker compose -f docker/docker-compose.yml down       # keep data
docker compose -f docker/docker-compose.yml down -v    # wipe Kafka and InfluxDB data
```

## 10. Results

## 11. Known issues and limitations

## 12. Conclusion and future work

## 13. References

- Becker, B. and Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository.
- Yeh, I-C. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository.
- Patki, N., Wedge, R. and Veeramachaneni, K. (2016). *The Synthetic Data Vault*. IEEE DSAA.