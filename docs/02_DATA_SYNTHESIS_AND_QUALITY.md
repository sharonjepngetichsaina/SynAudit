# Data synthesis and quality evaluation

Owner: Diana

Generates synthetic Adult and Credit Default datasets and evaluates their statistical and structural quality. Each synthetic dataset contains 40,000 generated records with no duplicate rows. The quality-monitoring component supports batch evaluation and streaming analysis, including the use of CluStream and DenStream to examine changes in data distributions and clustering structure.

---

## Setup

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Git Bash:

```bash
source .venv/Scripts/activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies. Generation needs `sdv`, which lives in the host-only
`requirements-generation.txt` (it also installs everything in `requirements.txt`):

```bash
pip install -r requirements-generation.txt
```

## Running it

**0. Prerequisite: real data.**

The generator reads `data/processed/<dataset>.csv`, created by the ingestion step:

```bash
python ingestion/download_data.py
python ingestion/preprocess.py
```

**1. Generate synthetic datasets.**

```bash
python scripts/generate_synthetic.py --dataset adult --num-rows 40000 --seed 42
python scripts/generate_synthetic.py --dataset credit-default --num-rows 40000 --seed 42
```

**2. Run batch quality evaluation.**

Run the project's quality evaluation process to compare the synthetic datasets against their original counterparts.

```bash
python scripts/evaluate_quality.py --dataset adult
python scripts/evaluate_quality.py --dataset credit-default
```

**3. Start the monitoring infrastructure.**

From the project root:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

**4. Check the running services.**

```bash
docker compose -f docker/docker-compose.yml ps
```

View the quality-monitoring logs:

```bash
docker compose -f docker/docker-compose.yml logs -f quality-adult
docker compose -f docker/docker-compose.yml logs -f quality-credit-default
```

---

## What this produces

| Output | Contents |
|---|---|
| `data/synthetic/adult-synthetic.csv` | 40,000 generated Adult records, no duplicates |
| `data/synthetic/credit-default-synthetic.csv` | 40,000 generated Credit Default records, no duplicates |
| `adult-numeric-quality.csv` | Numerical quality evaluation |
| `adult-categorical-quality.csv` | Categorical quality evaluation |
| `adult-correlation-difference.csv` | Correlation differences |
| `adult-consistency.csv` | Data consistency evaluation |
| `adult-sdmetrics.csv` | SDMetrics evaluation |
| `adult-summary.json` | Summary of Adult quality results |
| Credit Default evaluation files | Numerical, correlation, SDMetrics, and summary results |

Batch evaluation files are stored in `results/`.

---

## Streaming algorithms

**CluStream** — an online clustering algorithm for evolving data streams. It maintains compact micro-cluster summaries and helps identify changes in data distributions. In SynAudit, it provides clustering-based monitoring indicators, including cluster counts, centroid distances, membership-distribution differences, and coverage.

**DenStream** — a density-based clustering algorithm for evolving data streams. It maintains potential and outlier micro-clusters to represent dense regions and emerging patterns. Its monitoring indicators include potential-cluster and outlier counts, as well as clustering coverage.

Both algorithms are part of the streaming quality-monitoring component. They complement statistical quality metrics by examining the evolving structure of incoming data.

---

## Quality monitoring

**Batch evaluation** compares synthetic datasets with original data using numerical, categorical, correlation, consistency, and SDMetrics-based evaluations.

**Streaming evaluation** processes incoming records in windows to monitor changes in data quality over time.

**Clustering-based monitoring** uses CluStream and DenStream to analyse changes in the structure and distribution of streaming records.

**Storage** supports CSV output and InfluxDB integration in the quality-monitoring implementation.

---

## Implementation and verification notes

- Generated 40,000 unique synthetic records for each dataset.
- Batch quality evaluation results are available under `results/`.
- Docker Compose configuration validated successfully.
- Dashboard and quality-monitoring Docker images built successfully in the local environment.
- Python modules compiled successfully.
- Sketch tests passed: 4 tests.
- Full end-to-end Kafka processing and InfluxDB data flow remain subject to verification.