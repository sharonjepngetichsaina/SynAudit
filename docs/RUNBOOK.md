# Synthetic data generation and quality monitoring

Owner: Diana

Generates synthetic Adult and Credit Default datasets and evaluates their statistical and structural quality. Each dataset combines 10,000 original records with 30,000 generated records. The quality-monitoring component supports batch evaluation and streaming analysis, including the use of CluStream and DenStream to examine changes in data distributions and clustering structure.

---

## Setup

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running it

**1. Generate synthetic datasets.**

Generate 30,000 synthetic records for each dataset using the original processed Adult and Credit Default data.

```bash
python generate_synthetic.py
```

Each resulting dataset contains 40,000 records: 10,000 original and 30,000 generated.

**2. Run batch quality evaluation.**

Run the project's quality evaluation process to compare the synthetic datasets against their original counterparts.

```bash
python evaluate_quality.py
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
| Synthetic Adult dataset | Original and generated Adult records |
| Synthetic Credit Default dataset | Original and generated Credit Default records |
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

- Generated 30,000 additional records for each dataset.
- Batch quality evaluation results are available under `results/`.
- Docker Compose configuration validated successfully.
- Dashboard and quality-monitoring Docker images built successfully in the local environment.
- Python modules compiled successfully.
- Sketch tests passed: 4 tests.
- Full end-to-end Kafka processing and InfluxDB data flow remain subject to verification.
