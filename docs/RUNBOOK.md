# Synthetic data generation and quality monitoring

Owner: Diana

Generates synthetic Adult and Credit Default datasets and evaluates their statistical and structural quality. The datasets contain 10,000 original records and 30,000 generated records each. Quality evaluation combines batch statistical comparisons with streaming monitoring, including CluStream and DenStream clustering algorithms. Results are written to CSV and can also be stored in InfluxDB for dashboard visualization.

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

The generator uses the processed Adult and Credit Default datasets to create 30,000 synthetic records for each dataset.

```bash
python scripts/generate_synthetic.py
```

The resulting datasets contain 40,000 records each, combining the original and generated records.

**2. Run batch quality evaluation.**

```bash
python scripts/evaluate_quality.py
```

The evaluation compares synthetic and original data using statistical similarity, distribution, correlation, and consistency metrics.

**3. Start the monitoring infrastructure.**

From the project root:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

This starts the Kafka environment, InfluxDB, quality-monitoring services, and dashboard.

**4. Check the running services.**

```bash
docker compose -f docker/docker-compose.yml ps
```

View the quality-monitoring logs:

```bash
docker compose -f docker/docker-compose.yml logs -f quality-adult
docker compose -f docker/docker-compose.yml logs -f quality-credit-default
```

**5. Open the dashboard.**

[http://localhost:8501](http://localhost:8501)

---

## What this produces

| Output | Written by | Contents |
|---|---|---|
| Synthetic Adult dataset | `scripts/generate_synthetic.py` | Original and generated Adult records |
| Synthetic Credit Default dataset | `scripts/generate_synthetic.py` | Original and generated Credit Default records |
| Batch quality results | `scripts/evaluate_quality.py` | Statistical and structural comparison metrics |
| `results/stream/*-windows.csv` | Streaming quality services | Quality metrics calculated over streaming windows |
| InfluxDB measurements | Quality-monitoring services | Stored quality metrics for dashboard use |
| Dashboard | `dashboard/` | Visualization of monitoring results |

---

## Streaming algorithms

**CluStream** — an online clustering algorithm designed for evolving data streams. It maintains compact micro-cluster summaries and supports the analysis of changing data distributions. In SynAudit, it is used to calculate clustering-based monitoring indicators, including cluster counts, centroid distances, membership-distribution differences, and coverage.

**DenStream** — a density-based clustering algorithm for evolving data streams. It maintains potential and outlier micro-clusters to represent dense regions and emerging patterns. In SynAudit, its monitoring indicators include potential-cluster and outlier counts, as well as clustering coverage.

Both algorithms are implemented in `quality/clustering.py` and used by `quality/monitor.py` during sliding-window evaluation. Their results complement the statistical quality metrics and provide an additional view of how synthetic data behaves over time.

---

## Quality monitoring

**Batch evaluation** compares synthetic datasets against their original counterparts using statistical similarity, correlation, distribution, and consistency measurements.

**Streaming evaluation** calculates quality metrics over successive data windows, allowing changes in the incoming data to be monitored.

**Clustering-based evaluation** uses CluStream and DenStream to examine the evolving structure and distribution of streaming records.

**Storage integration** supports CSV output and InfluxDB storage of monitoring results.

**Dashboard integration** provides a visual interface for exploring the stored quality metrics.

---

## Implementation and verification notes

- Adult and Credit Default synthetic datasets: 40,000 records each (10,000 original and 30,000 generated).
- Docker Compose configuration validated successfully.
- Dashboard and quality-monitoring Docker images built successfully.
- Python modules compiled successfully.
- Sketch tests passed: 4 tests.
- Dashboard and supporting services started successfully.
- Full end-to-end Kafka processing and InfluxDB data flow remain subject to verification.
