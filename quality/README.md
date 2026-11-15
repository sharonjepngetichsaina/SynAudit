# Data quality evaluation

Evaluates how closely the synthetic Adult and Credit Default datasets preserve the statistical properties and relationships of the original real datasets. Supports batch quality evaluation and streaming quality monitoring.

## Algorithm

The batch evaluator compares real and synthetic datasets using several statistical and structural quality metrics:

- **Numerical quality**: compares means, standard deviations, minimum and maximum values, and standardized mean differences (SMD).
- **Categorical quality**: compares category distributions using total variation distance (TVD).
- **Correlation difference**: measures how closely the synthetic data preserves relationships between numerical features.
- **Consistency checks**: checks selected relationships and constraints between columns.
- **SDMetrics**: evaluates column-level distributions and pairwise trends, producing an overall quality score.

The streaming component consumes data through Kafka and supports ongoing quality monitoring. CluStream and DenStream are used to examine changes in clustering structure and data distributions over time.

## Files

- `metrics.py`: calculates statistical quality metrics.
- `consumer.py`: consumes streaming data from Kafka.
- `monitor.py`: performs streaming quality monitoring.
- `clustering.py`: provides CluStream and DenStream-based analysis.
- `sinks.py`: handles monitoring outputs and storage.
- `../scripts/evaluate_quality.py`: runs batch quality evaluation for a selected dataset.
- `../scripts/generate_synthetic.py`: generates synthetic datasets.

## Running it

Install the quality evaluation dependencies:

```bash
pip install -r requirements.txt
```

Run batch evaluation:

```bash
python scripts/evaluate_quality.py --dataset adult
python scripts/evaluate_quality.py --dataset credit-default
```

To skip SDMetrics evaluation:

```bash
python scripts/evaluate_quality.py --dataset adult --no-sdmetrics
```

The evaluator uses the processed real dataset and its corresponding synthetic dataset from the project data directories.

For streaming monitoring, start the Docker services:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Check the service status and quality-monitoring logs:

```bash
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f quality-adult
docker compose -f docker/docker-compose.yml logs -f quality-credit-default
```

## Output

Batch evaluation reports are saved in `results/`.

Depending on the dataset and applicable checks, the evaluator produces:

- `<dataset>-numeric-quality.csv`: numerical feature comparisons.
- `<dataset>-categorical-quality.csv`: categorical distribution comparisons, when categorical columns are present.
- `<dataset>-correlation-difference.csv`: differences between real and synthetic feature correlations.
- `<dataset>-consistency.csv`: consistency-check results, when applicable.
- `<dataset>-sdmetrics.csv`: SDMetrics evaluation results, when enabled.
- `<dataset>-summary.json`: summary of the evaluation.

Streaming monitoring outputs are handled by the monitoring and sink components.

## Results so far

The following results were obtained from the current 40,000-row synthetic datasets using the batch evaluator:

| **Dataset** | **Synthetic rows** | **Mean absolute correlation difference** | **SDMetrics overall quality** |
|---|---:|---:|---:|
| Adult | 40,000 | 0.0582 | 0.9381 |
| Credit Default | 40,000 | 0.0890 | 0.8687 |

Both synthetic datasets contained zero duplicate rows in these evaluation runs. Results describe the current generated datasets and may change when the data is regenerated.

*Note: Batch evaluation has been executed successfully. Full end-to-end Kafka processing and InfluxDB data flow remain subject to verification.*