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

Dependencies are in two files at the repository root:

- `requirements.txt` — runtime dependencies for the services. Installed in the host virtual environment and in every Docker image (`quality`, `sketches`, `dashboard`, `jobs`), all on `python:3.11-slim`, so Docker builds the dependency layer once and shares it.
- `requirements-generation.txt` — the synthetic-data generator only (`sdv`, which pulls in PyTorch). Host-only; it is **not** installed in any Docker image, because no container generates data. Installing it also installs everything in `requirements.txt`.

```bash
git clone https://github.com/sharonjepngetichsaina/SynAudit.git
cd SynAudit
python -m venv .venv
source .venv/Scripts/activate            # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -r requirements-generation.txt   # host: runtime + generator (sdv)
python -c "import pandas, sdv, sdmetrics, river, kafka, influxdb_client, streamlit, sklearn, matplotlib; print('OK')"
```

Expected output: `OK`.

Notes:

- `sdv` pulls in PyTorch, so the first host install takes several minutes. The Docker images do not install it, which keeps them small and the build fast.
- Earlier versions had five separate requirements files with conflicting Kafka clients (`kafka-python` vs `kafka-python-ng`). These were consolidated, and the generation-only `sdv` dependency was separated so the containers don't carry PyTorch.

## 4. Data ingestion (Matilda)

Run instructions: `docs/01_DATA_INGESTION.md`.

*(Results in progress.)*

## 5. Synthetic data generation and quality evaluation (Diana)

Run instructions: `docs/02_DATA_SYNTHESIS_AND_QUALITY.md`.

Each synthetic dataset has 40,000 generated rows with no duplicates. An earlier version held a 10,000-row batch twice plus 20,000 new rows, and some services read a different, stale synthetic file. Both problems were fixed before this run.

### 5.1 Generation method

The generator fits SDV's `GaussianCopulaSynthesizer` on the processed real data. A Gaussian copula models each column's marginal distribution separately and links them through a correlation matrix of normal variables. Two Adult-specific problems are handled before fitting:

- **Zero-inflated columns.** `capital_gain` and `capital_loss` are mostly zeros. A copula smooths the spike at 0 into a continuous distribution, which inflates the mean. Zeros are treated as missing during fitting, so the model learns the zero/non-zero decision separately, and they are restored to 0 after sampling.
- **Deterministic column pairs.** `education_num` is a fixed function of `education`. Modelled independently, the pair can contradict itself (for example `HS-grad` with `education_num` 14). The mapping is learned from the real data, `education_num` is dropped before fitting, and it is derived from `education` after sampling.
- **Invalid categorical combinations (Adult).** The copula reproduces each column's shares but links columns through one correlation matrix, so without a constraint it generates impossible rows such as a female `Husband` or a `Husband` who is not married. An SDV `FixedCombinations` constraint on (`relationship`, `sex`, `marital_status`) merges those columns into one before fitting, so only combinations present in the real data can be sampled.
- **Integer status codes (Credit Default).** `pay_0`, `pay_2`..`pay_6` are repayment-status codes with a large spike at 0 ("paid on time"). Modelled as continuous numbers the spike is smeared (52% of real rows at 0 became 31% synthetic). They are modelled as categorical in the fit metadata, so each code's frequency is matched; the saved metadata keeps them numeric, so downstream services are unaffected.
- **Heavy-tailed money columns (Credit Default).** `bill_amt*` use a `truncnorm` marginal bounded to the real range, which stops the over-spread of the default fit. `pay_amt*` are a spike of zeros (~20%) plus a heavy tail, so they use the same zero-inflation path as `capital_gain`. Two other approaches were tried and rejected: a signed-log transform (its convex inverse amplified tail errors into values 10-40x too large) and a KDE marginal (needs an N x N matrix and exhausts memory at this data size).
- **Duplicates.** Rows are sampled independently, so identical rows can occur by chance. The generator removes duplicates and samples again until it has the requested number of unique rows. Synthetic rows that happen to match a real record are deliberately kept, so the privacy check (section 7) can detect them.

### 5.2 Generation results: Adult

`python scripts/generate_synthetic.py --dataset adult --num-rows 40000 --seed 42`

| Check | Real | Synthetic |
|---|---|---|
| Rows × columns | 48,842 × 14 | 40,000 × 14 |
| Duplicate synthetic rows | — | 0 |
| `capital_gain`: share of zeros | 91.7% | 91.6% |
| `capital_gain`: mean | 1,079.1 | 1,088.0 |
| `capital_loss`: share of zeros | 95.3% | 95.3% |
| `capital_loss`: mean | 87.5 | 88.3 |
| `education` → `education_num` consistent | 100% | 100% |

The zero shares match within 0.1 percentage points, and the means are within 1%. Before the zero-inflation fix, the synthetic mean of `capital_gain` was about 35,784, roughly 33 times the real value.

### 5.3 Generation results: Credit Default

`python scripts/generate_synthetic.py --dataset credit-default --num-rows 40000 --seed 42`

| Check | Real | Synthetic |
|---|---|---|
| Rows × columns | 30,000 × 24 | 40,000 × 24 |
| Duplicate synthetic rows | — | 0 |
| `pay_2` share at code 0 | 52.4% | 52.2% |
| `pay_amt1` mean / zero share | 5,664 / 17.5% | 5,534 / 17.4% |
| `bill_amt1` mean / std | 51,223 / 73,636 | 51,224 / 73,676 |

Credit Default needed its own fixes, different from Adult's, because its columns differ: integer status codes and heavy-tailed money amounts rather than text categories with logical rules. The `pay_*` codes are modelled as categorical, and the money columns use `truncnorm` (`bill_amt*`) and the zero-inflation path (`pay_amt*`), as described in section 5.1. With those, no impossible negatives and no invalid codes appear, which was confirmed separately.

The synthetic set (40,000 rows) is larger than the real one (30,000). This is valid because rows are drawn from the fitted distribution rather than copied, but the privacy check in section 7 compares 40,000 synthetic rows against 30,000 real ones. We kept 40,000 rather than matching the real size for two reasons. The drift detectors (section 6) were calibrated on a 40,000-record synthetic stream. The utility check (section 7) already trims real and synthetic training sets to the same size, so the extra rows give the synthetic model no advantage.

### 5.4 Batch quality evaluation

`scripts/evaluate_quality.py` compares synthetic against real data on missing values, duplicates, numeric distributions, categorical distributions (total variation distance, TVD), correlation structure, logical consistency, zero-inflation and the SDMetrics quality report. Outputs are written to `results/`.

#### Adult

| Metric | Result |
|---|---|
| Missing-value rate: `workclass` / `occupation` / `native_country` | real 5.73% / 5.75% / 1.75% vs synthetic 5.70% / 5.64% / 1.67% |
| Duplicate rows | real 6,374 (13.0%), synthetic 0 |
| Numeric standardized mean difference (max) | 0.018 (`hours_per_week`) |
| Categorical TVD (max) | 0.0075 (`education`) |
| Mean absolute correlation difference | 0.056 |
| `education` → `education_num` consistency | 100% |
| Zero-share gap: `capital_gain` / `capital_loss` | 0.001 / 0.000 |
| SDMetrics overall | **0.901** |
| SDMetrics column shapes | 0.972 |
| SDMetrics column pair trends | 0.829 |

**Interpretation.** The marginal distributions (each column on its own) are reproduced closely. All categories are present, every TVD is below 0.01, numeric means are within 0.02 standard deviations, and the missing-value rates match. Two weaknesses remain:

1. **Tails are truncated.** Real `capital_gain` reaches 99,999 (std 7,452), synthetic only 42,509 (std 3,801). In Adult, 99,999 is a top-code shared by a few hundred records. A Gaussian copula cannot reproduce an isolated cluster that far out, so the mean is preserved but extreme cases are under-represented.
2. **Dependencies between columns are weaker than the marginals.** Column pair trends (0.83) and the correlation difference (0.056) show that the joint structure is less faithful than the individual columns. A Gaussian copula encodes categories as numbers and cannot, on its own, express logical rules such as *relationship = Husband ⇒ sex = Male*.

**Invalid combinations before and after the `FixedCombinations` constraint:**

| Combination | Before | After | Real |
|---|---|---|---|
| `Husband` + `Female` | 13.06% | 0.01% | 0.00% |
| `Wife` + `Male` | 3.10% | 0.01% | 0.01% |
| Spouse (`Husband`/`Wife`) but not `Married-*` | 23.60% | 0.00% | 0.00% |

Without the constraint, about a quarter of synthetic rows described an unmarried "spouse" and one in eight a female "Husband". With the constraint, the synthetic rates match the real data, which contains the same 0.01% residual. The generator's self-check reports a valid `relationship + sex + marital_status` combination in 100% of rows.

The 13% duplicate rate in the real data comes from dropping `fnlwgt` (see `docs/01_DATA_INGESTION.md`). It matters for the privacy check: a synthetic row equal to a frequent real row is not necessarily a leak.

#### Credit Default

| Metric | Baseline (default fit) | After fixes |
|---|---|---|
| SDMetrics overall | 0.886 | 0.869 |
| SDMetrics column shapes | 0.853 | **0.899** |
| SDMetrics column pair trends | 0.920 | 0.839 |
| `pay_*` zero-code gap (max) | 0.24 | 0.008 |
| `pay_amt*` standardized mean difference (max) | 0.17 | 0.010 |
| `bill_amt1` std ratio (synthetic / real) | 1.5 | 1.00 |
| Mean absolute correlation difference | 0.083 | 0.089 |

**Interpretation.** The per-column fidelity improved substantially: the `pay_*` zero-code gap fell from 0.24 to 0.008, the `pay_amt*` means are now within 1%, and `bill_amt1`'s over-spread is gone. Column shapes rose from 0.85 to 0.90.

The trade-off is that column pair trends fell from 0.92 to 0.84, so the overall score is marginally below the baseline. Modelling `pay_*` as categorical fixes their marginal distribution but weakens how the copula represents their correlation with the other columns, and the repayment-status codes are correlated with default and with the money columns. The fixes were kept because the defects they remove (for example a status code showing 31% zeros where the real data has 52%) are visible on inspection and are exactly what the monitoring should not have to report as a generator failure, whereas the pair-trend change is subtler; and because categorical is the correct model for an ordinal code.

**Residual limitation.** The extreme tail of `pay_amt*` is compressed: the synthetic maximum is about 90,000 against a real maximum near 870,000. The mean and spread are correct, so a handful of very large payments are under-represented.

## 6. Drift detection (g-mbira)

## 7. ML utility and privacy (Sharon)

## 8. Storage, sketches and dashboard (Alina)

## 9. End-to-end execution

Run every command from the repository root with the virtual environment active.
The order follows the data dependencies, not team ownership: each phase consumes
the output of the one before it. Component details are in `docs/`.

| Phase | Owner | Step | Details |
|---|---|---|---|
| 1 | Matilda | Prepare real data | `docs/01_DATA_INGESTION.md` |
| 2 | Diana | Generate synthetic data, batch quality | `docs/02_DATA_SYNTHESIS_AND_QUALITY.md` |
| 3 | Alina | Start infrastructure and services | `docker/docker-compose.yml` |
| 4 | Sharon, g-mbira | Utility, privacy, drift (batch, automatic) | `utility/`, `privacy/`, `drift/` |
| 5 | Matilda | Stream real and synthetic records | `docs/01_DATA_INGESTION.md` |
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

**Phase 5: stream the data into Kafka.**

The producers replay the processed and synthetic CSVs into their topics.

On **Windows**, Docker Desktop's port proxy is unreliable for the Kafka protocol
under load: it drops metadata refreshes, which surface as `KafkaTimeoutError:
Failed to update metadata`. The robust way is to run the producers *inside* the
Docker network, where `kafka:9092` is a direct hop with no proxy. This streams
all four sources, one after another:

```bash
for ds in adult credit-default; do for src in real synthetic; do
  docker compose -f docker/docker-compose.yml run --rm producer --dataset $ds --source $src --rate 0; done; done
```
In case adult real fails to stream, run the following:
```bash
docker compose -f docker/docker-compose.yml run --rm producer --dataset adult --source real --rate 0
```

On **Linux/macOS** (or any host where the broker is directly reachable) the
producers can run from the host against `localhost:29092`:

```bash
python ingestion/producer.py --dataset adult --source real --rate 500 --bootstrap 127.0.0.1:29092
python ingestion/producer.py --dataset adult --source synthetic --rate 500 --bootstrap 127.0.0.1:29092
python ingestion/producer.py --dataset credit-default --source real --rate 500 --bootstrap 127.0.0.1:29092
python ingestion/producer.py --dataset credit-default --source synthetic --rate 500 --bootstrap 127.0.0.1:29092
```

`producer.py` pins the Kafka client's `api_version`, so it skips the broker
version auto-probe that otherwise fails through Docker Desktop's proxy on Windows
(`NoBrokersAvailable`).

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

- SDV prints `UserWarning: We strongly recommend saving the metadata using 'save_to_json'` during fitting. This refers to the internal training metadata. The metadata that other services read (`data/synthetic/<dataset>-metadata.json`) is saved. The warning does not affect the output.
- SDMetrics prints a `FutureWarning` that its single-table quality report is deprecated. The scores are unaffected; the import should be updated to `sdmetrics.reports.QualityReport` in a later version.
- The Gaussian copula truncates heavy tails: `capital_gain` (Adult) and `pay_amt*` (Credit Default) reach only a fraction of the real maximum, though their means and spreads are correct.
- It cannot enforce logical rules between categorical columns on its own; the Adult `relationship`/`sex`/`marital_status` rule is handled with a `FixedCombinations` constraint, but other latent rules may remain (section 5.4).
- For Credit Default, modelling `pay_*` as categorical improves their marginals at a cost to cross-column correlation (column pair trends 0.92 to 0.84). See section 5.4 for the reasoning behind keeping it.

## 12. Conclusion and future work

## 13. References

- Becker, B. and Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository.
- Yeh, I-C. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository.
- Patki, N., Wedge, R. and Veeramachaneni, K. (2016). *The Synthetic Data Vault*. IEEE DSAA.