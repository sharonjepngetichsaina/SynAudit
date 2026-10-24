# Data ingestion and infrastructure

Owner: Matilda

Prepares the datasets, runs the Kafka environment, and replays records into the
topics the rest of the system consumes. Also implements two stream algorithms:
reservoir sampling and DGIM.

**Read [`../docs/stream-contract.md`](../docs/stream-contract.md) before writing
a consumer.** It defines the topic names and the record format, and it is the
interface between this layer and every other service.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

**1. Get and prepare the data.** Downloads from UCI, then writes canonical CSVs.
Datasets are not committed, so both steps are needed on a fresh clone.

```bash
python ingestion/download_data.py     # -> data/raw/
python ingestion/preprocess.py        # -> data/processed/
```

**2. Start Kafka.** Broker in KRaft mode, four topics, web UI on
[localhost:8080](http://localhost:8080).

```bash
cd docker && docker compose up -d
docker compose logs kafka-init        # should list the four topics
```

**3. Replay a dataset into its topic.**

```bash
python ingestion/producer.py --dataset adult --source real --rate 50
```

`--rate` is records per second; `0` streams as fast as possible. `--limit N`
stops after N records. `--dry-run` prints records instead of sending them,
which is the quickest way to see the format.

**4. Run the stream statistics services** (each in its own terminal, while the
producer runs in another).

```bash
python ingestion/reservoir_service.py --dataset adult --source real --size 1000
python ingestion/dgim_service.py --dataset adult --source real --window 1000
```

## What this produces

| Output | Written by | Contents |
|---|---|---|
| `data/processed/<dataset>.csv` | `preprocess.py` | Canonical rows, the producer's input |
| `data/reference/<dataset>-<source>-reservoir.json` | `reservoir_service.py` | Uniform sample of the real stream |
| `data/reference/<dataset>-<source>-dgim.json` | `dgim_service.py` | Sliding-window rates |

Snapshots are rewritten periodically and replaced atomically, so a reader never
sees a half-written file.

## Preprocessing decisions

- Column names lowercased with underscores. Credit Default ships with
  placeholder headers `X1`–`X23`, renamed from the dataset documentation.
  There is no `pay_1`; the original data skips it.
- Missing values become `null`. Adult encodes them as `?`. Left as a string they
  would look like a legitimate category to a drift detector.
- `fnlwgt` dropped from Adult: a census sampling weight, not a feature.
- Adult's `income` labels stripped of a trailing full stop. The UCI download
  combines train and test rows and the test rows read `<=50K.`, which would
  otherwise give four classes instead of two.
- **No imputation, no encoding, no deduplication.** Preprocessing that changes
  the distribution would mean the generator learns our preprocessing and drift
  detection measures our artefacts.

## Algorithms

**Reservoir sampling** (Vitter 1985, Algorithm R) — uniform sample of fixed size
over a stream of unknown length, one pass, O(k) memory. `reservoir.py` has no
Kafka dependency so it can be tested statistically; `test_reservoir.py` verifies
uniformity with a chi-square test.

**DGIM** (Datar, Gionis, Indyk, Motwani 2002) — counts 1s in a sliding window in
O(log² N) bits. Used for rates: share of high-income records, share of defaults,
share of records with a missing value. `--buckets-per-size r` sets the accuracy
trade-off; the error bound is 1/(r−1).

```bash
python ingestion/test_reservoir.py
python ingestion/test_dgim.py
```

## Known limitations so far

These affect how other services should read this layer's output.

**The reference sample is uniform over all history, not over recent records.**
If the real stream drifts, the sample is a mixture of old and new distributions.
A quality drop measured against it may be drift rather than generator failure —
read it alongside the drift detectors, which can tell the two apart.

**It is the wrong tool for membership checks.** Exact-copy detection needs every
real record, not a sample: 1,000 records out of 48,842 would catch roughly 2% of
copies. Privacy checks should consume the full real stream directly. Something we
might need to discuss.

**DGIM rates carry 10–18% error at the default r=2.** Too noisy to see a shift of
a few % so `--buckets-per-size 6` brings it to about 7% for roughly
triple the buckets, which is negligible.

**Adult holds 6,374 exact duplicate rows (13%) after `fnlwgt` is dropped.** The
raw file has about 50; removing the near-unique weight column collapses otherwise
identical people together: something to look out for with bloom filters
