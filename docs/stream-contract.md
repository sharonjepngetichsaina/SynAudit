# SynAudit stream contract

**Version:** 0.1

**Owner:** Matilda (data ingestion and infrastructure)

**Status:** proposed — open for comment by the team, to be frozen before consumers are built

This document defines the Kafka topics and the record format that every SynAudit service
codes against. It is the interface between the ingestion layer and the quality, drift,
utility and privacy services. If something here changes, it changes for everyone, so
changes go through a pull request rather than a direct commit.

---

## 1. Topics

The architecture diagram shows two logical streams, `real-stream` and `synthetic-stream`.
Each dataset gets its own concrete pair of topics, because Adult and Credit Default have
unrelated schemas and a Kafka topic should carry one schema.

| Topic | Contents | Produced by |
|---|---|---|
| `adult-real` | Real records from the Adult (Census Income) dataset | Ingestion |
| `adult-synthetic` | Synthetic records generated from Adult | Ingestion |
| `credit-default-real` | Real records from Default of Credit Card Clients | Ingestion |
| `credit-default-synthetic` | Synthetic records generated from Credit Default | Ingestion |

Naming rules:

- Hyphens only. Kafka treats topic names that differ only by `.` versus `_` as colliding in
  its internal metrics namespace and logs a warning. One separator avoids the problem.
- Lowercase, no spaces.
- Pattern is `<dataset>-<source>`, so new datasets extend the scheme without renaming anything.

Topic configuration:

| Setting | Value | Reason |
|---|---|---|
| Partitions | 1 | Guarantees total ordering. Drift detectors, DGIM and reservoir sampling all assume a sequential stream; multiple partitions would make runs non-reproducible. Dataset sizes do not need parallelism. |
| Replication factor | 1 | Single-broker development cluster. |
| `retention.ms` | 7 days | Long enough that a service started days after a producer run can still replay the full stream. |

---

## 2. Record format

Serialization is UTF-8 JSON, one record per Kafka message.

The original dataset row goes in `payload`, unchanged. Everything the pipeline needs about
the record sits beside it in the envelope.

```json
{
  "schema_version": "1.0",
  "dataset": "adult",
  "source": "real",
  "record_id": "adult-real-000123",
  "seq": 123,
  "emitted_at": "2026-10-24T08:45:00.123456Z",
  "payload": {
    "age": 39,
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "hours_per_week": 40,
    "native_country": "United-States",
    "income": "<=50K"
  }
}
```

### Envelope fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Version of this contract. Lets the format change without silently breaking consumers. |
| `dataset` | string | `adult` or `credit-default`. Redundant with the topic name, but carried so results written to the database can be tagged by dataset without the consumer tracking which topic it read from. |
| `source` | string | `real` or `synthetic`. Essential for any consumer that merges both streams — privacy checks and quality comparison both need provenance. |
| `record_id` | string | Stable unique identity, `<dataset>-<source>-<6-digit seq>`. Used for exact-match privacy checks and for tracing a record end to end. |
| `seq` | integer | Position in its own stream, starting at 0. Windows defined on `seq` are reproducible across runs; windows defined on wall-clock time are not. |
| `emitted_at` | string | Producer-side timestamp, ISO 8601, UTC, microsecond precision. Distinct from the Kafka broker timestamp. Used for time-series storage and for measuring end-to-end latency. |
| `payload` | object | The dataset row. Column names exactly as in the canonical CSV. |

### Payload rules

- **Column names** come from the canonical preprocessed CSV in `data/processed/`, not from
  the raw download. Lowercase, underscores, no spaces or dots.
- **Missing values** are JSON `null`. Adult encodes missing as `?` in the raw file; ingestion
  converts these. Synthetic records must use `null` too, so that a missing value never looks
  like a distinct category to a drift detector.
- **Types** are preserved: numeric columns are JSON numbers, categorical columns are JSON
  strings. No column is sent as a string "42".
- **The target column is never dropped.** `income` for Adult, `default_payment_next_month`
  for Credit Default. Train-on-synthetic-test-on-real evaluation depends on it.
- **No encoding is applied at the producer.** One-hot encoding, scaling and similar are the
  consumer's business. Encoding at the producer would force every service to share one choice.

---

## 3. Consuming

Each service uses its own consumer group, so every service receives a full copy of the stream.
Within a group Kafka splits partitions between members; across groups each group gets
everything. Since topics have one partition, run one consumer per group.

| Service | Consumer group |
|---|---|
| Quality evaluation | `synaudit-quality` |
| Drift detection | `synaudit-drift` |
| ML utility | `synaudit-utility` |
| Privacy evaluation | `synaudit-privacy` |
| Ingestion statistics (reservoir, DGIM) | `synaudit-ingest-stats` |

Set `auto.offset.reset=earliest`. Without it, a consumer starting after a producer run
begins at the end of the topic and silently sees nothing.

---

## 4. Replay behaviour

The real and synthetic producers run as separate processes with independent rates. They are
not synchronised.

The consequence matters for everyone downstream: **comparisons must be window-based, not
record-paired.** No consumer can assume that real record *i* corresponds to synthetic record *i*,
because the two streams advance at their own speeds. Compare distributions over windows instead.
This also reflects a realistic deployment, where a generator and a live source would never be
in lockstep.

Producer rate is configurable (records per second) so that a run can be slowed down for
demonstration or sped up for experiments.

---

## 5. Open questions

- Confirm the team is on InfluxDB and Grafana. The earlier role list said PostgreSQL and
  Streamlit; the Oct 20 architecture document says InfluxDB and Grafana.
- Synthetic data is assumed to be generated offline and written to a file that the producer
  replays, rather than generated in-stream. Confirm with Diana.
- Window size for `seq`-based windows: to be set once drift detection and DGIM are in place.

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-10-24 | First draft: topics, envelope, consumer groups, replay rules. |

---

## 6. Reference sample semantics

The ingestion layer maintains a uniform reference sample of each real stream
(reservoir sampling, Vitter 1985 Algorithm R), written to
`data/reference/<dataset>-<source>-reservoir.json`. Services that compare
incoming records against real data read this file rather than holding the
stream themselves.

Consumers must know two things about it:

**It is uniform over all history, not over recent records.** Every record seen
so far is in the sample with probability k/n. If the real stream drifts, the
sample is a mixture of the old and new distributions, not a picture of the
current one.

**So a quality drop measured against it may be drift, not generator failure.**
Quality comparison and drift detection must be read together: a divergence from
the reference sample means either the synthetic data degraded or the real data
moved, and the reference sample alone cannot tell them apart.

Known alternatives, not implemented in this version: a sliding-window reservoir,
or biased reservoir sampling (Aggarwal 2006), which weights recent records
exponentially. Deferred so the baseline is a standard, well-understood algorithm.

Each snapshot carries `records_seen` and `inclusion_probability`, so a consumer
can tell how much history the sample covers.
