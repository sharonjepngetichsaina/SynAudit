# What the quality service writes to InfluxDB

One point per window, plus one point per column per window. Everything is a field or a
low-cardinality tag (window_id is a **field**, not a tag, on purpose).

## Measurement `synaudit_quality`
Tags: `dataset` (`adult` | `credit-default`)

| Field | Meaning | Best | Starting alert threshold |
|---|---|---|---|
| `window_id` | counter, 1, 2, 3... per dataset | | |
| `n_real`, `n_synth` | records in each window | | |
| `smd_mean`, `smd_max` | standardized mean difference of numeric columns | 0 | `smd_max > 0.5` |
| `tvd_mean`, `tvd_max` | total variation distance of categorical columns | 0 | `tvd_max > 0.1` |
| `corr_diff_mean` | mean abs. difference of the correlation matrices | 0 | `> 0.1` |
| `zero_gap_max` | worst gap in the share of zeros (zero-inflated columns) | 0 | `> 0.1` |
| `consistency_education_education_num`, `consistency_min` | share of synthetic rows whose column pair exists in real data (adult only) | 1 | `< 0.99` |
| `sdmetrics_overall`, `sdmetrics_column_shapes`, `sdmetrics_column_pair_trends` | official SDMetrics Quality Report; only every N-th window | 1 | `< 0.8` |
| `clustream_micro_real`, `clustream_micro_synth` | number of CluStream micro-clusters | | |
| `clustream_centroid_dist` | distance between matched macro-cluster centers (features scaled to 0-1) | 0 | |
| `clustream_membership_tvd` | TVD of how the real and synthetic windows fall into the real macro-clusters | 0 | |
| `clustream_coverage_ratio` | synthetic-to-real distance to nearest real micro-cluster; ~1 is good | 1 | `> 1.5` |
| `denstream_micro_real`, `denstream_micro_synth` | potential micro-clusters | | |
| `denstream_outliers_real`, `denstream_outliers_synth` | outlier micro-clusters | | |
| `denstream_macro_real`, `denstream_macro_synth` | density-based macro-clusters | | differ strongly |
| `denstream_centroid_dist`, `denstream_membership_tvd`, `denstream_coverage_ratio` | same ideas, DenStream micro-clusters | 0 / 0 / 1 | `coverage > 1.5` |

Thresholds are starting points from offline tests, tune them once the real stream runs.
Fields that could not be computed in a window (e.g. SDMetrics between runs) are omitted, not written as NaN.

## Measurement `synaudit_column_quality`
Tags: `dataset`, `column`, `kind` (`numeric` | `categorical`)
Fields: `smd` (numeric columns), `tvd` (categorical columns), `zero_gap` (zero-inflated numeric columns)

## Flux examples for Grafana
```flux
from(bucket: "synaudit")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "synaudit_quality" and r.dataset == "adult")
  |> filter(fn: (r) => r._field == "smd_max" or r._field == "consistency_min")
```
```flux
from(bucket: "synaudit")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "synaudit_column_quality" and r._field == "smd")
  |> filter(fn: (r) => r.dataset == "adult")
```
Settings the service reads from the environment: `INFLUX_URL`, `INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET`.
Without `INFLUX_URL` it only writes `results/stream/<dataset>-windows.csv`.
