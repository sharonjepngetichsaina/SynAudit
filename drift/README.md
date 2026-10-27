# Drift detection

Detects when the distribution of the synthetic data stream changes over time.

- Window level: Kolmogorov-Smirnov test (numeric) and chi-square test (categorical)
  on windows of 500 records against a reference sample
- Record level: ADWIN (Bifet & Gavaldà, 2007) and Page-Hinkley (Page, 1954)

Input: Kafka topic `synthetic-stream` · Output: InfluxDB measurement `drift_events`

Owner: Grace