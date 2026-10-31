# Stream sketches (Alina)

Two synopsis algorithms from the Stream Mining lectures (L2), implemented from the papers:

| Algorithm | Paper | Question it answers in SynAudit |
|---|---|---|
| Count-Min Sketch | Cormode and Muthukrishnan (2005) | Does each category appear as often in the synthetic stream as in the real one? |
| HyperLogLog | Flajolet, Fusy, Gandouet and Meunier (2007) | Does the synthetic stream have as many distinct values per column as the real one? |

Both keep fixed memory however long the stream runs. Exact counts are kept beside them only to
measure their error for the report.

| File | What it is |
|---|---|
| `count_min.py`, `hyperloglog.py` | the two algorithms |
| `tracker.py` | one sketch of each kind per categorical column and stream |
| `sketch_service.py` | reads `<dataset>-real` and `<dataset>-synthetic` from Kafka (group `synaudit-sketches`), writes `sketches` and `sketch_freq` to InfluxDB |
| `evaluate.py` | accuracy vs memory for several sketch sizes, saved to `results/sketches/` |
| `test_sketches.py` | checks the guarantees from the papers |

```bash
python -m sketches.test_sketches
python -m sketches.sketch_service --dataset adult --from-csv          # without Kafka
python -m sketches.sketch_service --dataset adult --bootstrap localhost:29092
python -m sketches.evaluate --dataset adult
```
Defaults: Count-Min width 272, depth 5 (epsilon 0.01, delta 0.007); HyperLogLog precision 10 (1,024 registers, standard error 3.3%).
