# Quality service: runbook (Diana)

Run everything from the project root (`...\Downloads\SynAudit`) with the venv active:
`& "C:\Users\Diana Grigoryan\Downloads\.venv313\Scripts\Activate.ps1"`

## 0. Install the extra packages (once)
```powershell
pip install -r quality\requirements.txt
```
If `import kafka` fails with a `six.moves` error on Python 3.13, run `pip install kafka-python-ng` instead.

## 1. Regenerate the synthetic data (fixed generator)
```powershell
python scripts\generate_synthetic.py --dataset adult --num-rows 10000 --seed 42
python scripts\generate_synthetic.py --dataset credit-default --num-rows 10000 --seed 42
```
The last lines print a **Self-check**. You want: zeros in `capital_gain` close to the real share
(about 92%), and `education -> education_num` consistent in 100% of rows.
Tell the team the CSVs changed (Matilda's producer replays them).

## 2. Batch evaluation (baseline)
```powershell
python scripts\evaluate_quality.py --dataset adult
python scripts\evaluate_quality.py --dataset credit-default
```
Results in `results\`, including `<dataset>-summary.json` and `<dataset>-sdmetrics.csv`.
Healthy adult output: `capital_gain` SMD well below 1 (was 4.66), zero gap near 0, consistency 1.0.

## 3. Check Kafka has data
```powershell
python ingestion\producer.py --dataset adult --source real --limit 100 --rate 50 --bootstrap localhost:29092
python ingestion\producer.py --dataset adult --source synthetic --limit 100 --rate 50 --bootstrap localhost:29092
python ingestion\producer.py --dataset credit-default --source real --limit 100 --rate 50 --bootstrap localhost:29092
python ingestion\producer.py --dataset credit-default --source synthetic --limit 100 --rate 50 --bootstrap localhost:29092
python quality\check_topics.py --bootstrap localhost:29092
```
`check_topics.py` prints the message count and one sample message per topic. If the consumer later says
"message does not match the dataset columns", send the sample message to me.

## 4. Run the streaming consumer
Terminal A (consumer, small window for a quick test):
```powershell
python quality\consumer.py --dataset adult --bootstrap localhost:29092 --window 100 --step 50 --sdmetrics-every 2 --idle-timeout 30
```
Terminal B (feed it):
```powershell
python ingestion\producer.py --dataset adult --source real --limit 1000 --rate 100 --bootstrap localhost:29092
python ingestion\producer.py --dataset adult --source synthetic --limit 1000 --rate 100 --bootstrap localhost:29092
```
One line is printed per window and appended to `results\stream\adult-windows.csv`.
Normal operation uses `--window 500 --step 100` (the defaults).

Without Kafka, same code on the CSV files: `python quality\consumer.py --dataset adult --from-csv --max-windows 20`

## 5. InfluxDB
Set `INFLUX_URL`, `INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET` (see `docs\INFLUX_SCHEMA.md`) and the consumer writes there too:
```powershell
$env:INFLUX_URL="http://localhost:8086"; $env:INFLUX_TOKEN="<token from Alina>"; $env:INFLUX_ORG="<org>"; $env:INFLUX_BUCKET="<bucket>"
python quality\consumer.py --dataset adult --bootstrap localhost:29092
```

## 6. Docker
Paste `docker\quality.compose.snippet.yml` into `docker\docker-compose.yml`, check the service names it mentions, then:
```powershell
cd docker
docker compose up -d --build quality-adult quality-credit-default
docker logs -f synaudit-quality-adult
cd ..
```

## 7. Push to GitHub
```powershell
git status
git add scripts quality docs docker data\synthetic results
git commit -m "Quality service: generator fixes, SDMetrics, streaming consumer, CluStream and DenStream"
git push
```

## What is not verified yet (check these when you run it)
- SDV fit with the zero-as-missing trick: step 1's Self-check shows whether it worked. If `capital_gain` zeros are still far from 92%, tell me and I will switch the approach.
- The `sdmetrics` report call, the real Kafka connection, InfluxDB writes and the Docker build were written to the library APIs but could not be run where this was built.
- CluStream and DenStream are written from the papers' algorithms with the simplifications listed at the top of `quality\clustering.py`. Cross-check parameters and the CluStream relevance-stamp rule against the paper before the report.
