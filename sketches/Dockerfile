# Built from the repository root (see docker/sketches.compose.snippet.yml)
FROM python:3.11-slim
WORKDIR /app
COPY sketches/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY storage/ /app/storage/
COPY sketches/ /app/sketches/
ENTRYPOINT ["python", "-m", "sketches.sketch_service"]
