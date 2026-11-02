# Built from the repository root (see docker/storage.compose.snippet.yml)
FROM python:3.11-slim
WORKDIR /app
COPY dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY storage/ /app/storage/
COPY dashboard/ /app/dashboard/
WORKDIR /app/dashboard
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
