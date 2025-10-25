FROM python:3.12-slim

WORKDIR /app
COPY ../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY .. /app
WORKDIR /app/midupdate
ENV PYTHONPATH=/app/midupdate
CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8081"]
