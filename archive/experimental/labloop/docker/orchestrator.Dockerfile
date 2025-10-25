FROM python:3.11-slim
WORKDIR /app
COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY .. /app
CMD ["uvicorn", "labloop.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
