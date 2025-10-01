FROM python:3.11-slim
WORKDIR /app
COPY orchestrator/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY orchestrator /app/orchestrator
COPY cli /app/cli
RUN chmod +x /app/cli
ENV PYTHONPATH=/app
CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8080"]
