FROM python:3.11-slim

WORKDIR /app
COPY orchestrator/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY orchestrator /app/pilotkit/orchestrator
COPY configs /app/pilotkit/configs
COPY reports /app/pilotkit/reports
ENV PYTHONPATH=/app
CMD ["uvicorn", "pilotkit.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
