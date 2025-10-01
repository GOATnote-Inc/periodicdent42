FROM python:3.12-slim

WORKDIR /app
COPY ../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY .. /app
WORKDIR /app/midupdate
ENV PYTHONPATH=/app/midupdate
CMD ["python", "training/train.py", "--config-path", "training/cfg", "--config-name", "train"]
