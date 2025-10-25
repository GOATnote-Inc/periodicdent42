FROM python:3.11-slim
WORKDIR /app
COPY ../../rust ./
RUN pip install maturin && maturin build --release -m pycore/pyproject.toml
