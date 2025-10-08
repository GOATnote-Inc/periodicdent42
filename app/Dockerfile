FROM python:3.12-slim

# Cache-busting argument (set to current timestamp to force rebuild)
ARG CACHE_BUST=unknown

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Note: configs/ directory expected at parent level (handled by Cloud Build context)
# For local builds: ensure configs/ is available or comment out lab/campaign features

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Note: Cloud Run manages health checks automatically
# No HEALTHCHECK needed - it conflicts with Cloud Run's native health checking

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

