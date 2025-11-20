# Dockerfile for IEEE ETFA 2020 Unbalance Detection Approaches
# Base image with Python 3.11 for TensorFlow compatibility

FROM python:3.11-slim

# Build arguments
ARG VERSION=unknown
ARG BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

# Labels
LABEL org.opencontainers.image.title="Unbalanced Demo Apps"
LABEL org.opencontainers.image.description="Synchronized unbalance detection using CNN, FFT FCN, and Random Forest"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.source="https://github.com/kidproquo/unbalanced-demo-apps"
LABEL org.opencontainers.image.authors="Prince"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create data and output directories
RUN mkdir -p /app/data /app/figures/detections

# Install Python dependencies FIRST (for better layer caching)
# Copy only requirements.txt files first - they change less frequently
COPY approach_1_cnn/requirements.txt /tmp/requirements_cnn.txt
COPY approach_2_fft/requirements.txt /tmp/requirements_fft.txt
COPY approach_3_minimal_rfc/requirements.txt /tmp/requirements_rfc.txt

# Install dependencies (expensive operation - cache this layer)
RUN pip install --no-cache-dir -r /tmp/requirements_cnn.txt && \
    pip install --no-cache-dir -r /tmp/requirements_fft.txt && \
    pip install --no-cache-dir -r /tmp/requirements_rfc.txt && \
    rm /tmp/requirements_*.txt

# Copy models (large files, change occasionally)
COPY models/ /app/models/

# Copy shared utilities (change occasionally)
COPY utils/ /app/utils/

# Copy application code (changes frequently - copy last)
COPY approach_1_cnn/ /app/approach_1_cnn/
COPY approach_2_fft/ /app/approach_2_fft/
COPY approach_3_minimal_rfc/ /app/approach_3_minimal_rfc/
COPY data_coordinator.py /app/

# Copy data (commented out - mount as volume in docker-compose instead)
# COPY ../data/ /app/data/

# Set Python path to include /app
ENV PYTHONPATH=/app

# Default command (override in docker-compose)
CMD ["python", "-u", "data_coordinator.py", "--max-windows", "100"]
