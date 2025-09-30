# Debian-based micromamba (conda) image
FROM mambaorg/micromamba:1.5.8

LABEL org.opencontainers.image.title="MDPI Pipeline"
LABEL org.opencontainers.image.description="Dockerized MDPI end-to-end pipeline running run_pipeline.py"
LABEL org.opencontainers.image.source="https://github.com/Aerius01/MDPI.git"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MDPI_MODEL_DIR=/app/model \
    MDPI_INPUT_DIR=/app/input

WORKDIR /projects/MDPI

# Use micromamba to create and manage the environment
COPY docker/environment.yml /app/environment.yml
RUN micromamba create -f /app/environment.yml && \
    micromamba clean -a -y

# Copy the rest of the application code
COPY . /app

# The new entrypoint is the Flask server
ENTRYPOINT ["micromamba", "run", "-n", "mdpi-env", "python", "/app/docker/server.py"]
