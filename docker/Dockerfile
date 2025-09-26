# Debian-based micromamba (conda) image
FROM mambaorg/micromamba:1.5.8

LABEL org.opencontainers.image.title="MDPI Pipeline"
LABEL org.opencontainers.image.description="Dockerized MDPI end-to-end pipeline running run_pipeline.py"
LABEL org.opencontainers.image.source="https://github.com/your-org/your-repo"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MDPI_MODEL_DIR=/app/model \
    MDPI_INPUT_DIR=/app/input

WORKDIR /app

# Copy conda environment file and install into base env
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yml /tmp/environment.yml
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean -a -y

# Copy project files
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Run as non-root user to preserve host file ownership on bind mounts
USER $MAMBA_USER

# Run streamlined entrypoint that mirrors Streamlit defaults
ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "docker/entrypoint.py"]
