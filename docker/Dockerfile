# ARM64 Linux image, the same CPU architecture as a Raspberry Pi 4/5.
# On an Apple Silicon Mac this runs natively (no emulation).
FROM python:3.11-slim-bookworm

# OpenCV needs a couple of shared libraries even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY samples/ samples/
ENV PYTHONPATH=/app/src
# Models and results are mounted from the host (see docker-compose.yml)
ENTRYPOINT ["python", "-m", "edgevision"]
CMD ["--help"]
