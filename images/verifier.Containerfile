# Isolated post-run scientific verifier image with CASA 6.7.6.14 + radio_bench.grading.
FROM localhost/radio-bench-casa-base:latest

RUN uv pip install --no-cache --python /opt/casa-venv/bin/python \
        "httpx>=0.27" \
        "tomli-w>=1.0"

COPY radio-bench-src /opt/radio-bench

RUN uv pip install --no-cache --no-deps --python /opt/casa-venv/bin/python /opt/radio-bench

WORKDIR /workspace
