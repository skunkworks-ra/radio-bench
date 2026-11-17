# Isolated FastMCP sidecar serving ms-inspect (8000), ms-modify (8001), and ms-create (8002)
# built strictly from configured radio-analyst pack source and ref.
FROM localhost/radio-bench-casa-base:latest

ARG RADIO_ANALYST_SHA=00a11ea
ARG PACK_ID=ra_00a11ea

LABEL org.radio-bench.radio-analyst-sha="${RADIO_ANALYST_SHA}" \
      org.radio-bench.pack-id="${PACK_ID}"

ENV RADIO_MCP_TRANSPORT=http \
    RADIO_MCP_HOST=0.0.0.0 \
    RADIO_MCP_DIR=/workspace/.radio_mcp \
    ANALYST_REV=${RADIO_ANALYST_SHA} \
    RADIO_ANALYST_SHA=${RADIO_ANALYST_SHA} \
    RADIO_BENCH_PACK_ID=${PACK_ID}

RUN uv pip install --no-cache --python /opt/casa-venv/bin/python \
        "mcp[cli]>=1.0,<2" \
        "bokeh>=3.4"

COPY radio-analyst-src /opt/radio-analyst
COPY serve_mcp_sidecar.py /usr/local/bin/serve-mcp-sidecar.py

RUN uv pip install --no-cache --no-deps --python /opt/casa-venv/bin/python /opt/radio-analyst \
    && chmod 0755 /usr/local/bin/serve-mcp-sidecar.py

USER analyst
EXPOSE 8000 8001 8002
WORKDIR /workspace
CMD ["/opt/casa-venv/bin/python", "/usr/local/bin/serve-mcp-sidecar.py"]
