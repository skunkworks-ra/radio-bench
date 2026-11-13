# Isolated FastMCP sidecar serving ms-inspect (8000), ms-modify (8001), and ms-create (8002)
# built strictly from pinned RADIO_ANALYST_SHA (00a11ea).
FROM localhost/radio-bench-casa-base:latest

LABEL org.radio-bench.radio-analyst-sha="00a11ea"

ENV RADIO_MCP_TRANSPORT=http \
    RADIO_MCP_HOST=0.0.0.0 \
    RADIO_MCP_DIR=/workspace/.radio_mcp \
    ANALYST_REV=00a11ea \
    RADIO_ANALYST_SHA=00a11ea

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
