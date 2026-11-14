# Stage 1: Shared offline CASA 6.7.6.14 + Python 3.12 + measures data layer.
# Reused by agent-base, mcp-sidecar, and verifier to avoid duplicating ~2 GB layers.
FROM docker.io/library/debian:trixie-slim AS casa-base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    CASASITECONFIG=/etc/casa/config.py \
    TMPDIR=/var/tmp \
    MPLCONFIGDIR=/var/tmp/mplconfig \
    PATH="/opt/casa-venv/bin:/usr/local/bin:/usr/bin:/bin"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git bash xz-utils tar procps util-linux \
        libgfortran5 libgomp1 libquadmath0 libstdc++6 libx11-6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /etc/casa /opt/casa-data /opt/uv-python /var/tmp/mplconfig /workspace \
    && chmod 1777 /var/tmp/mplconfig

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.12 \
    && uv venv --python 3.12 /opt/casa-venv \
    && uv pip install --no-cache --python /opt/casa-venv/bin/python \
        "casatools==6.7.6.14" \
        "casatasks==6.7.6.14" \
        "six>=1.16" \
        "numpy>=1.26" \
        "astropy>=6.0" \
        "scipy>=1.11" \
        "pandas>=2.1" \
        "matplotlib>=3.8" \
        "pillow>=10.0" \
        "pyyaml>=6.0" \
        "pydantic>=2.0" \
    && for lib in libssl.so.3 libldap.so.2 liblber.so.2 libcurl.so.4; do \
         if [ -e "/usr/lib/x86_64-linux-gnu/$lib" ]; then \
           ln -sf "/usr/lib/x86_64-linux-gnu/$lib" "/opt/casa-venv/lib/python3.12/site-packages/casatools/__casac__/lib/$lib"; \
         fi; \
       done \
    && chmod -R a+rX /opt/casa-venv /opt/uv-python

COPY casa-data /opt/casa-data

ENV PATH="/opt/casa-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

RUN printf 'measurespath = "/opt/casa-data"\ndatapath = ["/opt/casa-data"]\ndata_auto_update = False\nmeasures_auto_update = False\n' > /etc/casa/config.py \
    && printf 'export PATH="/opt/casa-venv/bin:$PATH"\nexport CASASITECONFIG="/etc/casa/config.py"\nexport TMPDIR="/var/tmp"\nexport MPLCONFIGDIR="/var/tmp/mplconfig"\n' > /etc/profile.d/casa-venv.sh \
    && chmod 0644 /etc/profile.d/casa-venv.sh \
    && ln -sfn /opt/casa-venv/bin/python /usr/local/bin/python \
    && ln -sfn /opt/casa-venv/bin/python3 /usr/local/bin/python3 \
    && /usr/sbin/useradd -m -u 1000 -s /bin/bash analyst \
    && mkdir -p /root/.casa /home/analyst/.casa \
    && ln -sfn /opt/casa-data /root/.casa/data \
    && ln -sfn /opt/casa-data /home/analyst/.casa/data \
    && cp /etc/casa/config.py /root/.casa/config.py \
    && cp /etc/casa/config.py /home/analyst/.casa/config.py \
    && chown -R analyst:analyst /home/analyst/.casa /workspace \
    && chmod -R a+rX /etc/casa \
    && cd /var/tmp && /opt/casa-venv/bin/python -c "import casatools, casatasks; me = casatools.measures(); assert me.observatory('VLA')" && rm -f /var/tmp/casa-*.log

# Stage 2: Hard-isolated agent-base image (zero ms_inspect/ms_modify/ms_create/radio_bench code).
FROM localhost/radio-bench-casa-base:latest AS agent-base

ARG CLAUDE_CODE_VERSION=2.1.325
ARG OPENCODE_VERSION=1.19.5

COPY run-step.sh /usr/local/bin/run-step.sh
RUN apt-get update && apt-get install -y --no-install-recommends unzip nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/radio-tools/scripts/plugin /opt/agent-tools \
    && cp /usr/local/bin/run-step.sh /opt/radio-tools/scripts/plugin/run-step.sh \
    && chmod 0755 /usr/local/bin/run-step.sh /opt/radio-tools/scripts/plugin/run-step.sh

RUN HOME=/opt/agent-tools bash -c 'curl -fsSL https://claude.ai/install.sh | bash -s -- "$1"' _ "${CLAUDE_CODE_VERSION}" \
    && ln -sfn /opt/agent-tools/.local/bin/claude /usr/local/bin/claude \
    && HOME=/opt/agent-tools OPENCODE_INSTALL_DIR=/opt/agent-tools/.opencode/bin \
        bash -c 'curl -fsSL https://opencode.ai/install | bash -s -- --version "$1"' _ "${OPENCODE_VERSION}" \
    && if [ -x /opt/agent-tools/.opencode/bin/opencode ]; then \
         ln -sfn /opt/agent-tools/.opencode/bin/opencode /usr/local/bin/opencode; \
       elif [ -x /opt/agent-tools/.local/bin/opencode ]; then \
         ln -sfn /opt/agent-tools/.local/bin/opencode /usr/local/bin/opencode; \
       fi \
    && chmod -R a+rX /opt/agent-tools \
    && /usr/local/bin/claude --version \
    && /usr/local/bin/opencode --version \
    && rm -rf /tmp/cc-socks /tmp/opencode /opt/agent-tools/.claude.json /root/.cache

WORKDIR /workspace
