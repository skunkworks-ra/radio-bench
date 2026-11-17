#!/usr/bin/env bash
# Rootless Podman image builder and automated Wave 2 smoke verifier.
# Uses /mnt/fast/radio_bench_data/podman_tmp exclusively -- never /tmp.

set -euo pipefail

REPO_ROOT="/home/pjaganna/Software/radio-bench"
RADIO_ANALYST_REPO="/home/pjaganna/Software/radio-analyst"
RADIO_ANALYST_SHA="00a11ea"
HOST_CASA_DATA="/home/pjaganna/.casa/data"

SKIP_SMOKE=0
VERIFIER_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-smoke) SKIP_SMOKE=1 ;;
        --verifier-only) VERIFIER_ONLY=1 ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

export TMPDIR="/mnt/fast/radio_bench_data/podman_tmp"
export CONTAINER_TMPDIR="/mnt/fast/radio_bench_data/podman_tmp"
export MPLCONFIGDIR="/mnt/fast/radio-bench/mplconfig"
export CASASITECONFIG="/mnt/fast/radio-bench/casaconfig_local.py"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

STAGE_ROOT=""
cleanup() {
    if [ -n "$STAGE_ROOT" ] && [ -d "$STAGE_ROOT" ]; then
        rm -rf "$STAGE_ROOT"
    fi
}
trap cleanup EXIT

STAGE_ROOT="$(mktemp -d "$TMPDIR/image_stage.XXXXXX")"

if [ "$VERIFIER_ONLY" -eq 0 ]; then
    echo "[build_images] Staging CASA measures data..."
    mkdir -p "$STAGE_ROOT/casa_ctx"
    cp -a "$HOST_CASA_DATA" "$STAGE_ROOT/casa_ctx/casa-data"
    chmod -R a+rX "$STAGE_ROOT/casa_ctx/casa-data"

    echo "[build_images] Building localhost/radio-bench-casa-base:latest..."
    podman build \
        --pull=missing \
        --target casa-base \
        -t localhost/radio-bench-casa-base:latest \
        -f "$REPO_ROOT/images/agent-base.Containerfile" \
        "$STAGE_ROOT/casa_ctx"

    rm -rf "$STAGE_ROOT/casa_ctx"

    echo "[build_images] Staging run-step.sh at ${RADIO_ANALYST_SHA} and building agent-base..."
    mkdir -p "$STAGE_ROOT/agent_ctx"
    git -C "$RADIO_ANALYST_REPO" show "${RADIO_ANALYST_SHA}:scripts/plugin/run-step.sh" \
        > "$STAGE_ROOT/agent_ctx/run-step.sh"
    chmod 0755 "$STAGE_ROOT/agent_ctx/run-step.sh"

    echo "[build_images] Building localhost/radio-bench-agent-base:latest..."
    podman build \
        --pull=missing \
        --target agent-base \
        -t localhost/radio-bench-agent-base:latest \
        -f "$REPO_ROOT/images/agent-base.Containerfile" \
        "$STAGE_ROOT/agent_ctx"

    rm -rf "$STAGE_ROOT/agent_ctx"

    echo "[build_images] Cloning radio-analyst read-only at ${RADIO_ANALYST_SHA}..."
    mkdir -p "$STAGE_ROOT/sidecar_ctx"
    git clone --no-hardlinks --quiet "$RADIO_ANALYST_REPO" "$STAGE_ROOT/sidecar_ctx/radio-analyst-src"
    git -C "$STAGE_ROOT/sidecar_ctx/radio-analyst-src" checkout --quiet --detach "$RADIO_ANALYST_SHA"
    ACTUAL_SHA="$(git -C "$STAGE_ROOT/sidecar_ctx/radio-analyst-src" rev-parse HEAD)"
    if [[ "$ACTUAL_SHA" != "${RADIO_ANALYST_SHA}"* ]]; then
        echo "Pinned SHA mismatch: expected prefix ${RADIO_ANALYST_SHA}, got ${ACTUAL_SHA}" >&2
        exit 1
    fi
    rm -rf \
        "$STAGE_ROOT/sidecar_ctx/radio-analyst-src/.git" \
        "$STAGE_ROOT/sidecar_ctx/radio-analyst-src/docs" \
        "$STAGE_ROOT/sidecar_ctx/radio-analyst-src/design_docs" \
        "$STAGE_ROOT/sidecar_ctx/radio-analyst-src/talk" \
        "$STAGE_ROOT/sidecar_ctx/radio-analyst-src/tests" \
        "$STAGE_ROOT/sidecar_ctx/radio-analyst-src/STATE.md" \
        "$STAGE_ROOT/sidecar_ctx/radio-analyst-src/WORKLOG.md"
    chmod -R a+rX "$STAGE_ROOT/sidecar_ctx/radio-analyst-src"
    cp "$REPO_ROOT/images/serve_mcp_sidecar.py" "$STAGE_ROOT/sidecar_ctx/serve_mcp_sidecar.py"
    chmod 0755 "$STAGE_ROOT/sidecar_ctx/serve_mcp_sidecar.py"

    echo "[build_images] Building localhost/radio-bench-mcp-sidecar:latest..."
    podman build \
        --pull=missing \
        -t localhost/radio-bench-mcp-sidecar:latest \
        -f "$REPO_ROOT/images/mcp-sidecar.Containerfile" \
        "$STAGE_ROOT/sidecar_ctx"

    rm -rf "$STAGE_ROOT/sidecar_ctx"
fi

echo "[build_images] Staging radio-bench verifier package..."
mkdir -p "$STAGE_ROOT/verifier_ctx/radio-bench-src"
cp "$REPO_ROOT/pyproject.toml" "$STAGE_ROOT/verifier_ctx/radio-bench-src/pyproject.toml"
cp -L "$REPO_ROOT/README.md" "$STAGE_ROOT/verifier_ctx/radio-bench-src/README.md"
cp -r "$REPO_ROOT/src" "$STAGE_ROOT/verifier_ctx/radio-bench-src/src"
chmod -R a+rX "$STAGE_ROOT/verifier_ctx/radio-bench-src"

echo "[build_images] Building localhost/radio-bench-verifier:latest..."
podman build \
    --pull=missing \
    -t localhost/radio-bench-verifier:latest \
    -f "$REPO_ROOT/images/verifier.Containerfile" \
    "$STAGE_ROOT/verifier_ctx"

rm -rf "$STAGE_ROOT/verifier_ctx"

echo "[build_images] Pruning superseded dangling Podman image layers..."
podman image prune -f || true

if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "[build_images] Running automated Wave 2 isolation and 61-tool HTTP smoke checks..."
    PYTHON_BIN="python"
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 || ! "$PYTHON_BIN" -c "import radio_bench" >/dev/null 2>&1; then
        PYTHON_BIN="/mnt/fast/radio-bench/.pixi/envs/default/bin/python"
    fi
    (cd "$TMPDIR" && "$PYTHON_BIN" -m radio_bench.smoke_containers)
fi

echo "[build_images] DONE."
