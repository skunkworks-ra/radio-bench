#!/usr/bin/env bash
# Container image builder (Podman or Docker) and automated isolation / MCP smoke verifier.
# Resolves all directories and pack sources from bench.toml (+ bench.local.toml / env) -- never /tmp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_INIT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_SMOKE=0
VERIFIER_ONLY=0
SIDECAR_ONLY=0
PACK_ARG=""
CONFIG_ARG=""
WORK_DIR_ARG=""
DATA_DIR_ARG=""
RUNS_DIR_ARG=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --skip-smoke)
            SKIP_SMOKE=1
            shift
            ;;
        --verifier-only)
            VERIFIER_ONLY=1
            shift
            ;;
        --sidecar-only)
            SIDECAR_ONLY=1
            shift
            ;;
        --pack)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --pack" >&2
                exit 2
            fi
            PACK_ARG="$2"
            shift 2
            ;;
        --pack=*)
            PACK_ARG="${1#--pack=}"
            shift
            ;;
        --config)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --config" >&2
                exit 2
            fi
            CONFIG_ARG="$2"
            shift 2
            ;;
        --config=*)
            CONFIG_ARG="${1#--config=}"
            shift
            ;;
        --work-dir)
            WORK_DIR_ARG="$2"
            shift 2
            ;;
        --work-dir=*)
            WORK_DIR_ARG="${1#--work-dir=}"
            shift
            ;;
        --data-dir)
            DATA_DIR_ARG="$2"
            shift 2
            ;;
        --data-dir=*)
            DATA_DIR_ARG="${1#--data-dir=}"
            shift
            ;;
        --runs-dir)
            RUNS_DIR_ARG="$2"
            shift 2
            ;;
        --runs-dir=*)
            RUNS_DIR_ARG="${1#--runs-dir=}"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [ "$VERIFIER_ONLY" -eq 1 ] && [ "$SIDECAR_ONLY" -eq 1 ]; then
    echo "Cannot combine --verifier-only and --sidecar-only" >&2
    exit 2
fi

PYTHON_BIN="python"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 || ! PYTHONPATH="$REPO_ROOT_INIT/src:${PYTHONPATH:-}" "$PYTHON_BIN" -c "import radio_bench.config" >/dev/null 2>&1; then
    if [ -x "$REPO_ROOT_INIT/.pixi/envs/default/bin/python" ]; then
        PYTHON_BIN="$REPO_ROOT_INIT/.pixi/envs/default/bin/python"
    fi
fi

CFG_CMD=("$PYTHON_BIN" -m radio_bench.config --shell-env)
if [ -n "$CONFIG_ARG" ]; then
    CFG_CMD+=(--config "$CONFIG_ARG")
fi
if [ -n "$WORK_DIR_ARG" ]; then
    CFG_CMD+=(--work-dir "$WORK_DIR_ARG")
fi
if [ -n "$DATA_DIR_ARG" ]; then
    CFG_CMD+=(--data-dir "$DATA_DIR_ARG")
fi
if [ -n "$RUNS_DIR_ARG" ]; then
    CFG_CMD+=(--runs-dir "$RUNS_DIR_ARG")
fi
if [ -n "$PACK_ARG" ]; then
    CFG_CMD+=(--pack "$PACK_ARG")
fi
if [ "$VERIFIER_ONLY" -eq 0 ] || [ "$SKIP_SMOKE" -eq 0 ]; then
    CFG_CMD+=(--materialize-pack)
fi

eval "$(PYTHONPATH="$REPO_ROOT_INIT/src:${PYTHONPATH:-}" "${CFG_CMD[@]}")"

CONTAINER_CLI="${CONTAINER_CLI:-podman}"
RADIO_BENCH_ENGINE="${RADIO_BENCH_ENGINE:-$CONTAINER_CLI}"
RADIO_BENCH_PLATFORM="${RADIO_BENCH_PLATFORM:-}"

export TMPDIR="$CONTAINER_TMPDIR"
export CONTAINER_TMPDIR
export MPLCONFIGDIR
export CASASITECONFIG
export DOCKER_CONFIG
export DOCKER_HOST
export CONTAINER_CLI
export RADIO_BENCH_ENGINE
export RADIO_BENCH_PLATFORM

BUILD_FLAGS=()
if [ "$CONTAINER_CLI" = "podman" ]; then
    BUILD_FLAGS+=(--pull=missing)
elif [ "$CONTAINER_CLI" = "docker" ]; then
    export DOCKER_BUILDKIT=1
    export BUILDX_BUILDER=default
fi
if [ -n "$RADIO_BENCH_PLATFORM" ]; then
    BUILD_FLAGS+=(--platform "$RADIO_BENCH_PLATFORM")
fi

mkdir -p "$CONTAINER_TMPDIR" "$MPLCONFIGDIR"

STAGE_ROOT=""
cleanup() {
    if [ -n "$STAGE_ROOT" ] && [ -d "$STAGE_ROOT" ]; then
        rm -rf "$STAGE_ROOT"
    fi
}
trap cleanup EXIT

STAGE_ROOT="$(mktemp -d "$CONTAINER_TMPDIR/image_stage.XXXXXX")"

if [ "$VERIFIER_ONLY" -eq 0 ] && [ "$SIDECAR_ONLY" -eq 0 ]; then
    echo "[build_images] Staging CASA measures data from ${HOST_CASA_DATA}..."
    mkdir -p "$STAGE_ROOT/casa_ctx"
    cp -RL "$HOST_CASA_DATA" "$STAGE_ROOT/casa_ctx/casa-data"
    chmod -R a+rX "$STAGE_ROOT/casa_ctx/casa-data"

    echo "[build_images] Building localhost/radio-bench-casa-base:latest..."
    "$CONTAINER_CLI" build \
        ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"} \
        --target casa-base \
        -t localhost/radio-bench-casa-base:latest \
        -f "$REPO_ROOT/images/agent-base.Containerfile" \
        "$STAGE_ROOT/casa_ctx"

    rm -rf "$STAGE_ROOT/casa_ctx"

    echo "[build_images] Staging run-step.sh at ${RADIO_ANALYST_SHA} and building agent-base..."
    mkdir -p "$STAGE_ROOT/agent_ctx"
    git -C "$RADIO_ANALYST_REPO" show "${RADIO_ANALYST_FULL_SHA}:scripts/plugin/run-step.sh" \
        > "$STAGE_ROOT/agent_ctx/run-step.sh"
    chmod 0755 "$STAGE_ROOT/agent_ctx/run-step.sh"

    echo "[build_images] Building localhost/radio-bench-agent-base:latest..."
    "$CONTAINER_CLI" build \
        ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"} \
        --target agent-base \
        -t localhost/radio-bench-agent-base:latest \
        -f "$REPO_ROOT/images/agent-base.Containerfile" \
        "$STAGE_ROOT/agent_ctx"

    rm -rf "$STAGE_ROOT/agent_ctx"
fi

if [ "$VERIFIER_ONLY" -eq 0 ]; then
    echo "[build_images] Exporting pack ${PACK_ID} read-only via git archive at ${RADIO_ANALYST_SHA} (${RADIO_ANALYST_FULL_SHA})..."
    mkdir -p "$STAGE_ROOT/sidecar_ctx/radio-analyst-src"
    git -C "$RADIO_ANALYST_REPO" archive --format=tar "$RADIO_ANALYST_FULL_SHA" \
        | tar -xf - -C "$STAGE_ROOT/sidecar_ctx/radio-analyst-src"
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

    echo "[build_images] Building ${SIDECAR_IMAGE} (pack=${PACK_ID}, sha=${RADIO_ANALYST_SHA})..."
    "$CONTAINER_CLI" build \
        ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"} \
        --build-arg "RADIO_ANALYST_SHA=${RADIO_ANALYST_SHA}" \
        --build-arg "PACK_ID=${PACK_ID}" \
        -t "$SIDECAR_IMAGE" \
        -f "$REPO_ROOT/images/mcp-sidecar.Containerfile" \
        "$STAGE_ROOT/sidecar_ctx"

    if [ "$PACK_ID" = "$DEFAULT_PACK" ]; then
        "$CONTAINER_CLI" tag "$SIDECAR_IMAGE" localhost/radio-bench-mcp-sidecar:latest
    fi

    rm -rf "$STAGE_ROOT/sidecar_ctx"
fi

if [ "$SIDECAR_ONLY" -eq 0 ]; then
    echo "[build_images] Staging radio-bench verifier package..."
    mkdir -p "$STAGE_ROOT/verifier_ctx/radio-bench-src"
    cp "$REPO_ROOT/pyproject.toml" "$STAGE_ROOT/verifier_ctx/radio-bench-src/pyproject.toml"
    cp -L "$REPO_ROOT/README.md" "$STAGE_ROOT/verifier_ctx/radio-bench-src/README.md"
    cp "$REPO_ROOT/bench.toml" "$STAGE_ROOT/verifier_ctx/radio-bench-src/bench.toml"
    cp -r "$REPO_ROOT/configs" "$STAGE_ROOT/verifier_ctx/radio-bench-src/configs"
    cp -rL "$REPO_ROOT/suites" "$STAGE_ROOT/verifier_ctx/radio-bench-src/suites"
    cp -r "$REPO_ROOT/src" "$STAGE_ROOT/verifier_ctx/radio-bench-src/src"
    chmod -R a+rX "$STAGE_ROOT/verifier_ctx/radio-bench-src"

    echo "[build_images] Building localhost/radio-bench-verifier:latest..."
    "$CONTAINER_CLI" build \
        ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"} \
        -t localhost/radio-bench-verifier:latest \
        -f "$REPO_ROOT/images/verifier.Containerfile" \
        "$STAGE_ROOT/verifier_ctx"

    rm -rf "$STAGE_ROOT/verifier_ctx"
fi

echo "[build_images] Pruning superseded dangling ${CONTAINER_CLI} image layers..."
"$CONTAINER_CLI" image prune -f || true

if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "[build_images] Running automated container isolation and MCP HTTP smoke checks..."
    SMOKE_CMD=("$PYTHON_BIN" -m radio_bench.smoke_containers --pack "$PACK_ID")
    if [ -n "$CONFIG_ARG" ]; then
        SMOKE_CMD+=(--config "$CONFIG_ARG")
    fi
    (cd "$WORK_DIR" && PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" "${SMOKE_CMD[@]}")
fi

echo "[build_images] DONE."
