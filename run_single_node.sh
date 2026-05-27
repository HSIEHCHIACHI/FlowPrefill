#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script - Single-Node P2P NIXL XpYd Architecture
# =============================================================================
# This script launches an XpYd (X prefill + Y decode) disaggregated PD setup
# on a SINGLE machine, with all servers communicating via NIXL. It supports
# arbitrary X and Y, for example:
#
# - 1P1D: 1 Prefill server + 1 Decode server
# - 2P2D: 2 Prefill servers + 2 Decode servers (default below)
# - 3P1D: 3 Prefill servers + 1 Decode server
# - etc.
#
# Each prefill / decode server is one vllm process on this host. GPUs and HTTP
# ports must be partitioned so servers don't overlap; the script does NOT
# verify that for you.
#
# For multi-host deployments use run_multi_node.sh instead.
#
# Environment variables:
#   MODEL                          Model to serve
#   TP_SIZE                        TP per server
#   PROFILER                       Profiler file
#   PREFILL_GPUS / DECODE_GPUS     Per-server GPU IDs. ';' between servers,
#                                  ',' between GPUs. E.g. 0,1;2,3
#   PREFILL_PORTS / DECODE_PORTS   Per-server HTTP ports, ',' separated
#   PROXY_PORT                     Proxy server port
#   TIMEOUT_SECONDS                Per-server startup timeout
#   LOG_ROOT / RUN_ID              Log root and per-run sub-dir
# =============================================================================

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
TP_SIZE=${TP_SIZE:-2}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL_NAME="$(basename -- "$MODEL")"
PROFILER=${PROFILER:-"${SCRIPT_DIR}/profiler/profile_${MODEL_NAME}_tp${TP_SIZE}.npy"}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-8192}

# Per-run log directory: log/<timestamp>_run_single_node/
LOG_ROOT=${LOG_ROOT:-"${SCRIPT_DIR}/log"}
RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)_run_single_node"}
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"

# Default 1P1D configuration (1 prefill server + 1 decode server)
PREFILL_GPUS=${PREFILL_GPUS:-0,1}
DECODE_GPUS=${DECODE_GPUS:-2,3}
PREFILL_PORTS=${PREFILL_PORTS:-20000}
DECODE_PORTS=${DECODE_PORTS:-21000}

# 2P2D configuration (2 prefill + 2 decode, each server uses 2 GPUs)
# ';' separates servers; ',' separates GPUs within a server.
# PREFILL_GPUS=${PREFILL_GPUS:-0,1;2,3} # 2P
# PREFILL_PORTS=${PREFILL_PORTS:-20000,20001}
# DECODE_GPUS=${DECODE_GPUS:-4,5;6,7} # 2D
# DECODE_PORTS=${DECODE_PORTS:-21000,21001}

PREFILL_NIXL_PORT=5500
DECODE_NIXL_PORT=7500
PREFILL_MEM_FRACTION_STATIC=0.8
DECODE_MEM_FRACTION_STATIC=0.85

# Check if the profiler file exists.
if [[ ! -f "$PROFILER" ]]; then
  echo "ERROR: profiler file not found: $PROFILER" >&2
  echo "       Set PROFILER to an existing file." >&2
  exit 1
fi

echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  TP Size: $TP_SIZE"
echo "  Profiler File: $PROFILER"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Proxy Port: $PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo "  Log Dir: $RUN_LOG_DIR"
echo ""

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    local rc=${1:-0}
    echo "Stopping everything…"
    trap - INT TERM EXIT       # prevent re-entrancy
    # Delegate to clean.sh (kills vllm/proxy/worker, frees ports, releases GPU,
    # clears NIXL/UCX shm). Pass every http/proxy port we may have used so it
    # can free them too.
    PROXY_PORT=$PROXY_PORT \
    EXTRA_PORTS="${PREFILL_PORTS//,/ } ${DECODE_PORTS//,/ }" \
        bash "$SCRIPT_DIR/clean.sh" local || true
    exit "$rc"
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

main() {
    check_num_gpus
    ensure_python_library_installed vllm

    mkdir -p "$RUN_LOG_DIR"

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM
    # EXIT trap is a safety net: even if main() returns or dies unexpectedly,
    # we still try to clean up.
    trap cleanup EXIT

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - $RUN_LOG_DIR/prefill*.log: Prefill server logs"
    echo "  - $RUN_LOG_DIR/decode*.log: Decode server logs"
    echo "  - $RUN_LOG_DIR/proxy.log: Proxy server log"

    # Parse GPU and port arrays
    IFS=';' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=';' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_ids=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local nixl_port=$((PREFILL_NIXL_PORT + i))

        echo "  Prefill server $((i+1)): GPU $gpu_ids, Port $port, Nixl Port $nixl_port"
        PYTHONPATH=/workspace/FlowPrefill \
        VLLM_WORKER_MULTIPROC_METHOD=fork \
        VLLM_PROFILER_PATH=$PROFILER \
        CUDA_VISIBLE_DEVICES=$gpu_ids \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port \
        UCX_TLS=^rc,^ud,^dc,^ib,^rdmacm \
        vllm serve $MODEL \
        --port $port \
        --is-flowprefill \
        --num-runners 128 \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $PREFILL_MEM_FRACTION_STATIC \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        --enforce-eager \
        --trust-remote-code \
        --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}" > "$RUN_LOG_DIR/prefill$((i+1)).log" 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================
    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_ids=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local nixl_port=$((DECODE_NIXL_PORT + i))

        echo "  Decode server $((i+1)): GPU $gpu_ids, Port $port, Nixl Port $nixl_port"
        PYTHONPATH=/workspace/FlowPrefill \
        CUDA_VISIBLE_DEVICES=$gpu_ids \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port \
        UCX_TLS=^rc,^ud,^dc,^ib,^rdmacm \
        vllm serve $MODEL \
        --port $port \
        --no-is-flowprefill \
        --tensor-parallel-size $TP_SIZE \
        --gpu-memory-utilization $DECODE_MEM_FRACTION_STATIC \
        --no-enable-prefix-caching \
        --no-enable-chunked-prefill \
        --enforce-eager \
        --trust-remote-code \
        --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}" > "$RUN_LOG_DIR/decode$((i+1)).log" 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            cleanup 1
        fi
    done

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server on port $PROXY_PORT..."
    python3 proxy.py \
    --port $PROXY_PORT \
    --prefiller-hosts $(printf '0.0.0.0 %.0s' "${PREFILL_PORT_ARRAY[@]}") \
    --prefiller-ports "${PREFILL_PORT_ARRAY[@]}" \
    --decoder-hosts $(printf '0.0.0.0 %.0s' "${DECODE_PORT_ARRAY[@]}") \
    --decoder-ports "${DECODE_PORT_ARRAY[@]}" > "$RUN_LOG_DIR/proxy.log" 2>&1 &
    PROXY_PID=$!
    PIDS+=($PROXY_PID)

    # =============================================================================
    # Start serving
    # =============================================================================
    echo ""
    echo "All servers are up. Starting serving..."
    wait $PROXY_PID
    cleanup
}

main
