#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script - P2P NCCL XpYd Architecture
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# P2P NCCL communication. The architecture supports various XpYd configurations:
#
# - 1P1D: 1 Prefill server + 1 Decode servers (current default)
# - 3P1D: 3 Prefill servers + 1 Decode server
# - etc.
#
# Configuration can be customized via environment variables:
#   MODEL: Model to serve
#   TP_SIZE: Tensor parallelism size
#   PROFILER: Profiler file
#   PREFILL_GPUS: Comma-separated GPU IDs for prefill servers
#   DECODE_GPUS: Comma-separated GPU IDs for decode servers
#   PREFILL_PORTS: Comma-separated ports for prefill servers
#   DECODE_PORTS: Comma-separated ports for decode servers
#   PROXY_PORT: Proxy server port used to setup XpYd connection.
#   TIMEOUT_SECONDS: Server startup timeout
# =============================================================================

# Configuration - can be overridden via environment variables
# llama-3.1-8B-Instruct, llama_8b_a800_tp1_prefill
# Qwen2.5-14B-Instruct, Qwen_14b_a800_tp2_prefill
MODEL=${MODEL:-/home/fit/zhaijdzz/WORK/models/llama-3.1-8B-Instruct}
TP_SIZE=${TP_SIZE:-1}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL_NAME="$(basename -- "$MODEL")"
PROFILER=${PROFILER:-"${SCRIPT_DIR}/profiler/profile_${MODEL_NAME}_tp${TP_SIZE}_online.npy"}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-8192}
DO_PROFILING=${DO_PROFILING:-0} # 1=profiling, 0=no profiling

# Default 1P1D configuration (1 Prefill + 1 Decode)
PREFILL_GPUS=${PREFILL_GPUS:-"0"}
DECODE_GPUS=${DECODE_GPUS:-"1"}
PREFILL_PORTS=${PREFILL_PORTS:-20000}
DECODE_PORTS=${DECODE_PORTS:-21001}

# Check if the profiler file exists.
if [[ ! -f "$PROFILER" ]]; then
  if [[ "$DO_PROFILING" != "1" ]]; then
    echo "ERROR: profiler file not found: $PROFILER" >&2
    echo "       Set DO_PROFILING=1 to generate it, or set PROFILER to an existing file." >&2
    exit 1
  fi
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
echo "  Do Profiling: $DO_PROFILING"
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
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    pkill -9 -f "proxy.py"
    kill -- -$$            # negative PID  ==  "this whole process-group"
    wait                   # reap children so we don't leave zombies
    exit 0
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

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - prefill*.log: Prefill server logs"
    echo "  - decode*.log: Decode server logs"
    echo "  - proxy.log: Proxy server log"

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
        local kv_port=$((5500 + i))

        echo "  Prefill server $((i+1)): GPU $gpu_ids, Port $port, KV Port $kv_port"
        VLLM_WORKER_MULTIPROC_METHOD=fork \
        VLLM_PROFILER_PATH=$PROFILER \
        CUDA_VISIBLE_DEVICES=$gpu_ids \
        UCX_NET_DEVICES=all \
        vllm serve $MODEL \
        --port $port \
        --is-flowprefill \
        --num-runners 128 \
        --tensor-parallel-size $TP_SIZE \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        --enforce-eager \
        --trust-remote-code \
        --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\", \"kv_port\":\"$kv_port\"}" > ./log/prefill$((i+1)).log 2>&1 &
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
        local kv_port=$((5700 + i))

        echo "  Decode server $((i+1)): GPU $gpu_ids, Port $port, KV Port $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_ids \
        UCX_NET_DEVICES=all \
        vllm serve $MODEL \
        --port $port \
        --no-is-flowprefill \
        --tensor-parallel-size $TP_SIZE \
        --no-enable-prefix-caching \
        --no-enable-chunked-prefill \
        --enforce-eager \
        --trust-remote-code \
        --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\", \"kv_port\":\"$kv_port\"}" > ./log/decode$((i+1)).log 2>&1 &
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
            cleanup
            exit 1
        fi
    done

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server on port $PROXY_PORT..."
    python3 proxy.py \
    --port $PROXY_PORT \
    --prefiller-hosts $(printf 'localhost %.0s' "${PREFILL_PORT_ARRAY[@]}") \
    --prefiller-ports "${PREFILL_PORT_ARRAY[@]}" \
    --decoder-hosts $(printf 'localhost %.0s' "${DECODE_PORT_ARRAY[@]}") \
    --decoder-ports "${DECODE_PORT_ARRAY[@]}" &
    PROXY_PID=$!
    PIDS+=($PROXY_PID)
    # PIDS+=($!)

    # =============================================================================
    # Profiling
    # =============================================================================
    if [ "$DO_PROFILING" -eq 1 ]; then
        python ./profiler/profiling_online.py \
            --model $MODEL \
            --tp-size $TP_SIZE
        echo "Profiling done. Cleaning up..."
        cleanup

    # =============================================================================
    # Start serving
    # =============================================================================
    else
        echo ""
        echo "All servers are up. Starting serving..."
        wait $PROXY_PID
        cleanup
    fi
}

main
