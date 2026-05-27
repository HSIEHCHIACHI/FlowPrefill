#!/bin/bash

# =============================================================================
# FlowPrefill Disaggregated Serving Script - Multi-Node P2P NIXL XpYd Architecture
# =============================================================================
# This script launches an XpYd (X prefill + Y decode) disaggregated PD setup
# across one or more machines via SSH, using NIXL for KV transfer. It supports
# arbitrary X and Y; for example:
#
# - 1P1D across 2 machines  (default)
# - 2P2D across 2 machines  (each machine hosts 1 prefill + 1 decode)
# - 2P2D across 4 machines  (one server per machine)
# - 2P2D on a single machine (same host appears 4 times in HOSTFILE)
# - 3P1D, 4P2D, …
#
# Each entry in PREFILL_HOSTS / DECODE_HOSTS corresponds to ONE server. The
# same host may repeat (when several servers share a machine); when that
# happens each repeated server MUST use disjoint GPUs and a distinct HTTP port.
#
# For single-host-only deployments you can also use run_single_node.sh (no SSH).
#
# Environment variables:
#   MODEL                          Model to serve
#   TP_SIZE                        TP per server
#   PROFILER                       Profiler file (must exist on every node)
#   SSH_USER                       SSH user (default: root)
#   PREFILL_HOSTS / DECODE_HOSTS   Per-server hostnames/IPs, ';' separated.
#                                  One entry per server; same host may repeat.
#                                  E.g. PREFILL_HOSTS="10.1.50.99;10.1.50.99"
#   PREFILL_GPUS / DECODE_GPUS     Per-server GPU IDs. ';' between servers,
#                                  ',' between GPUs. E.g. "0,1;2,3"
#   PREFILL_PORT / DECODE_PORT     Per-server HTTP ports, ',' separated
#   PROXY_PORT                     Proxy server port (runs locally on the launcher)
#   TIMEOUT_SECONDS                Per-server startup timeout
#   SOCKET_IFNAME                  NIC for NCCL / Gloo / UCX (auto-detect)
#   LOG_ROOT / RUN_ID              Log root and per-run sub-dir (on each node)
# =============================================================================

ACTION="${1:-start}"
if [[ "${ACTION}" != "start" && "${ACTION}" != "stop" ]]; then
    echo "Usage: bash $(basename "$0") [start|stop]" >&2
    exit 1
fi

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
TP_SIZE=${TP_SIZE:-2}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL_NAME="$(basename -- "$MODEL")"
PROFILER=${PROFILER:-"${SCRIPT_DIR}/profiler/profile_${MODEL_NAME}_tp${TP_SIZE}.npy"}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-8192}
SSH_USER=${SSH_USER:-root}
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no)
SOCKET_IFNAME=${SOCKET_IFNAME:-auto}

# Remote working dir & log dir (mirror local layout on each remote node)
REMOTE_REPO_DIR=${REMOTE_REPO_DIR:-"${SCRIPT_DIR}"}
LOG_ROOT=${LOG_ROOT:-"${REMOTE_REPO_DIR}/log"}
RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)_run_multi_node"}
REMOTE_LOG_DIR="${LOG_ROOT}/${RUN_ID}"

# Per-server hostnames. ';' between servers; same host may repeat when several
# servers share one machine. The number of entries determines how many prefill
# / decode servers are launched.
# 1P1D: PREFILL_HOSTS=10.1.50.99            DECODE_HOSTS=10.1.51.76
# 2P2D: PREFILL_HOSTS=10.1.50.99;10.1.50.99 DECODE_HOSTS=10.1.51.76;10.1.51.76
PREFILL_HOSTS=${PREFILL_HOSTS:-10.1.50.99}
DECODE_HOSTS=${DECODE_HOSTS:-10.1.51.76}

# Per-server GPU IDs.
# ';' separates servers (one slot per host entry, in order),
# ',' separates GPUs inside one server. When two servers share a host,
# their GPU sets must NOT overlap.
PREFILL_GPUS=${PREFILL_GPUS:-0,1}
DECODE_GPUS=${DECODE_GPUS:-0,1}

# Per-server HTTP ports, ',' separated, one per host entry.
# When two servers share a host they MUST use different ports.
PREFILL_PORT=${PREFILL_PORT:-20000} # 2P: 20000,20001
DECODE_PORT=${DECODE_PORT:-21000} # 2D: 21000,21001

PREFILL_NIXL_PORT=5500
DECODE_NIXL_PORT=7500
PREFILL_MEM_FRACTION_STATIC=0.8
DECODE_MEM_FRACTION_STATIC=0.85

# Check if the profiler file exists (local check; remote nodes must mirror it).
if [[ ! -f "$PROFILER" ]]; then
  echo "ERROR: profiler file not found: $PROFILER" >&2
  echo "       Set PROFILER to an existing file (must exist on every remote node too)." >&2
  exit 1
fi

# Parse host / GPU / port arrays (';' for servers, ',' for ports/GPUs).
IFS=';' read -ra PREFILL_HOSTS     <<< "$PREFILL_HOSTS"
IFS=';' read -ra DECODE_HOSTS      <<< "$DECODE_HOSTS"
IFS=';' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
IFS=';' read -ra DECODE_GPU_ARRAY  <<< "$DECODE_GPUS"
IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORT"
IFS=',' read -ra DECODE_PORT_ARRAY  <<< "$DECODE_PORT"

PREFILL_NUM=${#PREFILL_HOSTS[@]}
DECODE_NUM=${#DECODE_HOSTS[@]}
HOSTS=("${PREFILL_HOSTS[@]}" "${DECODE_HOSTS[@]}")
REQUIRED_HOSTS=$((PREFILL_NUM + DECODE_NUM))
PROXY_HOST="${PREFILL_HOSTS[0]}"

if [[ "${#PREFILL_GPU_ARRAY[@]}" -lt "$PREFILL_NUM" || "${#PREFILL_PORT_ARRAY[@]}" -lt "$PREFILL_NUM" ]]; then
    echo "ERROR: PREFILL_GPUS / PREFILL_PORT must have at least $PREFILL_NUM entries (got ${#PREFILL_GPU_ARRAY[@]} / ${#PREFILL_PORT_ARRAY[@]})" >&2
    exit 1
fi
if [[ "${#DECODE_GPU_ARRAY[@]}" -lt "$DECODE_NUM" || "${#DECODE_PORT_ARRAY[@]}" -lt "$DECODE_NUM" ]]; then
    echo "ERROR: DECODE_GPUS / DECODE_PORT must have at least $DECODE_NUM entries (got ${#DECODE_GPU_ARRAY[@]} / ${#DECODE_PORT_ARRAY[@]})" >&2
    exit 1
fi

# All hostfile entries that participate in this run, deduplicated.
# Same host may appear multiple times (e.g. two prefill servers per host); we
# only need to ssh into each physical machine once for setup/cleanup.
UNIQ_HOSTS=()
declare -A _seen_host=()
for h in "${HOSTS[@]:0:$REQUIRED_HOSTS}"; do
    if [[ -z "${_seen_host[$h]:-}" ]]; then
        _seen_host[$h]=1
        UNIQ_HOSTS+=("$h")
    fi
done
unset _seen_host

echo "Warning: P2P NIXL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  TP Size: $TP_SIZE"
echo "  Profiler File: $PROFILER"
echo "  Prefill Hosts: ${PREFILL_HOSTS[*]}, GPUs: $PREFILL_GPUS, Ports: ${PREFILL_PORT_ARRAY[*]}"
echo "  Decode Hosts:  ${DECODE_HOSTS[*]}, GPUs: $DECODE_GPUS, Ports: ${DECODE_PORT_ARRAY[*]}"
echo "  Proxy: launcher host, Port: $PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo "  SSH User: $SSH_USER"
echo "  Socket Ifname: $SOCKET_IFNAME"
echo "  Run ID: $RUN_ID"
echo "  Log Dir (per host): $REMOTE_LOG_DIR"
echo ""

PIDS=()

ensure_python_library_installed_remote() {
    # Verify $1 importable on a given remote host.
    local host=$1
    local lib=$2
    echo "Checking if $lib is installed on $host ..."
    if ! ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "python3 -c 'import $lib'" > /dev/null 2>&1; then
        echo "$lib is not installed on $host. Please install it via pip install $lib."
        exit 1
    else
        echo "$lib is installed on $host."
    fi
}

check_num_gpus_remote() {
    # Check if the number of GPUs on $1 is >= the GPUs requested for that role.
    local host=$1
    local need_gpus=$2
    local num_gpus
    num_gpus=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
    if [[ -z "$num_gpus" || "$num_gpus" -lt "$need_gpus" ]]; then
        echo "ERROR: $host has only $num_gpus GPU(s); need >= $need_gpus." >&2
        exit 1
    else
        echo "Found $num_gpus GPUs on $host."
    fi
}

# Resolve a default SOCKET_IFNAME by asking host1 which NIC reaches a different host.
resolve_socket_ifname() {
    [[ "$SOCKET_IFNAME" == "auto" ]] || return 0
    local peer=""
    # Prefer a host that is NOT the same IP as HOSTS[0]; otherwise we'd be
    # routing to ourselves and get "dev lo".
    for h in "${HOSTS[@]:1}"; do
        if [[ "$h" != "${HOSTS[0]}" ]]; then
            peer="$h"
            break
        fi
    done
    if [[ -z "$peer" ]]; then
        # Only one unique host in the hostfile — no cross-node socket needed.
        # Fall back to the first non-loopback NIC.
        SOCKET_IFNAME=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@${HOSTS[0]}" \
            "ip -o -4 addr show scope global up | awk '{print \$2; exit}'")
    else
        SOCKET_IFNAME=$(ssh "${SSH_OPTS[@]}" "$SSH_USER@${HOSTS[0]}" \
            "ip -o route get $peer 2>/dev/null | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\") {print \$(i+1); exit}}'")
    fi
    if [[ -z "$SOCKET_IFNAME" || "$SOCKET_IFNAME" == "lo" ]]; then
        echo "ERROR: failed to auto-detect SOCKET_IFNAME on ${HOSTS[0]} (peer=${peer:-<none>})" >&2
        exit 1
    fi
    echo "Auto-detected SOCKET_IFNAME=$SOCKET_IFNAME (peer=${peer:-<single-host>})"
}

cleanup_host() {
    # Delegate to clean.sh on the remote host. clean.sh does:
    #   1. pkill vllm/proxy/worker patterns (safely, skipping ssh/bash itself)
    #   2. fuser -k -n tcp on configured ports
    #   3. fuser -k /dev/nvidia* to release stuck CUDA contexts
    #   4. clear leaked NIXL/UCX/vllm shm objects
    local host=$1
    # All possible http/proxy ports on this run (any of these may be in use).
    local extra_ports="${PREFILL_PORT_ARRAY[*]} ${DECODE_PORT_ARRAY[*]}"
    # -T: no pty; -o ServerAliveInterval=15: detect dead links.
    # Run clean.sh under setsid so a SIGINT propagated through our group does
    # not kill it mid-flight before it finishes wiping the remote node.
    ssh -T "${SSH_OPTS[@]}" -o ServerAliveInterval=15 "$SSH_USER@$host" "
        setsid -w bash -c '
            PROXY_PORT=$PROXY_PORT \\
            EXTRA_PORTS=\"$extra_ports\" \\
            bash \"${REMOTE_REPO_DIR}/clean.sh\" local
        '
    " < /dev/null
}

cleanup() {
    # Ignore SIGINT inside cleanup so a second Ctrl+C from the user does NOT
    # kill the ssh children that are running clean.sh on remote nodes. We must
    # let those finish, otherwise remote vllm/proxy procs survive.
    trap '' INT TERM
    trap - EXIT
    echo "Stopping everything…"

    # 1. kill the local proxy (started by main as a background child)
    if [[ -n "${PROXY_PID:-}" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
        kill -9 "$PROXY_PID" 2>/dev/null || true
    fi
    pkill -9 -f "${SCRIPT_DIR}/proxy.py" 2>/dev/null || true

    # 2. clean every remote host (vllm prefill / decode processes).
    #    Use synchronous ssh, fan out via &, then wait.
    local ssh_pids=()
    for host in "${UNIQ_HOSTS[@]}"; do
        echo "  Cleaning $host ..."
        cleanup_host "$host" &
        ssh_pids+=("$!")
    done
    for pid in "${ssh_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "Cleanup done."
    exit 0
}

wait_for_server() {
  local host=$1
  local port=$2
  local path=${3:-/v1/completions}
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on $host:$port$path..."

  while true; do
    if ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" \
        "curl -s 'localhost:${port}${path}'" > /dev/null 2>&1; then
      echo "Server on $host:$port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on $host:$port"
      return 1
    fi

    sleep 1
  done
}

main() {
    if [[ "$ACTION" == "stop" ]]; then
        echo "Stop requested. Cleaning up all hosts..."
        for host in "${UNIQ_HOSTS[@]}"; do
            echo "Cleaning $host ..."
            cleanup_host "$host" &
        done
        wait
        exit 0
    fi

    # Pre-flight checks on every participating host
    for host in "${UNIQ_HOSTS[@]}"; do
        ensure_python_library_installed_remote "$host" vllm &
    done
    wait
    for host in "${PREFILL_HOSTS[@]}"; do
        check_num_gpus_remote "$host" "$TP_SIZE" &
    done
    for host in "${DECODE_HOSTS[@]}"; do
        check_num_gpus_remote "$host" "$TP_SIZE" &
    done
    wait

    resolve_socket_ifname

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    # Cleanup stale processes on every host before launching
    echo "Cleaning stale vllm/proxy processes on all hosts ..."
    for host in "${UNIQ_HOSTS[@]}"; do
        cleanup_host "$host" &
    done
    wait

    # Ensure log dir exists on every remote host
    for host in "${UNIQ_HOSTS[@]}"; do
        ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "mkdir -p '${REMOTE_LOG_DIR}'" < /dev/null &
    done
    wait

    echo "Launching disaggregated serving components..."
    echo "Per-node log files (on each remote host):"
    echo "  - ${REMOTE_LOG_DIR}/prefill*.log: Prefill server logs (per remote host)"
    echo "  - ${REMOTE_LOG_DIR}/decode*.log:  Decode server logs (per remote host)"
    echo "  - ${LOG_ROOT}/${RUN_ID}/proxy.log: Proxy server log (local launcher host)"

    # =============================================================================
    # Launch Prefill Servers (X Producers, one per remote host)
    # =============================================================================
    echo ""
    echo "Starting ${#PREFILL_HOSTS[@]} prefill server(s)..."
    for i in "${!PREFILL_HOSTS[@]}"; do
        local host=${PREFILL_HOSTS[$i]}
        local gpu_ids=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local nixl_port=$((PREFILL_NIXL_PORT + i))

        echo "  Prefill server $((i+1)): Host $host, GPUs $gpu_ids, Port $port, Nixl Port $nixl_port"
        ssh -f "${SSH_OPTS[@]}" "$SSH_USER@$host" "
            cd ${REMOTE_REPO_DIR} && \
            PYTHONPATH=$REMOTE_REPO_DIR \
            VLLM_WORKER_MULTIPROC_METHOD=fork \
            UCX_NET_DEVICES=all \
            VLLM_PROFILER_PATH=$PROFILER \
            CUDA_VISIBLE_DEVICES=$gpu_ids \
            VLLM_NIXL_SIDE_CHANNEL_HOST=$host \
            VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port \
            vllm serve $MODEL \
            --host 0.0.0.0 \
            --port $port \
            --is-flowprefill \
            --num-runners 128 \
            --tensor-parallel-size $TP_SIZE \
            --gpu-memory-utilization $PREFILL_MEM_FRACTION_STATIC \
            --max-num-batched-tokens 8192 \
            --no-enable-prefix-caching \
            --enforce-eager \
            --trust-remote-code \
            --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
            > ${REMOTE_LOG_DIR}/prefill$((i+1)).log 2>&1 < /dev/null &
        " < /dev/null
    done

    # =============================================================================
    # Launch Decode Servers (Y Decoders, one per remote host)
    # =============================================================================
    echo ""
    echo "Starting ${#DECODE_HOSTS[@]} decode server(s)..."
    for i in "${!DECODE_HOSTS[@]}"; do
        local host=${DECODE_HOSTS[$i]}
        local gpu_ids=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local nixl_port=$((DECODE_NIXL_PORT + i))

        echo "  Decode server $((i+1)): Host $host, GPUs $gpu_ids, Port $port, Nixl Port $nixl_port"
        ssh -f "${SSH_OPTS[@]}" "$SSH_USER@$host" "
            cd ${REMOTE_REPO_DIR} && \
            PYTHONPATH=$REMOTE_REPO_DIR \
            UCX_NET_DEVICES=all \
            CUDA_VISIBLE_DEVICES=$gpu_ids \
            VLLM_NIXL_SIDE_CHANNEL_HOST=$host \
            VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port \
            vllm serve $MODEL \
            --host 0.0.0.0 \
            --port $port \
            --no-is-flowprefill \
            --tensor-parallel-size $TP_SIZE \
            --gpu-memory-utilization $DECODE_MEM_FRACTION_STATIC \
            --no-enable-prefix-caching \
            --no-enable-chunked-prefill \
            --enforce-eager \
            --trust-remote-code \
            --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
            > ${REMOTE_LOG_DIR}/decode$((i+1)).log 2>&1 < /dev/null &
        " < /dev/null
    done

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for i in "${!PREFILL_HOSTS[@]}"; do
        local host=${PREFILL_HOSTS[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        if ! wait_for_server "$host" "$port"; then
            echo "Failed to start prefill server on $host:$port"
            cleanup
            exit 1
        fi
    done
    for i in "${!DECODE_HOSTS[@]}"; do
        local host=${DECODE_HOSTS[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        if ! wait_for_server "$host" "$port"; then
            echo "Failed to start decode server on $host:$port"
            cleanup
            exit 1
        fi
    done

    # =============================================================================
    # Launch Proxy Server (locally on the launcher host)
    # =============================================================================
    echo ""
    echo "Starting proxy server locally on port $PROXY_PORT..."
    # Per-server (host, port) tuples — proxy receives one host and one port
    # per server, so two servers on the same host appear twice with different
    # ports.
    local prefiller_hosts_args="${PREFILL_HOSTS[*]}"
    local prefiller_ports_args="${PREFILL_PORT_ARRAY[*]}"
    local decoder_hosts_args="${DECODE_HOSTS[*]}"
    local decoder_ports_args="${DECODE_PORT_ARRAY[*]}"

    # Local log dir for the proxy (mirrors the per-node log layout).
    mkdir -p "${LOG_ROOT}/${RUN_ID}"

    # `env -u ...` strips any inherited http/https proxy so that proxy.py
    # forwards requests to the prefill/decode IPs directly instead of going
    # through a corporate proxy that can't reach internal addresses.
    nohup env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
              -u all_proxy -u ALL_PROXY -u no_proxy -u NO_PROXY \
              NO_PROXY='*' no_proxy='*' \
        python3 "${SCRIPT_DIR}/proxy.py" \
        --host 0.0.0.0 \
        --port $PROXY_PORT \
        --prefiller-hosts $prefiller_hosts_args \
        --prefiller-ports $prefiller_ports_args \
        --decoder-hosts  $decoder_hosts_args \
        --decoder-ports  $decoder_ports_args \
        > "${LOG_ROOT}/${RUN_ID}/proxy.log" 2>&1 < /dev/null &
    PROXY_PID=$!

    # Wait until proxy actually listens before declaring the system ready.
    # Use a plain local curl since the proxy lives on the launcher host.
    local proxy_start_ts=$(date +%s)
    while true; do
        if curl -s "localhost:${PROXY_PORT}/healthcheck" > /dev/null 2>&1; then
            echo "Proxy on localhost:${PROXY_PORT} is ready."
            break
        fi
        if ! kill -0 "$PROXY_PID" 2>/dev/null; then
            echo "Proxy died during startup; see ${LOG_ROOT}/${RUN_ID}/proxy.log"
            cleanup
            exit 1
        fi
        if (( $(date +%s) - proxy_start_ts >= TIMEOUT_SECONDS )); then
            echo "Timeout waiting for proxy on localhost:${PROXY_PORT}"
            cleanup
            exit 1
        fi
        sleep 1
    done

    # =============================================================================
    # Start serving
    # =============================================================================
    local launcher_host
    launcher_host=$(hostname -f 2>/dev/null || hostname)
    echo ""
    echo "All servers are up. Proxy URL: http://${launcher_host}:${PROXY_PORT}  (also http://0.0.0.0:${PROXY_PORT} on this host)"
    echo "Proxy log: ${LOG_ROOT}/${RUN_ID}/proxy.log"
    echo "Per-node vllm logs: ${REMOTE_LOG_DIR}/ (on each remote host)"

    # Stay in the foreground so Ctrl+C reaches us and triggers cleanup().
    # The proxy is a local child, so wait on it directly.
    wait "$PROXY_PID"
}

# EXIT trap is a safety net: even if main() returns or dies unexpectedly, we
# still try to clean up every remote node.
trap '[[ "$ACTION" == "start" ]] && cleanup' EXIT

main
