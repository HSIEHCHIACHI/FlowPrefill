#!/bin/bash

# =============================================================================
# Standalone cleanup for vLLM Disaggregated Serving
# =============================================================================
# Kills leftover vllm/proxy/worker procs and any matching process holding
# /dev/nvidia*, frees configured ports, and clears NIXL/UCX shm leftovers.
#
# Usage:
#   bash clean.sh              # clean THIS machine (same as `bash clean.sh local`)
#   bash clean.sh local        # explicit local cleanup
#   bash clean.sh host1 host2  # ssh into the given hosts and clean each
#
# Configuration via env:
#   SSH_USER       SSH user           (default: root)
#   PREFILL_PORT   http port to free  (default: 20000)
#   DECODE_PORT    http port to free  (default: 21000)
#   PROXY_PORT     proxy port to free (default: 8192)
#   EXTRA_PORTS    extra ports to free, space-separated (optional)
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SSH_USER=${SSH_USER:-root}
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5)

PREFILL_PORT=${PREFILL_PORT:-20000}
DECODE_PORT=${DECODE_PORT:-21000}
PROXY_PORT=${PROXY_PORT:-8192}
EXTRA_PORTS=${EXTRA_PORTS:-""}   # optional, space-separated

# The cleanup snippet that runs on each target host (local or remote).
build_clean_snippet() {
    local ports="$PREFILL_PORT $DECODE_PORT $PROXY_PORT $EXTRA_PORTS"
    cat <<EOF
set +e
echo "[\$(hostname)] === cleanup start ==="

# Helper: kill PIDs but only when the process looks like one of OUR vllm /
# proxy procs. Matches by cmdline AND by environ — needed because vllm spawns
# worker procs whose cmdline is just \`python -c "from multiprocessing.spawn ..."\`
# (no vllm/FlowPrefill keyword), but whose environ inherits our VLLM_* /
# PYTHONPATH=FlowPrefill markers.
self_pid=\$\$
self_ppid=\$(awk '{print \$4}' /proc/\$self_pid/stat 2>/dev/null)
safe_kill() {
    local victims=""
    for pid in "\$@"; do
        case "\$pid" in ''|*[!0-9]*) continue;; esac
        [ "\$pid" = "\$self_pid"  ] && continue
        [ "\$pid" = "\$self_ppid" ] && continue
        # Skip if the process already exited between listing and now.
        [ -r /proc/\$pid/cmdline ] || continue
        cmdline=\$(cat /proc/\$pid/cmdline 2>/dev/null | tr '\\0' ' ')
        environ=\$(cat /proc/\$pid/environ 2>/dev/null | tr '\\0' '\\n')
        if echo "\$cmdline" | grep -qE 'vllm serve|vllm\\.entrypoints|EngineCore|VllmWorker|Worker_TP|NixlConnector|proxy\\.py'; then
            victims="\$victims \$pid"
        elif echo "\$environ" | grep -qE '^(VLLM_NIXL_SIDE_CHANNEL_|VLLM_PROFILER_PATH=|PYTHONPATH=.*FlowPrefill)'; then
            victims="\$victims \$pid"
        fi
    done
    if [ -n "\$victims" ]; then
        echo "[\$(hostname)] killing:\$victims"
        kill -9 \$victims >/dev/null 2>&1
    fi
}

# 1. by cmdline pattern
for pat in 'vllm serve' 'vllm.entrypoints' 'multiproc_executor' \\
           'EngineCore' 'VllmWorker' 'Worker_TP' 'NixlConnector' \\
           'proxy.py'; do
    pids=\$(pgrep -f "\$pat" 2>/dev/null)
    [ -n "\$pids" ] && safe_kill \$pids
done

# 2. by tcp port (only our http/proxy ports — go through safe_kill anyway)
if command -v fuser >/dev/null 2>&1; then
    for p in $ports; do
        pids=\$(fuser -n tcp \$p 2>/dev/null | tr -s ' ' '\\n' | awk '/^[0-9]+\$/' | sort -u | xargs)
        [ -n "\$pids" ] && safe_kill \$pids
    done
elif command -v lsof >/dev/null 2>&1; then
    for p in $ports; do
        pids=\$(lsof -t -iTCP:\$p -sTCP:LISTEN 2>/dev/null | sort -u | xargs)
        [ -n "\$pids" ] && safe_kill \$pids
    done
else
    echo "[\$(hostname)] warning: neither fuser nor lsof found; cannot clean by port"
fi

# 3. by /dev/nvidia* — only kill GPU-holders that ALSO match our cmdline keywords.
gpu_holders=""
if command -v fuser >/dev/null 2>&1; then
    gpu_holders=\$(fuser /dev/nvidia* 2>/dev/null | tr -s ' ' '\\n' | awk '/^[0-9]+\$/' | sort -u | xargs)
elif command -v lsof >/dev/null 2>&1; then
    gpu_holders=\$(lsof -t /dev/nvidia* 2>/dev/null | sort -u | xargs)
fi
if [ -n "\$gpu_holders" ]; then
    safe_kill \$gpu_holders
else
    echo "[\$(hostname)] no processes holding /dev/nvidia*"
fi

# 4. clear leaked shm objects belonging to vllm/NIXL/UCX only.
#    Leave /dev/shm/sem.* alone — those are POSIX semaphores used by other
#    workloads (jupyter, multiprocessing, joblib, ...).
shm_removed=\$(ls /dev/shm 2>/dev/null | grep -Ei 'nixl|ucx|vllm' | wc -l)
rm -f /dev/shm/*nixl* /dev/shm/*ucx* /dev/shm/*vllm* 2>/dev/null
echo "[\$(hostname)] removed \$shm_removed vllm/nixl/ucx shm object(s)"

# 5. final report
remaining=""
if command -v fuser >/dev/null 2>&1; then
    remaining=\$(fuser /dev/nvidia* 2>/dev/null | tr -s ' ' '\\n' | awk '/^[0-9]+\$/' | sort -u | xargs)
fi
if [ -n "\$remaining" ]; then
    echo "[\$(hostname)] WARNING: still holding GPU: \$remaining"
    if command -v ps >/dev/null 2>&1; then
        ps -p \$remaining -o pid,user,stat,cmd 2>/dev/null
    fi
else
    echo "[\$(hostname)] GPU clean."
fi
echo "[\$(hostname)] === cleanup done ==="
EOF
}

clean_local() {
    local snippet
    snippet=$(build_clean_snippet)
    bash -c "$snippet"
}

clean_remote() {
    local host=$1
    local snippet
    snippet=$(build_clean_snippet)
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$host" "bash -lc $(printf '%q' "$snippet")" < /dev/null
}

# Parse target hosts
TARGET_HOSTS=()
if [[ $# -eq 0 || "$1" == "local" ]]; then
    TARGET_HOSTS=("__local__")
else
    TARGET_HOSTS=("$@")
fi

echo "Cleaning hosts: ${TARGET_HOSTS[*]}"
echo "Ports to free: $PREFILL_PORT $DECODE_PORT $PROXY_PORT $EXTRA_PORTS"
echo ""

PIDS=()
for host in "${TARGET_HOSTS[@]}"; do
    if [[ "$host" == "__local__" ]]; then
        ( clean_local ) &
    else
        ( clean_remote "$host" ) &
    fi
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=1
done

echo ""
if (( FAILED )); then
    echo "Some cleanup tasks reported errors (see output above)."
    exit 1
fi
echo "All cleanup tasks finished."
