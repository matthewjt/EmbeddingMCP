#!/usr/bin/env bash
# Start/stop/status for the embedding MCP stack. User-level, PID-file based.
set -u
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
LLAMA_TAG=${LLAMA_TAG:-b11551}
LLAMA_DIR="$HOME/.local/llama-cuda/llama-$LLAMA_TAG"
CUDART_DIR="$HOME/.local/llama-cuda/cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64"
LLAMA_PORT=${LLAMA_PORT:-8080}
MCP_PORT=${MCP_PORT:-8090}
MCP_HOST=${MCP_HOST:-0.0.0.0}
MCP_PATH=${MCP_PATH:-/mcp}
EMBED_MODEL_REPO=${EMBED_MODEL_REPO:-unsloth/embeddinggemma-2-GGUF}
EMBED_QUANT=${EMBED_QUANT:-UD-Q4_K_XL}
LLAMA_CTX=${LLAMA_CTX:-8192}
LLAMA_PARALLEL=${LLAMA_PARALLEL:-4}
LLAMA_UBATCH=${LLAMA_UBATCH:-2048}
mkdir -p state/pids

is_up() { [ -f "state/pids/$1" ] && kill -0 "$(cat "state/pids/$1")" 2>/dev/null; }

start_llama() {
  is_up llama && { echo "llama-server already up (pid $(cat state/pids/llama))"; return; }
  LD_LIBRARY_PATH="$LLAMA_DIR:$CUDART_DIR" setsid bash -c \
    "echo \$\$ > '$PWD/state/pids/llama'; exec '$LLAMA_DIR/llama-server' -hf '$EMBED_MODEL_REPO:$EMBED_QUANT' --embedding \
     --host 0.0.0.0 --port '$LLAMA_PORT' --ctx-size '$LLAMA_CTX' --parallel '$LLAMA_PARALLEL' -ub '$LLAMA_UBATCH' \
     >> '$PWD/state/llama.log' 2>&1" </dev/null >/dev/null 2>&1 &
  for _ in $(seq 1 60); do curl -s --max-time 2 "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null && { echo "llama-server up (:$LLAMA_PORT)"; return; }; sleep 2; done
  echo "llama-server FAILED to start; see state/llama.log"; return 1
}

start_mcp() {
  is_up mcp && { echo "mcp server already up (pid $(cat state/pids/mcp))"; return; }
  EMBED_TRANSPORT=streamable-http EMBED_HOST="$MCP_HOST" EMBED_PORT="$MCP_PORT" EMBED_PATH="$MCP_PATH" \
    EMBED_BACKEND_URL="http://127.0.0.1:$LLAMA_PORT" \
    setsid bash -c "echo \$\$ > '$PWD/state/pids/mcp'; exec .venv/bin/python src/server.py >> '$PWD/state/mcp.log' 2>&1" </dev/null >/dev/null 2>&1 &
  sleep 4
  curl -s --max-time 3 -o /dev/null "http://127.0.0.1:$MCP_PORT$MCP_PATH" && { echo "mcp server up (:$MCP_PORT$MCP_PATH)"; return; }
  echo "mcp server FAILED to start; see state/mcp.log"; return 1
}

stop() {
  for s in llama mcp; do
    if is_up "$s"; then kill "$(cat "state/pids/$s")" && echo "stopped $s"; rm -f "state/pids/$s";
    else echo "$s not running"; fi
  done
}

case "${1:-status}" in
  start)  start_llama && start_mcp ;;
  stop)   stop ;;
  status)
    for s in llama mcp; do is_up "$s" && echo "$s UP (pid $(cat state/pids/$s))" || echo "$s down"; done ;;
  restart) stop; sleep 2; start_llama && start_mcp ;;
  *) echo "usage: $0 {start|stop|status|restart}"; exit 2 ;;
esac
