#!/usr/bin/env bash
# Read-only preflight/health checker. Never installs, never kills.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0
ok()   { echo "  PASS  $*"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
warn() { echo "  WARN  $*"; }

echo "== preflight"
[ "$(uname -m)" = x86_64 ] && ok "arch x86_64" || bad "arch $(uname -m) unsupported"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >/dev/null 2>&1 \
  && ok "nvidia-smi works: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
  || warn "no nvidia-smi (CPU-only fallback path)"
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge 10 ] && ok "disk free ${AVAIL_GB} GB" || bad "disk free ${AVAIL_GB} GB (<10)"
CORES=$(nproc); RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
ok "cpu cores=$CORES ram=${RAM_GB}GB"

echo "== installs"
[ -x "$HOME/.local/llama-cuda/llama-${LLAMA_TAG:-b11551}/llama-server" ] \
  && ok "llama-server CUDA binary present" || warn "CUDA binary missing (run setup.sh)"
[ -d .venv ] && ok ".venv present" || bad ".venv missing (run setup.sh)"
[ -d "$HOME/.cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF" ] \
  && ok "model cache present" || warn "model cache missing (setup pulls it)"

echo "== runtime"
LLAMA_PORT=${LLAMA_PORT:-8080}; MCP_PORT=${MCP_PORT:-8090}
curl -s --max-time 3 "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null && ok "llama-server :$LLAMA_PORT" || warn "llama-server :$LLAMA_PORT not responding"
MCP_PATH=${MCP_PATH:-/mcp}
CODE=$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$MCP_PORT$MCP_PATH")
[ "$CODE" != 000 ] && ok "mcp server :$MCP_PORT (http $CODE)" || warn "mcp server :$MCP_PORT not responding"

echo "== summary: $PASS pass / $FAIL fail"
exit "$FAIL"
