#!/usr/bin/env bash
# One-shot installer for the embedding MCP stack. User-level, idempotent, no sudo.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$PWD"

[ -f .env ] && set -a && . ./.env && set +a
LLAMA_TAG=${LLAMA_TAG:-b11551}
EMBED_MODEL_REPO=${EMBED_MODEL_REPO:-unsloth/embeddinggemma-2-GGUF}
EMBED_QUANT=${EMBED_QUANT:-UD-Q4_K_XL}

echo "== [1/7] preflight"
[ "$(uname -m)" = x86_64 ] || { echo "unsupported arch $(uname -m)"; exit 1; }
CORES=$(nproc); RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge 10 ] || { echo "need >=10 GB free disk, have $AVAIL_GB"; exit 1; }
USE_CUDA=1
if ! (command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1); then
  echo "no NVIDIA GPU detected -> CPU-only mode (CUDA download skipped)"; USE_CUDA=0
fi
echo "cores=$CORES ram=${RAM_GB}GB disk_free=${AVAIL_GB}GB cuda=$USE_CUDA"
mkdir -p state
echo "{\"cores\":$CORES,\"ram_gb\":$RAM_GB,\"disk_free_gb\":$AVAIL_GB,\"cuda\":$USE_CUDA,\"arch\":\"$(uname -m)\"}" > state/hw.json

echo "== [2/7] uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null && echo "uv ready: $(uv --version)"

echo "== [3/7] llama.cpp CUDA build ($LLAMA_TAG)"
LLAMA_DIR="$HOME/.local/llama-cuda/llama-$LLAMA_TAG"
CUDART_DIR="$HOME/.local/llama-cuda/cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64"
if [ "$USE_CUDA" = 1 ] && [ ! -x "$LLAMA_DIR/llama-server" ]; then
  mkdir -p "$HOME/.local/llama-cuda" && cd "$HOME/.local/llama-cuda"
  for A in "llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz" "cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz"; do
    echo "downloading $A"
    curl -fsSL -o "$A" "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/$A"
  done
  cp "$REPO/ops/checksums.txt" /tmp/emb_checksums.txt
  sha256sum -c /tmp/emb_checksums.txt --ignore-missing
  tar --zstd -xf "llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz"
  tar --zstd -xf "cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz"
  rm -f "llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz" "cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz"
  cd "$REPO"
  echo "extracted to $LLAMA_DIR"
elif [ "$USE_CUDA" = 1 ]; then
  echo "CUDA build already present"
fi
if [ "$USE_CUDA" = 0 ] && [ ! -x "$HOME/.local/bin/llama" ]; then
  curl -LsSf https://llama.app/install.sh | sh
fi

echo "== [4/7] python env"
[ -d .venv ] || uv venv --python 3.14 .venv 2>/dev/null || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "mcp>=2,<3" httpx2 >/dev/null
.venv/bin/python -c "from mcp.server.mcpserver import MCPServer; import httpx2; print('sdk ok')"

echo "== [5/7] hardware-aware knobs"
LLAMA_CTX=${LLAMA_CTX:-auto}; LLAMA_PARALLEL=${LLAMA_PARALLEL:-auto}
[ "$LLAMA_CTX" = auto ] && LLAMA_CTX=$([ "$RAM_GB" -ge 32 ] && echo 16384 || echo 8192)
[ "$LLAMA_PARALLEL" = auto ] && LLAMA_PARALLEL=$(( CORES / 2 )); [ "$LLAMA_PARALLEL" -lt 2 ] && LLAMA_PARALLEL=2
[ "$LLAMA_PARALLEL" -gt 16 ] && LLAMA_PARALLEL=16
cat > state/env.sh <<EOF
export LLAMA_TAG=$LLAMA_TAG
export LLAMA_CTX=$LLAMA_CTX
export LLAMA_PARALLEL=$LLAMA_PARALLEL
export LLAMA_UBATCH=${LLAMA_UBATCH:-2048}
export LLAMA_PORT=${LLAMA_PORT:-8080}
export MCP_PORT=${MCP_PORT:-8090}
export EMBED_MODEL_REPO=$EMBED_MODEL_REPO
export EMBED_QUANT=$EMBED_QUANT
EOF
echo "ctx=$LLAMA_CTX parallel=$LLAMA_PARALLEL (state/env.sh written)"

echo "== [6/7] start services (pulls model on first run)"
sed -i "s|^LLAMA_CTX=.*|LLAMA_CTX=$LLAMA_CTX|; s|^LLAMA_PARALLEL=.*|LLAMA_PARALLEL=$LLAMA_PARALLEL|" .env 2>/dev/null || true
./run.sh start

echo "== [7/7] verify"
EMBED_SERVER_PY=src/server.py .venv/bin/python src/test_client.py | tail -6
echo
echo "SETUP_COMPLETE"
echo "Next (operator approval required, never automated):"
echo "  firewall:  bash ops/firewall.sh   (prints sudo ufw commands)"
echo "  client:    see ops/register-opencode.md"
