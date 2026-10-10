# embedding-mcp

Portable embedding MCP: EmbeddingGemma-2 (768-d, text/image/audio/video)
served by llama.cpp `llama-server` behind a Python MCP server
(streamable-http). Verified on a GTX 1050 Ti host; hardware-adaptive
setup for other machines.

## Quickstart (x86_64 Linux, NVIDIA)

```bash
git clone <your-repo-url> embedding-mcp && cd embedding-mcp
cp .env.example .env       # edit ports/quant/subnet as needed
bash setup.sh              # idempotent, user-level, no sudo
bash doctor.sh             # health checks
bash run.sh status
```

Then (operator-approved steps only): `bash ops/firewall.sh` (prints
ufw commands) and `ops/register-opencode.md` (client registration).

## Architecture

```
client -> MCP streamable-http :8090/mcp -> llama-server :8080 (--embedding)
         model: unsloth/embeddinggemma-2-GGUF:UD-Q4_K_XL + mmproj-Q8_0
```

## Tools

| Tool | Purpose |
|---|---|
| `embed_text` | 1-32 texts -> 768-d normalized vectors |
| `embed_image` | base64 PNG/JPEG/WebP -> vector |
| `embed_audio` | base64 WAV -> vector (experimental encoder) |
| `compare_texts` | cosine similarity of two strings |
| `embedding_health` | backend/model/modality status |

## Layout

- `setup.sh` installer (preflight, uv, CUDA tarballs w/ sha256, venv,
  model pull, hw-aware knobs, start, verify)
- `run.sh start|stop|status|restart` (PID-file based)
- `doctor.sh` read-only checks
- `src/` server + test/bench/quality scripts
- `ops/` firewall + client registration (manual-approval only)
- `state/` logs, PIDs, hardware profile (gitignored)

## Verified benchmarks (GTX 1050 Ti, b11551 CUDA)

Text p50 ~10 ms, image ~200 ms, 208 items/s @ batch 32, peak VRAM
~2.3 GB. Retrieval quality suite: R@1/MRR/nDCG = 1.0.

## Limits

Audio encoder experimental; video works at backend (`input_video`) but
no MCP tool yet; no auth on LAN — keep off public internet; chunk
batching <=8 texts/request for large ingest jobs.

## Rollback

`bash run.sh stop`; remove `~/.local/llama-cuda`, this repo dir, and
model cache `~/.cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF`.
Restore timestamped OpenCode config backup; `sudo ufw delete <n>`.
