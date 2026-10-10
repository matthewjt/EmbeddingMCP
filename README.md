# EmbeddingMCP

Self-hosted **embedding MCP server**: gives any MCP client (OpenCode,
Claude, etc.) text, image, audio, and video embeddings computed on your
own GPU — 768-dimensional vectors from Google **EmbeddingGemma-2**,
served by **llama.cpp** behind a Python **MCP** (streamable-http)
endpoint. Nothing leaves your LAN.

Companion guide for LLM agents: [`SKILL.md`](SKILL.md).

## What it does

| MCP tool | Input | Output |
|---|---|---|
| `embed_text` | 1-32 strings (≤8000 chars each) | 768-d L2-normalized vectors |
| `embed_image` | base64 PNG/JPEG/WebP (≤10 MB) | 768-d vector |
| `embed_audio` | base64 WAV (≤10 MB, experimental) | 768-d vector |
| `compare_texts` | two strings | cosine similarity in one call |
| `embedding_health` | — | backend/model/modality status |

All modalities land in **one vector space**: a text vector can be
compared directly with an image vector (cross-modal search).

## Architecture

```
MCP client (OpenCode, ...)
   |  streamable-http  http://<server>:8090/mcp
   v
src/server.py  (Python, mcp SDK v2 MCPServer)
   |  OpenAI-compatible /v1/embeddings (localhost)
   v
llama-server (llama.cpp CUDA build, :8080, --embedding)
   model: unsloth/embeddinggemma-2-GGUF:UD-Q4_K_XL (+ mmproj-Q8_0)
```

- The MCP layer adds validation, limits, batching, error categories,
  and secret-free responses. It is a thin, stateless proxy.
- `llama-server` does all inference. Swapping quant or model is a
  `.env` change; the MCP API never changes.

## How setup works (`setup.sh`, 7 steps, idempotent, no sudo)

1. **Preflight** — arch must be x86_64; ≥10 GB free disk; NVIDIA probe
   (falls back to CPU-only mode); writes `state/hw.json`.
2. **uv** — installed user-level if missing.
3. **llama.cpp CUDA build** — downloads pinned release tarballs
   (`llama-<TAG>-bin-ubuntu-cuda-12.8-x64.tar.gz` + matching
   `cudart-...`) from GitHub releases, verifies **sha256** against
   `ops/checksums.txt`, extracts to `~/.local/llama-cuda/`.
4. **Python env** — `uv venv` (3.14, falls back 3.12) + `mcp>=2,<3`,
   `httpx2`.
5. **Hardware-aware knobs** — `--ctx-size` and `--parallel` computed
   from core count/RAM (overridable in `.env`); written to
   `state/env.sh`.
6. **Start services** — `run.sh start` launches llama-server (pulls
   the ~1.2 GB model from Hugging Face on first run) then the MCP
   server, tracking PIDs in `state/pids/`.
7. **Verify** — runs the stdio test suite (`src/test_client.py`);
   prints `SETUP_COMPLETE` plus the two operator-approval steps.

## Install (new machine)

```bash
git clone https://github.com/matthewjt/EmbeddingMCP.git embedding-mcp
cd embedding-mcp
cp .env.example .env      # ports, quant, LAN subnet — edit as needed
bash setup.sh
bash doctor.sh            # 9 read-only checks, exit 0 = healthy
bash run.sh status
```

Operator-approved follow-ups (the scripts only **print** these; they
never run sudo):

- `bash ops/firewall.sh` — ufw rules for LAN-only access.
- `ops/register-opencode.md` — exact JSON for OpenCode (backup first).

## Daily operation

```bash
bash run.sh start | stop | status | restart
tail -f state/llama.log state/mcp.log
bash doctor.sh                       # health
.venv/bin/python src/bench.py --name myhost   # latency/VRAM benchmark
.venv/bin/python src/retrieval_test.py Q4     # retrieval quality (needs server on :8081)
```

## Verified performance (GTX 1050 Ti 4 GB, llama.cpp b11551 CUDA)

| Metric | CUDA (production) | CPU fallback |
|---|---|---|
| Text p50 | 9.7 ms | 32.8 ms |
| Image p50 | 200 ms | 447 ms |
| Throughput (batch 32) | 208 items/s | — |
| Peak VRAM (all quants) | ~2.3 GB | — |

Retrieval quality suite (24 docs / 8 queries): R@1, R@3, MRR, nDCG@10
= 1.0 for both UD-Q4_K_XL and Q8_0.

## Limits & known behavior (important for agents)

- **Chunk large batches to ≤8 texts per request** — bigger single
  requests can silently return fewer vectors than sent.
- Audio encoder is experimental (usable, weaker cross-modal margins).
- Video embeddings work at the backend (`input_video`) but have no
  MCP tool yet.
- Embeddings describe *meaning/similarity* — they do not read,
  transcribe, or OCR content.
- No authentication: LAN-only by design, never expose publicly.

## Rollback

```bash
bash run.sh stop
rm -rf ~/.local/llama-cuda <repo-dir>
rm -rf ~/.cache/huggingface/hub/models--unsloth--embeddinggemma-2-GGUF
# OpenCode: restore timestamped config backup; ufw: sudo ufw delete <n>
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `501` from `/v1/embeddings` | llama-server started without `--embedding` |
| `unknown model architecture: gemma-embedding2` (Ollama) | Ollama can't run this model on Linux (ollama#18825); use llama.cpp |
| MCP up, tools error `backend_unavailable` | llama-server down: `bash run.sh start`, check `state/llama.log` |
| CUDA binary won't start | missing `LD_LIBRARY_PATH` (run.sh sets it); driver/CUDA-12.8 mismatch |
| Fewer vectors than texts sent | request >8 texts in one call — chunk to 8 |
| Model pull fails | no HF reachability; pre-download on a networked box into `~/.cache/huggingface` |

## Repo layout

```
setup.sh doctor.sh run.sh .env.example README.md SKILL.md
src/    server.py test_client.py bench.py retrieval_test.py
ops/    checksums.txt firewall.sh register-opencode.md
state/  (gitignored: logs, pids, hw.json)
```

## Provenance

Built and verified 2026-10-10 on Arch Linux, GTX 1050 Ti
(driver 580), llama.cpp b11551 CUDA, `unsloth/embeddinggemma-2-GGUF:
UD-Q4_K_XL`, mcp SDK 2.3.0, Python 3.14. Model license: Apache-2.0.
