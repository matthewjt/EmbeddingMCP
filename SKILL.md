---
name: embedding-mcp
description: Use the self-hosted EmbeddingMCP server (EmbeddingGemma-2 via llama.cpp) to compute 768-d multimodal embeddings for semantic search, similarity ranking, deduplication, and clustering over text, images, and audio.
---

# EmbeddingMCP — Agent Usage Guide

## What this MCP is

A remote MCP server (default `http://<mcp-server>:8090/mcp`) that
computes **768-dimensional, L2-normalized embeddings** with
EmbeddingGemma-2. Text, images, and audio embed into **one shared
vector space**, so a text query can directly retrieve matching images
or audio by cosine similarity.

**Key mental model:** embeddings measure *meaning and similarity*,
they do not read content. This MCP cannot summarize, transcribe, OCR,
or answer questions about a document. Pair it with your own reading
tools: use embeddings to **find** the relevant pieces, then read the
winning pieces yourself.

## Tools

| Tool | Args | Returns |
|---|---|---|
| `embed_text` | `texts`: array of 1-32 strings (≤8000 chars each) | `vectors` (one 768-d list per input, in order), `dimensions`, `elapsed_ms` |
| `embed_image` | `image_base64`: PNG/JPEG/WebP, ≤10 MB | `vector`, `dimensions` |
| `embed_audio` | `audio_base64`: WAV, ≤10 MB (experimental) | `vector`, `dimensions` |
| `compare_texts` | `text_a`, `text_b` | cosine similarity (fast server-side shortcut) |
| `embedding_health` | — | backend reachability, model, verified modalities |

Vectors are L2-normalized → cosine similarity = dot product.

## Core workflow: semantic search over a corpus

1. **Chunk** the material into ~200-500 word passages (paragraph
   boundaries, ~15% overlap). Keep `(id, text)` pairs.
2. **Embed in batches of ≤8 texts per `embed_text` call.** Larger
   single requests can silently return fewer vectors than sent —
   always verify `len(vectors) == len(batch)` and re-chunk on
   mismatch.
3. **Store** vectors locally (SQLite + numpy is fine; 1500 chunks ≈
   5 MB).
4. **Query**: embed the question, score every stored vector by dot
   product, return top-k with passage text + citation.
5. **Answer from the hits**: read the retrieved passages yourself and
   synthesize — never answer from vector scores alone.

## Other patterns

- **Deduplication**: embed items, cluster/flag pairs with cosine
  > ~0.9 (tune per domain).
- **Classification**: embed labeled examples once; embed each new
  item; assign the label of the nearest example (kNN).
- **Cross-modal image search**: embed a text query, compare against
  stored image vectors.
- **Audio matching**: embed reference clips, match new clips;
  experimental — treat margins below ~0.1 as noise.

- **Whole-document Q&A**: see README "Document pipeline" —
  `src/ingest_book.py` (chunk+embed a book/article into SQLite) then
  `src/query_book.py` (rank passages for a question). Verified on a
  62k-word ebook: 173 chunks, chapter-labelled hits.

## Rules & gotchas

- Call `embedding_health` first if any tool errors; distinguish
  `backend_unavailable` (llama-server down) from tool validation
  errors.
- Respect limits: 32 texts/call, 8000 chars/text, 10 MB media.
- Send base64 **without** data-URI prefix to `embed_image` /
  `embed_audio` (the server adds the wrapper).
- Similarity guidance (empirical starting points, not gospel):
  ≥0.75 strongly related, 0.45-0.75 same topic, <0.35 unrelated.
- Embeddings are lossy and non-invertible: you cannot recover text
  from a vector.
- The server has no auth and is LAN-trust only — never embed
  sensitive content you would not send to your own server.
- Long jobs: embedding is fast (~10 ms/text on GPU); if a call takes
  seconds, suspect queueing — check health before retrying.

## Verify it works (from any MCP client)

```python
from mcp import Client
async with Client("http://<mcp-server>:8090/mcp") as c:
    r = await c.call_tool("embed_text", {"texts": ["hello", "hello"]})
    # expect dimensions: 768 and cosine(a, b) == 1.0 for identical strings
```
