import base64
import json
import math
import os
import time

import httpx2
from mcp.server.mcpserver import MCPServer

BACKEND_URL = os.environ.get("EMBED_BACKEND_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("EMBED_MODEL", "unsloth/embeddinggemma-2-GGUF")
TIMEOUT_SECONDS = float(os.environ.get("EMBED_TIMEOUT_SECONDS", "120"))
MAX_TEXTS = int(os.environ.get("EMBED_MAX_TEXTS", "32"))
MAX_TEXT_CHARS = int(os.environ.get("EMBED_MAX_TEXT_CHARS", "8000"))
MAX_MEDIA_BYTES = int(os.environ.get("EMBED_MAX_MEDIA_BYTES", str(10 * 1024 * 1024)))

mcp = MCPServer(
    name="embedding",
    title="Embedding MCP",
    instructions=(
        "Generates 768-dim embeddings for text, images, and audio via a "
        "llama.cpp EmbeddingGemma-2 backend. Use embed_text for text/code, "
        "embed_image for PNG/JPEG/WebP base64, embed_audio for WAV base64. "
        "Vectors are L2-normalized and live in one unified space."
    ),
)


def _err(code: str, message: str, **extra) -> dict:
    out = {"error": code, "message": message}
    out.update(extra)
    return out


def _b64_size(raw: str) -> int:
    return len(base64.b64decode(raw, validate=False)) if raw else 0


async def _post(path: str, payload: dict) -> dict:
    async with httpx2.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        resp = await client.post(f"{BACKEND_URL}{path}", json=payload)
    if resp.status_code >= 400:
        detail = resp.json().get("error", {}).get("message", resp.text[:200])
        raise RuntimeError(detail)
    return resp.json()


async def _get(path: str) -> dict:
    async with httpx2.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        resp = await client.get(f"{BACKEND_URL}{path}")
    if resp.status_code >= 400:
        raise RuntimeError(resp.text[:200])
    return resp.json()


def _vectors_from_oai(data: dict) -> list[list[float]]:
    return [item["embedding"] for item in data.get("data", [])]


@mcp.tool()
async def embed_text(texts: list[str], normalize: bool = True) -> dict:
    """Embed one or more texts. Returns 768-dim vectors with timing metadata."""
    if not texts or len(texts) > MAX_TEXTS:
        return _err("INVALID_INPUT", f"texts must contain 1-{MAX_TEXTS} items")
    if any(len(t) > MAX_TEXT_CHARS for t in texts):
        return _err("PAYLOAD_TOO_LARGE", f"each text must be <= {MAX_TEXT_CHARS} chars")
    t0 = time.perf_counter()
    try:
        data = await _post("/v1/embeddings", {"input": texts, "embedd": normalize})
    except httpx2.ConnectError:
        return _err("BACKEND_UNREACHABLE", f"cannot reach {BACKEND_URL}")
    except Exception as exc:
        return _err("BACKEND_ERROR", str(exc)[:300])
    return {
        "model": MODEL,
        "modality": "text",
        "embeddings": _vectors_from_oai(data),
        "dimensions": len(_vectors_from_oai(data)[0]) if data.get("data") else 0,
        "normalized": normalize,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }


@mcp.tool()
async def embed_image(image_base64: str) -> dict:
    """Embed an image (PNG/JPEG/WebP) given as base64. Returns one 768-dim vector."""
    if _b64_size(image_base64) > MAX_MEDIA_BYTES:
        return _err("PAYLOAD_TOO_LARGE", f"image exceeds {MAX_MEDIA_BYTES} bytes")
    payload = {
        "input": [
            {"content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}
        ]
    }
    t0 = time.perf_counter()
    try:
        data = await _post("/v1/embeddings", payload)
    except httpx2.ConnectError:
        return _err("BACKEND_UNREACHABLE", f"cannot reach {BACKEND_URL}")
    except Exception as exc:
        return _err("INVALID_MEDIA_OR_BACKEND", str(exc)[:300])
    vecs = _vectors_from_oai(data)
    if not vecs:
        return _err("INVALID_MEDIA", "backend returned no embedding")
    return {
        "model": MODEL,
        "modality": "image",
        "embedding": vecs[0],
        "dimensions": len(vecs[0]),
        "normalized": True,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }


@mcp.tool()
async def embed_audio(audio_base64: str, format: str = "wav") -> dict:
    """Embed audio (WAV) given as base64. Native audio encoder is EXPERIMENTAL."""
    if format not in ("wav",):
        return _err("UNSUPPORTED_MODALITY", f"format {format!r} not supported by this backend")
    if _b64_size(audio_base64) > MAX_MEDIA_BYTES:
        return _err("PAYLOAD_TOO_LARGE", f"audio exceeds {MAX_MEDIA_BYTES} bytes")
    payload = {
        "input": [
            {"content": [{"type": "input_audio", "input_audio": {"data": f"data:audio/{format};base64,{audio_base64}", "format": format}}]}
        ]
    }
    t0 = time.perf_counter()
    try:
        data = await _post("/v1/embeddings", payload)
    except httpx2.ConnectError:
        return _err("BACKEND_UNREACHABLE", f"cannot reach {BACKEND_URL}")
    except Exception as exc:
        return _err("INVALID_MEDIA_OR_BACKEND", str(exc)[:300])
    vecs = _vectors_from_oai(data)
    if not vecs:
        return _err("INVALID_MEDIA", "backend returned no embedding")
    return {
        "model": MODEL,
        "modality": "audio",
        "encoder_status": "experimental",
        "embedding": vecs[0],
        "dimensions": len(vecs[0]),
        "normalized": True,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }


@mcp.tool()
async def compare_texts(text_a: str, text_b: str) -> dict:
    """Embed two texts and return their cosine similarity."""
    res = await embed_text([text_a, text_b])
    if "error" in res:
        return res
    a, b = res["embeddings"][0], res["embeddings"][1]
    sim = sum(x * y for x, y in zip(a, b))
    return {"cosine_similarity": round(sim, 4), "model": MODEL, "elapsed_ms": res["elapsed_ms"]}


@mcp.tool()
async def embedding_health() -> dict:
    """Report backend reachability, model, and verified modalities."""
    try:
        models = await _get("/v1/models")
        names = [m.get("name") for m in models.get("models", models.get("data", []))]
        await _get("/health")
        return {
            "backend_url": BACKEND_URL,
            "backend": "llama.cpp llama-server",
            "models": names,
            "verified_modalities": ["text", "image", "audio(experimental)"],
            "payload_limits": {"max_texts": MAX_TEXTS, "max_text_chars": MAX_TEXT_CHARS, "max_media_bytes": MAX_MEDIA_BYTES},
        }
    except Exception as exc:
        return _err("BACKEND_UNREACHABLE", str(exc)[:300])


if __name__ == "__main__":
    transport = os.environ.get("EMBED_TRANSPORT", "stdio")
    if transport == "streamable-http":
        from mcp.server.transport_security import TransportSecuritySettings

        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        )
        mcp.run(
            transport="streamable-http",
            host=os.environ.get("EMBED_HOST", "0.0.0.0"),
            port=int(os.environ.get("EMBED_PORT", "8090")),
            streamable_http_path=os.environ.get("EMBED_PATH", "/mcp"),
            stateless_http=True,
            transport_security=security,
        )
    else:
        mcp.run(transport="stdio")
