import asyncio
import base64
import json
import os
import sys

import httpx2
from mcp import Client
from mcp.client.stdio import StdioServerParameters


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[os.environ.get("EMBED_SERVER_PY", "src/server.py")])
    async with Client(params) as client:
        tools = await client.list_tools()
        print("TOOLS:", [t.name for t in tools.tools])

        res = await client.call_tool("embedding_health", {})
        payload = json.loads(res.content[0].text)
        print("HEALTH:", json.dumps(payload)[:200])
        assert "error" not in payload, payload

        res = await client.call_tool(
            "embed_text",
            {"texts": ["the cat sits on the mat", "a feline rests on a mat", "unrelated tax law paragraph"]},
        )
        payload = json.loads(res.content[0].text)
        vecs = payload["embeddings"]
        cos = lambda a, b: round(sum(x * y for x, y in zip(a, b)), 4)  # noqa: E731
        print("TEXT: dim", payload["dimensions"], "cos(1,2)", cos(vecs[0], vecs[1]), "cos(1,3)", cos(vecs[0], vecs[2]))
        assert payload["dimensions"] == 768 and cos(vecs[0], vecs[1]) > cos(vecs[0], vecs[2])

        img = base64.b64encode(open("/tmp/red.png", "rb").read()).decode()
        res = await client.call_tool("embed_image", {"image_base64": img})
        payload = json.loads(res.content[0].text)
        print("IMAGE: dim", payload.get("dimensions"), payload.get("error", "ok"))
        assert payload.get("dimensions") == 768

        aud = base64.b64encode(open("/tmp/hum_low.wav", "rb").read()).decode()
        res = await client.call_tool("embed_audio", {"audio_base64": aud})
        payload = json.loads(res.content[0].text)
        print("AUDIO: dim", payload.get("dimensions"), payload.get("error", "ok"))
        assert payload.get("dimensions") == 768

        res = await client.call_tool("compare_texts", {"text_a": "hello world", "text_b": "hello world"})
        payload = json.loads(res.content[0].text)
        print("COMPARE:", payload["cosine_similarity"])
        assert payload["cosine_similarity"] > 0.99

    print("ALL_MCP_TESTS_PASSED")


asyncio.run(main())
