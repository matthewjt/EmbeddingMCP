import asyncio
import json
import sqlite3
import sys

import numpy as np
from mcp import Client

URL = "http://10.223.223.120:8090/mcp"


async def query(db_path, q, k=8):
    async with Client(URL) as c:
        r = await c.call_tool("embed_text", {"texts": [q]})
        qv = np.asarray(json.loads(r.content[0].text)["embeddings"][0], dtype=np.float32)
    rows = sqlite3.connect(db_path).execute("SELECT chapter,text,vec FROM chunks").fetchall()
    scored = []
    for i, (ch, t, blob) in enumerate(rows):
        v = np.frombuffer(blob, dtype=np.float32)
        scored.append((float(qv @ v), i, ch, t))
    scored.sort(key=lambda x: -x[0])
    for s, i, ch, t in scored[:k]:
        print(f"[{s:.3f}] chunk {i} ({ch}): {t[:220]}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(query(sys.argv[1], " ".join(sys.argv[2:])))
