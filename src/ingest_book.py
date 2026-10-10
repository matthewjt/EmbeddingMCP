import asyncio
import json
import re
import sqlite3
import sys

import numpy as np
from mcp import Client

URL = "http://10.223.223.120:8090/mcp"
WORDS_PER_CHUNK = 400
OVERLAP_WORDS = 60

CHAPTER_RE = re.compile(r"^(Chapter\s+\d+[^\n]*|CHAPTER\s+\d+[^\n]*)$", re.M)


def chunk_text(text):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf, words = [], [], 0
    chapter = "front-matter"
    for p in paras:
        m = CHAPTER_RE.match(p.split("\n")[0])
        if m:
            if buf:
                chunks.append((chapter, " ".join(buf)))
                buf, words = [], 0
            chapter = m.group(1).strip()
        buf.append(p)
        words += len(p.split())
        if words >= WORDS_PER_CHUNK:
            chunks.append((chapter, " ".join(buf)))
            tail = " ".join(buf).split()[-OVERLAP_WORDS:]
            buf, words = [" ".join(tail)], OVERLAP_WORDS
    if buf:
        chunks.append((chapter, " ".join(buf)))
    return chunks


async def embed_batch(client, texts):
    r = await client.call_tool("embed_text", {"texts": texts})
    d = json.loads(r.content[0].text)
    if len(d["embeddings"]) != len(texts):
        raise RuntimeError(f"batch mismatch: sent {len(texts)} got {len(d['vectors'])}")
    return d["embeddings"]


async def main(path, db_path):
    text = open(path, encoding="utf-8", errors="replace").read()
    chunks = chunk_text(text)
    print(f"chunks: {len(chunks)}")
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, chapter TEXT, text TEXT, vec BLOB)")
    db.execute("DELETE FROM chunks")
    async with Client(URL) as c:
        for i in range(0, len(chunks), 8):
            batch = chunks[i : i + 8]
            vecs = await embed_batch(c, [t for _, t in batch])
            for (ch, t), v in zip(batch, vecs):
                db.execute("INSERT INTO chunks(chapter,text,vec) VALUES(?,?,?)",
                           (ch, t, np.asarray(v, dtype=np.float32).tobytes()))
            print(f"embedded {min(i+8, len(chunks))}/{len(chunks)}", flush=True)
    db.commit()
    print("INGEST_COMPLETE", len(chunks), "chunks")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
