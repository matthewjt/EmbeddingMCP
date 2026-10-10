import json
import sys

import httpx2

URL = "http://127.0.0.1:8081/v1/embeddings"

CORPUS = {
    "d01": "The cat sat on the windowsill watching birds outside",
    "d02": "A feline rested on the mat by the front door",
    "d03": "Python code uses indentation to define blocks",
    "d04": "Java virtual machine executes compiled bytecode",
    "d05": "Photosynthesis converts light into chemical energy in leaves",
    "d06": "Plants synthesize sugars from carbon dioxide and water",
    "d07": "The stock market rallied after the earnings beat expectations",
    "d08": "Equities climbed as quarterly profits exceeded analyst estimates",
    "d09": "The marathon covered 42 kilometers through the city streets",
    "d10": "Long distance running demands steady training and nutrition",
    "d11": "The orchestra rehearsed the symphony before the premiere",
    "d12": "Musicians tuned their string instruments in the concert hall",
    "d13": "Volcanic eruptions reshape coastlines and enrich island soil",
    "d14": "Lava flows cool into basalt rock over time",
    "d15": "The chef reduced the sauce and finished it with butter",
    "d16": "Simmering a sauce concentrates its flavor",
    "d17": "Satellites relay telemetry from orbit around the Earth",
    "d18": "Radio telescopes capture faint signals from distant galaxies",
    "d19": "Democracy relies on free press and informed voters",
    "d20": "Ancient Rome connected its provinces with paved roads",
    "d21": "Bees perform a waggle dance to share flower locations",
    "d22": "Honey is made from flower nectar collected by worker bees",
    "d23": "The train departed the platform exactly at noon",
    "d24": "Quantum devices exploit superposition and entanglement",
}

QUERIES = {
    "q1": ("a pet animal resting indoors", ["d01", "d02"]),
    "q2": ("programming language syntax", ["d03", "d04"]),
    "q3": ("how plants make food from sunlight", ["d05", "d06"]),
    "q4": ("financial markets going up", ["d07", "d08"]),
    "q5": ("live music performance", ["d11", "d12"]),
    "q6": ("cooking on the stove", ["d15", "d16"]),
    "q7": ("bee behavior and honey", ["d21", "d22"]),
    "q8": ("space observation and communication", ["d17", "d18"]),
}


def embed(texts):
    out = []
    for i in range(0, len(texts), 8):
        r = httpx2.post(URL, json={"input": texts[i : i + 8]}, timeout=120)
        r.raise_for_status()
        got = [item["embedding"] for item in r.json()["data"]]
        if len(got) != len(texts[i : i + 8]):
            raise RuntimeError(f"expected {len(texts[i:i+8])} vectors, got {len(got)}")
        out.extend(got)
    return out


def main():
    ids = sorted(CORPUS)
    vecs = {}
    docs = embed([CORPUS[i] for i in ids])
    for i, v in zip(ids, docs):
        vecs[i] = v

    per_query = {}
    mrrs, ndcgs, rec1, rec3 = [], [], 0, 0
    for q, (text, rel) in QUERIES.items():
        qv = embed([text])[0]
        ranked = sorted(ids, key=lambda d: -sum(a * b for a, b in zip(qv, vecs[d])))
        ranks = [ranked.index(r) + 1 for r in rel]
        best = min(ranks)
        rec1 += 1 if best == 1 else 0
        rec3 += 1 if best <= 3 else 0
        mrrs.append(1.0 / best)
        dcg = sum(1.0 / (r ** 0.5) for r in ranks[:10])
        idcg = sum(1.0 / ((i + 1) ** 0.5) for i in range(len(rel)))
        ndcgs.append(dcg / idcg)
        per_query[q] = {"best_rank": best, "ranks": ranks}

    n = len(QUERIES)
    print(json.dumps({
        "model": sys.argv[1] if len(sys.argv) > 1 else "?",
        "recall_at_1": round(rec1 / n, 3),
        "recall_at_3": round(rec3 / n, 3),
        "mrr": round(sum(mrrs) / n, 3),
        "ndcg_at_10": round(sum(ndcgs) / n, 3),
        "per_query": per_query,
    }, indent=2))


if __name__ == "__main__":
    main()
