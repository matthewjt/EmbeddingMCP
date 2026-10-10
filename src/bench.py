import argparse
import base64
import json
import statistics
import subprocess
import threading
import time

import httpx2

TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "A fast fox leaps over a sleepy dog",
    "Neural networks learn distributed vector representations",
    "Embeddings place similar meanings close together in space",
    "The stock market rose sharply after the earnings report",
    "Quarterly revenue increased by fifteen percent year over year",
    "Python is a high-level programming language emphasizing readability",
    "Java runs on the JVM and compiles to bytecode",
    "Photosynthesis converts sunlight into chemical energy in plants",
    "Cellular respiration releases energy from glucose in mitochondria",
    "The orchestra tuned its instruments before the conductor arrived",
    "Jazz musicians improvise over chord changes in real time",
    "Continental drift slowly moves tectonic plates across the mantle",
    "Earthquakes release stored elastic energy along fault lines",
    "The chef simmered the sauce and seasoned it with salt",
    "Baking bread requires flour, water, yeast, and patience",
    "A satellite orbits the planet and transmits telemetry data",
    "Radio telescopes detect faint signals from distant galaxies",
    "The marathon runner trained through the cold winter months",
    "Swimming laps improves cardiovascular endurance steadily",
    "Democracy depends on an informed and engaged citizenry",
    "Ancient Rome built roads that connected three continents",
    "The violinist played a Bach partita without accompaniment",
    "Honeybees communicate the location of flowers by dancing",
    "Quantum computers exploit superposition and entanglement",
    "The librarian catalogued the new arrivals by author and year",
    "Volcanic eruptions shape islands and enrich surrounding soil",
    "The court adjourned until Monday morning",
    "Trains arrive at the station exactly on the hour",
    "The old lighthouse guided ships through the fog",
    "Bees make honey from the nectar of flowers",
    "The meeting was scheduled for next Tuesday afternoon",
]

RED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip()
            try:
                self.samples.append(int(out.splitlines()[0]))
            except ValueError:
                pass
            time.sleep(0.5)

    def stop(self):
        self._stop.set()
        return max(self.samples) if self.samples else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bin", required=True, help="path to llama-server binary")
    ap.add_argument("--quant", default="UD-Q4_K_XL")
    ap.add_argument("--mmproj", default="Q8_0")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--extra", default="")
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    model_arg = f"unsloth/embeddinggemma-2-GGUF:{args.quant}"
    binpart = [args.bin] + (["serve"] if args.bin.endswith("/llama") else [])
    cmd = (
        f"{' '.join(binpart)} -hf {model_arg} --embedding --host 127.0.0.1 --port {args.port} "
        f"--ctx-size 8192 --parallel 4 -ub 2048 {args.extra}"
    ).split()
    log = open(f"/tmp/bench_{args.name}.log", "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=lambda: __import__("os").setsid())

    def embed(payload):
        r = httpx2.post(f"{url}/v1/embeddings", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                if httpx2.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("server did not start")

        sampler = VramSampler()
        sampler.start()

        t0 = time.perf_counter()
        embed({"input": [TEXTS[0]]})
        cold_ms = int((time.perf_counter() - t0) * 1000)

        warm = []
        for i in range(10):
            t0 = time.perf_counter()
            embed({"input": [TEXTS[i % len(TEXTS)]]})
            warm.append((time.perf_counter() - t0) * 1000)

        throughput = {}
        for batch in (1, 8, 32):
            t0 = time.perf_counter()
            embed({"input": TEXTS[:batch]})
            dt = time.perf_counter() - t0
            throughput[batch] = round(batch / dt, 2)

        img = base64.b64encode(RED_PNG).decode()
        img_payload = {"input": [{"content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}]}
        embed(img_payload)
        img_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            embed(img_payload)
            img_times.append((time.perf_counter() - t0) * 1000)

        peak_vram = sampler.stop()
        result = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host_id": "omarchy-embedding",
            "gpu": "GTX 1050 Ti 4GB (driver 580.178.04)",
            "config_name": args.name,
            "binary": args.bin,
            "model_tag": model_arg,
            "mmproj": args.mmproj,
            "extra_args": args.extra,
            "cold_load_ms": cold_ms,
            "warm_text_ms_p50": round(statistics.median(warm), 1),
            "warm_text_ms_p95": round(sorted(warm)[9], 1),
            "throughput_items_per_s": throughput,
            "warm_image_ms_p50": round(statistics.median(img_times), 1),
            "peak_vram_mb_sampled": peak_vram,
            "success": True,
        }
    except Exception as exc:
        result = {"config_name": args.name, "success": False, "error": str(exc)[:300]}
    finally:
        subprocess.run(["kill", "-TERM", str(proc.pid)], capture_output=True)

    with open("/tmp/bench_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
