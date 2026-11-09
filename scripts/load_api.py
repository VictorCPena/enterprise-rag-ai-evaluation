"""Bounded localhost HTTP load and process restart study. No production SLA claim."""

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import requests


def run(module, payload, output, per_level=24):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}"
    logs = output.with_suffix(".server.log")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")

    def start():
        file = logs.open("a")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                module,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            stdout=file,
            stderr=file,
            env=env,
        )
        for _ in range(180):
            if proc.poll() is not None:
                raise RuntimeError("Server startup failed; inspect server log")
            try:
                if requests.get(url + "/health", timeout=2).status_code == 200:
                    return proc, file
            except requests.RequestException:
                pass
            time.sleep(0.25)
        proc.terminate()
        proc.wait(timeout=10)
        file.close()
        raise TimeoutError("Health deadline")

    proc, file = start()
    records = []
    endpoint = "/predict" if module.startswith("forecasting") else "/query"

    def request(_):
        began = time.perf_counter()
        try:
            response = requests.post(url + endpoint, json=payload, timeout=30)
            return {"ms": 1000 * (time.perf_counter() - began), "status": response.status_code}
        except requests.RequestException:
            return {"ms": 1000 * (time.perf_counter() - began), "status": 0}

    try:
        for concurrency in [1, 4, 8]:
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(request, range(per_level)))
            seconds = time.perf_counter() - started
            latencies = [r["ms"] for r in results]
            records.append(
                {
                    "concurrency": concurrency,
                    "requests": per_level,
                    "seconds": seconds,
                    "requests_per_second": per_level / seconds,
                    "p50_ms": float(np.median(latencies)),
                    "p95_ms": float(np.quantile(latencies, 0.95)),
                    "max_ms": float(max(latencies)),
                    "errors": sum(r["status"] != 200 for r in results),
                    "statuses": {
                        str(status): sum(r["status"] == status for r in results)
                        for status in set(r["status"] for r in results)
                    },
                }
            )
    finally:
        proc.terminate()
        proc.wait(timeout=20)
        file.close()
    recovered, file = start()
    try:
        recovery = requests.post(url + endpoint, json=payload, timeout=30).status_code == 200
    finally:
        recovered.terminate()
        recovered.wait(timeout=20)
        file.close()
    import resource

    report = {
        "endpoint": endpoint,
        "transport": "actual HTTP on localhost",
        "levels": records,
        "restart_recovery": recovery,
        "server_peak_rss_mb_max_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        "limits": "Local bounded study; no sustained saturation, distributed availability or production SLA demonstrated",
    }
    output.write_text(json.dumps(report, indent=2))
    if not recovery or any(row["errors"] for row in records):
        raise RuntimeError("HTTP study failed; inspect the recorded error/recovery counts")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)
    a = p.parse_args()
    if a.module.startswith("forecasting"):
        import pandas as pd

        d = pd.read_parquet("data/tlc_hourly.parquet")
        zone = int(d.zone.iloc[0])
        g = d[d.zone == zone].tail(200)
        payload = {
            "zone": zone,
            "horizon": 1,
            "history": [{"time": str(r.time), "count": float(r.count)} for r in g.itertuples()],
        }
    else:
        payload = {
            "text": "Why is loading a pickle model from an untrusted source unsafe?",
            "k": 5,
            "strategy": "hybrid",
        }
    print(json.dumps(run(a.module, payload, Path("evidence/http_load.json")), indent=2))
