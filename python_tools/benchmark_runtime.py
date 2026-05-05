#!/usr/bin/env python3
"""Run the C++ runtime repeatedly and summarize JSONL latency logs."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark edge_ai_runtime")
    parser.add_argument("--exe", required=True, help="Path to compiled edge_ai_runtime executable")
    parser.add_argument("--jobs", required=True, help="Path to jobs JSON file")
    parser.add_argument("--runs", type=int, default=10, help="Number of invocations")
    parser.add_argument("--schedule", default="priority", choices=["fifo", "priority"])
    parser.add_argument("--log", default="logs/inference_metrics.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    if log_path.exists():
        log_path.unlink()

    command = [args.exe, "--jobs", args.jobs, "--schedule", args.schedule, "--metrics"]
    for run in range(args.runs):
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"Run {run + 1} failed:\n{completed.stderr}")

    records = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    successes = [record for record in records if record.get("status") == "success"]
    latencies = [float(record["latency_ms"]) for record in successes]
    success_rate = len(successes) / len(records) if records else 0.0

    print(f"runs: {args.runs}")
    print(f"log_records: {len(records)}")
    print(f"success_rate: {success_rate:.3f}")
    if latencies:
        print(f"average_latency_ms: {statistics.mean(latencies):.3f}")
        print(f"p95_latency_ms: {statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies):.3f}")
    else:
        print("average_latency_ms: 0.000")
        print("p95_latency_ms: 0.000")


if __name__ == "__main__":
    main()

