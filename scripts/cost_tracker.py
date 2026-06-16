import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def print_summary(entries: list[dict]):
    if not entries:
        print("No entries in cost log.")
        return

    total = len(entries)
    total_cost = sum(e["cost"] for e in entries)
    latencies = sorted(e["latency_ms"] for e in entries)
    p95 = latencies[int(total * 0.95)] if total > 1 else latencies[0]

    by_type = defaultdict(lambda: {"count": 0, "cost": 0.0, "latency_sum": 0.0})
    for e in entries:
        t = e["type"]
        by_type[t]["count"] += 1
        by_type[t]["cost"] += e["cost"]
        by_type[t]["latency_sum"] += e["latency_ms"]

    print("\n=== TruthGate Cost Report ===")
    print(f"Total queries:    {total}")
    print(f"Total cost:       ${total_cost:.4f}")
    print(f"Mean cost/query:  ${total_cost/total:.6f}")
    print(f"Budget limit:     $0.020000  ({'✓ PASS' if total_cost/total <= 0.02 else '✗ FAIL'})")
    print(f"Mean latency:     {sum(latencies)/total:.0f}ms")
    print(f"p95 latency:      {p95:.0f}ms  ({'✓ PASS' if p95 <= 8000 else '✗ FAIL'})")
    print(f"\nBy response type:")
    for t, stats in sorted(by_type.items()):
        mean_lat = stats["latency_sum"] / stats["count"]
        mean_cost = stats["cost"] / stats["count"]
        print(f"  {t:<15} count={stats['count']}  mean_cost=${mean_cost:.6f}  mean_lat={mean_lat:.0f}ms")


def print_detail(entries: list[dict]):
    print("\n=== Per-Query Detail ===")
    for i, e in enumerate(entries):
        print(f"{i+1:3}. [{e['type']:<13}] cost=${e['cost']:.5f}  lat={e['latency_ms']:.0f}ms  q={e['question'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="./logs/cost_log.jsonl")
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    entries = load_log(Path(args.log))
    print_summary(entries)
    if args.detail:
        print_detail(entries)
