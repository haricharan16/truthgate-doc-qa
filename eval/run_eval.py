"""
TruthGate Evaluation Harness
Run: python eval/run_eval.py
  or: make eval

Runs all 60 questions, computes metrics, prints table, saves results to JSON.

Metrics computed:
- Answer accuracy (answerable category): % correctly answered (not refused)
- Refusal precision: of questions we refused, what % were actually unanswerable
- Refusal recall: of actually unanswerable questions, what % did we refuse
- False-premise detection rate: % of false-premise questions correctly flagged
- Adversarial pass rate: % of adversarial questions handled correctly
- Mean cost per query, p95 latency
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.core.pipeline import TruthGatePipeline
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Index
from src.retrieval.embedder import Embedder

logging.basicConfig(level=logging.WARNING)  # quiet during eval
logger = logging.getLogger(__name__)


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def run_eval(pipeline: TruthGatePipeline, questions: list[dict], verbose: bool = False) -> list[dict]:
    results = []
    total = len(questions)

    print(f"\nRunning {total} questions...\n")
    print(f"{'#':>3}  {'Category':<14} {'Expected':<14} {'Got':<14} {'Score':>6}  {'Cost':>8}  {'Lat':>7}  Question")
    print("-" * 110)

    for i, q in enumerate(questions):
        question = q["question"]
        expected = q["expected_type"]
        category = q["category"]

        t0 = time.time()
        try:
            response = pipeline.query(question)
            got_type = response.response_type
            cost = response.total_cost_usd
            latency = response.total_latency_ms
        except Exception as e:
            got_type = "ERROR"
            cost = 0.0
            latency = (time.time() - t0) * 1000
            logger.error(f"Q{q['id']}: {e}")

        correct = (got_type == expected)

        # Adversarial q59: trap for false-premise — expected ANSWERED
        if q.get("is_trap_for_fp"):
            expected = "ANSWERED"
            correct = (got_type == "ANSWERED")

        status = "✓" if correct else "✗"
        score = 1 if correct else 0

        print(
            f"{q['id']:>3}  {category:<14} {expected:<14} {got_type:<14} {status:>6}  "
            f"${cost:.5f}  {latency/1000:.2f}s  {question[:55]}"
        )

        result = {
            "id": q["id"],
            "category": category,
            "question": question,
            "expected_type": expected,
            "got_type": got_type,
            "correct": correct,
            "cost_usd": cost,
            "latency_ms": latency,
            "notes": q.get("notes", ""),
        }

        if got_type == "ANSWERED":
            result["answer"] = response.answer[:200]
            result["citations"] = response.citations
        elif got_type == "NOT_IN_DOCS":
            result["refusal_reason"] = response.refusal_reason
            result["best_score"] = response.best_retrieval_score
        elif got_type == "FALSE_PREMISE":
            result["correction"] = response.correction

        results.append(result)

    return results


def compute_metrics(results: list[dict]) -> dict:
    by_cat = {
        "answerable": [r for r in results if r["category"] == "answerable"],
        "unanswerable": [r for r in results if r["category"] == "unanswerable"],
        "false_premise": [r for r in results if r["category"] == "false_premise"],
        "adversarial": [r for r in results if r["category"] == "adversarial"],
    }

    # Answer accuracy: answerable questions correctly answered
    ans = by_cat["answerable"]
    answer_accuracy = sum(1 for r in ans if r["correct"]) / len(ans) if ans else 0

    # Refusal metrics
    unans = by_cat["unanswerable"]
    refusal_recall = sum(1 for r in unans if r["got_type"] == "NOT_IN_DOCS") / len(unans) if unans else 0

    # Refusal precision: of all NOT_IN_DOCS responses, what fraction were truly unanswerable
    all_refused = [r for r in results if r["got_type"] == "NOT_IN_DOCS"]
    truly_unanswerable_refused = [r for r in all_refused if r["category"] == "unanswerable"]
    refusal_precision = len(truly_unanswerable_refused) / len(all_refused) if all_refused else 0

    refusal_f1 = (
        2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
        if (refusal_precision + refusal_recall) > 0 else 0
    )

    # False premise detection
    fp = by_cat["false_premise"]
    fp_detection_rate = sum(1 for r in fp if r["got_type"] == "FALSE_PREMISE") / len(fp) if fp else 0

    # Adversarial
    adv = by_cat["adversarial"]
    adv_pass_rate = sum(1 for r in adv if r["correct"]) / len(adv) if adv else 0

    # Cost + latency
    costs = [r["cost_usd"] for r in results]
    latencies = sorted(r["latency_ms"] for r in results)
    p95_idx = int(len(latencies) * 0.95)
    p95_lat = latencies[min(p95_idx, len(latencies)-1)]

    return {
        "answer_accuracy": answer_accuracy,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "fp_detection_rate": fp_detection_rate,
        "adversarial_pass_rate": adv_pass_rate,
        "mean_cost_usd": sum(costs) / len(costs),
        "total_cost_usd": sum(costs),
        "p95_latency_ms": p95_lat,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "total_questions": len(results),
        "correct_total": sum(1 for r in results if r["correct"]),
    }


def print_metrics(metrics: dict):
    print("\n" + "=" * 60)
    print("TRUTHGATE EVAL RESULTS")
    print("=" * 60)
    print(f"{'Metric':<35} {'Value':>10}  {'Target':>10}  Pass?")
    print("-" * 60)

    rows = [
        ("Answer Accuracy (answerable)", metrics["answer_accuracy"], 0.80, "%"),
        ("Refusal Precision", metrics["refusal_precision"], 0.80, "%"),
        ("Refusal Recall", metrics["refusal_recall"], 0.80, "%"),
        ("Refusal F1", metrics["refusal_f1"], 0.80, "%"),
        ("False Premise Detection Rate", metrics["fp_detection_rate"], 0.75, "%"),
        ("Adversarial Pass Rate", metrics["adversarial_pass_rate"], 0.40, "%"),
    ]
    for label, val, target, unit in rows:
        pct = f"{val*100:.1f}%"
        tgt = f"{target*100:.0f}%"
        ok = "✓" if val >= target else "✗"
        print(f"{label:<35} {pct:>10}  {tgt:>10}  {ok}")

    print("-" * 60)
    cost_ok = "✓" if metrics["mean_cost_usd"] <= 0.02 else "✗"
    lat_ok = "✓" if metrics["p95_latency_ms"] <= 8000 else "✗"
    print(f"{'Mean Cost/Query':<35} {'${:.5f}'.format(metrics['mean_cost_usd']):>10}  {'$0.02':>10}  {cost_ok}")
    print(f"{'p95 Latency':<35} {'{:.0f}ms'.format(metrics['p95_latency_ms']):>10}  {'8000ms':>10}  {lat_ok}")
    print(f"{'Total Cost (60 queries)':<35} {'${:.4f}'.format(metrics['total_cost_usd']):>10}")
    print(f"{'Overall Correct':<35} {'{}/{}'.format(metrics['correct_total'], metrics['total_questions']):>10}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TruthGate Eval Harness")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--output", default="eval/results_latest.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--category", help="Run only this category")
    args = parser.parse_args()

    # Load pipeline
    print("Loading pipeline...")
    bm25_path = Path("./data/bm25_index.pkl")
    vs = VectorStore()

    if vs.is_empty():
        print("ERROR: Vector store is empty. Run `make ingest` first.")
        sys.exit(1)
    if not bm25_path.exists():
        print("ERROR: BM25 index not found. Run `make ingest` first.")
        sys.exit(1)

    bm25 = BM25Index.load(bm25_path)
    embedder = Embedder()
    retriever = HybridRetriever(vs, bm25, embedder)
    pipeline = TruthGatePipeline(retriever)
    print(f"Pipeline ready. Index has {vs.count()} chunks.")

    # Load questions
    questions = load_questions(Path(args.questions))
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
        print(f"Running {len(questions)} questions from category: {args.category}")

    # Run eval
    t_start = time.time()
    results = run_eval(pipeline, questions, verbose=args.verbose)
    total_time = time.time() - t_start

    print(f"\nCompleted {len(results)} questions in {total_time:.1f}s")

    # Compute + print metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_time_s": total_time,
        "metrics": metrics,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
