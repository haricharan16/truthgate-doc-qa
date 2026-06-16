"""
TruthGate CLI — Gemini-powered, free.

Usage:
  python run_query.py "How does Airflow schedule DAGs?"
  python run_query.py "Why does Airflow store indexes in JSON?"
  python run_query.py --verbose "What is the CeleryExecutor?"
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def main():
    parser = argparse.ArgumentParser(description="TruthGate — Airflow docs QA (free Gemini)")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.question:
        console.print("[bold red]Error:[/] Provide a question.")
        console.print('Example: python run_query.py "What is a DAG?"')
        sys.exit(1)

    console.print("[dim]Loading pipeline...[/dim]", end="\r")
    try:
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.bm25 import BM25Index
        from src.retrieval.embedder import Embedder
        from src.retrieval.hybrid import HybridRetriever
        from src.core.pipeline import TruthGatePipeline

        vs = VectorStore()
        bm25_path = Path("./data/bm25_index.pkl")

        if vs.is_empty():
            console.print("[red]Error:[/] Run [bold]python scripts/ingest.py[/bold] first.")
            sys.exit(1)
        if not bm25_path.exists():
            console.print("[red]Error:[/] BM25 index missing. Run [bold]python scripts/ingest.py[/bold] first.")
            sys.exit(1)

        pipeline = TruthGatePipeline(
            HybridRetriever(vs, BM25Index.load(bm25_path), Embedder())
        )
    except Exception as e:
        console.print(f"[red]Pipeline load failed:[/] {e}")
        sys.exit(1)

    console.print(f"[dim]Querying...[/dim]", end="\r")
    result = pipeline.query(args.question)
    console.clear()

    console.print(Panel(f"[bold]{args.question}[/bold]", title="[cyan]Question[/cyan]", border_style="cyan"))

    if result.response_type == "ANSWERED":
        console.print(Panel(result.answer, title="[green]✓ ANSWERED[/green]", border_style="green"))
        if result.citations:
            console.print("\n[bold]Citations:[/bold]")
            for c in result.citations:
                console.print(f"  [dim]• Source {c['source_n']}:[/dim] {c['title']}")
                console.print(f"    [link]{c['url']}[/link]")
        if result.has_hedging:
            console.print("\n[yellow]⚠ Answer contains uncertain language — verify against docs.[/yellow]")

    elif result.response_type == "NOT_IN_DOCS":
        reason_map = {
            "below_threshold": f"No sufficiently relevant content found (best score: {result.best_retrieval_score:.3f})",
            "llm_refused": "Retrieved content doesn't contain enough information to answer",
            "no_chunks": "No content retrieved",
        }
        console.print(Panel(
            f"[yellow]Cannot answer from Airflow documentation.[/yellow]\n\n"
            f"[dim]Reason: {reason_map.get(result.refusal_reason, result.refusal_reason)}[/dim]",
            title="[yellow]⊘ NOT IN DOCS[/yellow]", border_style="yellow"
        ))

    elif result.response_type == "FALSE_PREMISE":
        console.print(Panel(
            f"[red]This question contains a false premise.[/red]\n\n"
            f"[bold]Correction:[/bold] {result.correction}\n"
            f"[dim]Detected by: {result.fp_method}[/dim]",
            title="[red]✗ FALSE PREMISE[/red]", border_style="red"
        ))

    elif result.response_type == "ERROR":
        console.print(Panel(f"[red]{result.answer}[/red]", title="[red]ERROR[/red]", border_style="red"))

    # Stats
    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("k", style="dim")
    t.add_column("v")
    t.add_row("Model", "gemini-1.5-flash (free)")
    t.add_row("Cost", f"${result.total_cost_usd:.5f}  (free tier = $0.00)")
    t.add_row("Latency", f"{result.total_latency_ms:.0f}ms")
    t.add_row("Chunks retrieved", str(result.chunks_retrieved))
    if result.best_retrieval_score:
        t.add_row("Best similarity", f"{result.best_retrieval_score:.3f}")
    console.print(t)


if __name__ == "__main__":
    main()
