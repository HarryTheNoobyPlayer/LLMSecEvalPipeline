"""Smoke test for the code generator.

Picks a small sample of prompts from ``results/data_loader_output/prompts_clean.jsonl``,
runs the OllamaGenerator against them, and pretty-prints the generated code
side-by-side with the original prompt.

Assumes Ollama is reachable at the host configured in ``config.yaml``
(default: ``http://localhost:11434``).  When developing against a remote GPU
host, open an SSH port-forward first:

    ssh -L 11434:localhost:11434 your-gpu-host

Usage:
    python scripts/smoke_test_codegen.py [N]

where N is the number of prompts to sample (default 5).
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from llmseceval.code_generator import CodeGeneratorRunner, OllamaGenerator
from llmseceval.config import load_config

ROOT = Path(__file__).parent.parent
PROMPTS_PATH = ROOT / "results" / "data_loader_output" / "prompts_clean.jsonl"
SMOKE_DIR = ROOT / "results" / "codegen_smoke"
SAMPLE_FILE = SMOKE_DIR / "sample_prompts.jsonl"


def sample_prompts(source: Path, n: int, seed: int = 42) -> list[dict]:
    """Read all prompts from *source*, return a deterministic random sample of size *n*."""
    if not source.exists():
        raise FileNotFoundError(f"Prompts file not found: {source}. Run the data loader first.")
    with open(source, "r", encoding="utf-8") as fh:
        all_prompts = [json.loads(line) for line in fh if line.strip()]
    rng = random.Random(seed)
    return rng.sample(all_prompts, min(n, len(all_prompts)))


def write_sample(prompts: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")


def print_results(results_jsonl: Path, console: Console) -> None:
    with open(results_jsonl, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    for i, rec in enumerate(records, 1):
        header = f"[bold cyan]#{i}  prompt_id={rec['prompt_id']}[/] | "
        if rec["error"]:
            header += f"[bold red]ERROR: {rec['error']}[/]"
        else:
            header += (
                f"tokens={rec.get('token_count')}  "
                f"time={rec.get('generation_time_s', 0):.2f}s"
            )
        console.rule(header)

        prompt_text = rec["prompt_text"]
        if len(prompt_text) > 600:
            prompt_text = prompt_text[:600] + "  […truncated]"
        console.print(Panel(prompt_text, title="prompt", border_style="blue"))

        code = rec.get("generated_code") or "(none)"
        if rec.get("generated_code"):
            console.print(Panel(Syntax(code, "python", theme="monokai", line_numbers=True),
                                title="generated_code", border_style="green"))
        else:
            console.print(Panel(code, title="generated_code", border_style="red"))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    console = Console()

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    cfg = load_config(ROOT / "config.yaml")
    console.print(
        f"[bold]Backend:[/] {cfg.code_generator.backend}  "
        f"[bold]Host:[/] {cfg.code_generator.ollama_host}  "
        f"[bold]Model:[/] {cfg.code_generator.model_name}",
    )
    console.print(f"[dim]Sampling {n} prompts from {PROMPTS_PATH.name}[/]")

    sample = sample_prompts(PROMPTS_PATH, n)
    write_sample(sample, SAMPLE_FILE)
    console.print(f"[dim]Wrote sample → {SAMPLE_FILE}[/]\n")

    # Clear any stale output so we always generate fresh in a smoke test.
    out_jsonl = SMOKE_DIR / "generated_code.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()

    generator = OllamaGenerator(cfg.code_generator)
    runner = CodeGeneratorRunner(generator, cfg.code_generator)
    runner.run(prompts_jsonl=SAMPLE_FILE, output_dir=SMOKE_DIR)

    console.print()
    print_results(out_jsonl, console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
