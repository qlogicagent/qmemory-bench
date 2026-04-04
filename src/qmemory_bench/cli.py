"""CLI entry point for QMemory Benchmark.

Usage:
    qmemory-bench run [--provider deepseek --api-key sk-x --target http://localhost:18800]
    qmemory-bench ui
    qmemory-bench list-datasets
"""

from __future__ import annotations

import asyncio

import click


@click.group()
@click.version_option(package_name="qmemory-bench")
def main() -> None:
    """QMemory Benchmark — multi-dimensional memory evaluation suite."""


@main.command()
@click.option("--target", default="http://localhost:18800", help="QMemory server URL")
@click.option("--provider", default="deepseek", help="LLM provider for judge (deepseek/openai/zhipu/kimi/qwen/doubao/minimax)")
@click.option("--api-key", default="", help="LLM API key (use comma-separated for multiple keys)")
@click.option("--model", default="", help="LLM model override")
@click.option("--scale", default="standard", type=click.Choice(["micro", "quick", "standard", "full"]), help="Evaluation scale (micro: 20 diagnostic questions)")
@click.option("--preset", default="public-main", help="Dataset preset: public-main/release-full/supporting/regression/all")
@click.option("--datasets", default="", help="Comma-separated dataset names (overrides --preset)")
@click.option("--output", "-o", default="", help="Output report file (JSON)")
@click.option("--skip-ingest", is_flag=True, default=False, help="Skip inject & cleanup, reuse existing DB memories")
@click.option("--eval-user", default="", help="Fixed eval user_id prefix (for --skip-ingest reuse)")
@click.option("--no-cleanup", is_flag=True, default=False, help="Keep data after run (for creating golden DB)")
@click.option("--max-per-category", default=0, type=int, help="Cap questions per category (0=unlimited, e.g. 10 for medium-scale)")
def run(
    target: str,
    provider: str,
    api_key: str,
    model: str,
    scale: str,
    preset: str,
    datasets: str,
    output: str,
    skip_ingest: bool,
    eval_user: str,
    no_cleanup: bool,
    max_per_category: int,
) -> None:
    """Run benchmark evaluation against a QMemory server."""
    from qmemory_bench.runner import BenchmarkConfig, run_benchmark

    if not api_key:
        click.echo("Error: --api-key is required for LLM judge scoring", err=True)
        raise SystemExit(1)

    # Support comma-separated multiple API keys
    keys = [k.strip() for k in api_key.split(",") if k.strip()]
    primary_key = keys[0]
    extra_keys = keys[1:] if len(keys) > 1 else []

    config = BenchmarkConfig(
        target_url=target,
        provider=provider,
        api_key=primary_key,
        api_keys=extra_keys,
        model=model,
        scale=scale,
        dataset_names=[d.strip() for d in datasets.split(",") if d.strip()],
        dataset_preset=preset,
        output_path=output or None,
        skip_ingest=skip_ingest,
        eval_user_base=eval_user or "",
        no_cleanup=no_cleanup,
        max_per_category=max_per_category,
    )

    click.echo(f"QMemory Benchmark v{_get_version()}")
    click.echo(f"  Target: {target}")
    click.echo(f"  Provider: {provider}")
    click.echo(f"  Scale: {scale}")
    click.echo(f"  Preset: {preset}")
    click.echo(f"  Datasets: {datasets or '(from preset)'}")
    if skip_ingest:
        click.echo(f"  Skip Ingest: YES (eval_user={eval_user or 'auto'})")
    click.echo()

    report = asyncio.run(run_benchmark(config))

    # Print results
    from qmemory_bench.runner import print_report
    print_report(report)

    if output:
        click.echo(f"\nReport saved to {output}")


@main.command()
@click.option("--port", default=8090, type=int, help="UI server port")
@click.option("--target", default="http://localhost:18800", help="Default QMemory URL")
def ui(port: int, target: str) -> None:
    """Launch the benchmark UI (NiceGUI web app)."""
    click.echo(f"Starting QMemory Benchmark UI on http://localhost:{port}")
    from qmemory_bench.ui.app import launch_ui
    launch_ui(port=port, default_target=target)


@main.command(name="list-datasets")
def list_datasets() -> None:
    """List available evaluation datasets."""
    from qmemory_bench.dataset import AVAILABLE_DATASETS, DATASET_GROUP_LABELS

    click.echo("Available datasets:")
    for name, info in AVAILABLE_DATASETS.items():
        click.echo(f"  {name:25s} {info['description']}")
        click.echo(
            f"  {'':25s} Questions: {info['question_count']} | "
            f"Group: {DATASET_GROUP_LABELS.get(info.get('tier', 'supporting'), info.get('tier', 'supporting'))}"
        )


def _get_version() -> str:
    from qmemory_bench import __version__
    return __version__
