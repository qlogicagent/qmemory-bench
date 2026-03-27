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
@click.option("--api-key", default="", help="LLM API key")
@click.option("--model", default="", help="LLM model override")
@click.option("--scale", default="standard", type=click.Choice(["quick", "standard", "full"]), help="Evaluation scale")
@click.option("--preset", default="public-main", help="Dataset preset: public-main/release-full/supporting/regression/all")
@click.option("--datasets", default="", help="Comma-separated dataset names (overrides --preset)")
@click.option("--output", "-o", default="", help="Output report file (JSON)")
def run(
    target: str,
    provider: str,
    api_key: str,
    model: str,
    scale: str,
    preset: str,
    datasets: str,
    output: str,
) -> None:
    """Run benchmark evaluation against a QMemory server."""
    from qmemory_bench.runner import BenchmarkConfig, run_benchmark

    if not api_key:
        click.echo("Error: --api-key is required for LLM judge scoring", err=True)
        raise SystemExit(1)

    config = BenchmarkConfig(
        target_url=target,
        provider=provider,
        api_key=api_key,
        model=model,
        scale=scale,
        dataset_names=[d.strip() for d in datasets.split(",") if d.strip()],
        dataset_preset=preset,
        output_path=output or None,
    )

    click.echo(f"QMemory Benchmark v{_get_version()}")
    click.echo(f"  Target: {target}")
    click.echo(f"  Provider: {provider}")
    click.echo(f"  Scale: {scale}")
    click.echo(f"  Preset: {preset}")
    click.echo(f"  Datasets: {datasets or '(from preset)'}")
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
