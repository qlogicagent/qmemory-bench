#!/usr/bin/env python3
"""delta_report.py — Compare any two benchmark report JSONs per-question.

Usage:
  python scripts/delta_report.py OLD.json NEW.json
  python scripts/delta_report.py OLD.json NEW.json -o delta.json
  python scripts/delta_report.py OLD.json NEW.json --failures-only
  python scripts/delta_report.py OLD.json NEW.json --threshold 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def build_qmap(data: dict) -> dict[tuple[str, str], dict]:
    """Build (dataset, question_id) → result dict from a benchmark report."""
    qmap = {}
    for ds_name, ds_val in data.get("datasets", {}).items():
        if not isinstance(ds_val, dict):
            continue
        for r in ds_val.get("results", []):
            qid = r.get("question_id", "")
            qmap[(ds_name, qid)] = r
    return qmap


def compute_deltas(old_data: dict, new_data: dict) -> list[dict]:
    """Compute per-question deltas between two reports."""
    old_q = build_qmap(old_data)
    new_q = build_qmap(new_data)

    all_keys = sorted(set(old_q.keys()) & set(new_q.keys()))
    deltas = []
    for key in all_keys:
        old_r = old_q[key]
        new_r = new_q[key]
        old_s = old_r.get("score", 0)
        new_s = new_r.get("score", 0)
        deltas.append({
            "ds": key[0],
            "qid": key[1],
            "query": old_r.get("query", ""),
            "old_score": old_s,
            "new_score": new_s,
            "delta": new_s - old_s,
            "category": old_r.get("category", ""),
            "expected": old_r.get("expected", ""),
            "old_answer": old_r.get("answer", ""),
            "new_answer": new_r.get("answer", ""),
            "old_reason": old_r.get("reason", ""),
            "new_reason": new_r.get("reason", ""),
        })
    deltas.sort(key=lambda x: x["delta"])
    return deltas


def print_summary(deltas: list[dict], old_label: str, new_label: str) -> None:
    """Print summary statistics."""
    improved = [d for d in deltas if d["delta"] > 0]
    regressed = [d for d in deltas if d["delta"] < 0]
    unchanged = [d for d in deltas if d["delta"] == 0]

    print(f"\n{'=' * 70}")
    print(f"DELTA REPORT: {old_label} → {new_label}")
    print(f"{'=' * 70}")
    print(f"Matched questions: {len(deltas)}")
    print(f"Improved:  {len(improved):>3} ({sum(d['delta'] for d in improved):+d} total)")
    print(f"Regressed: {len(regressed):>3} ({sum(d['delta'] for d in regressed):+d} total)")
    print(f"Unchanged: {len(unchanged):>3}")
    if deltas:
        mean_d = sum(d["delta"] for d in deltas) / len(deltas)
        print(f"Mean delta: {mean_d:+.2f}")


def print_dataset_breakdown(deltas: list[dict]) -> None:
    """Print per-dataset summary."""
    ds_deltas: dict[str, list] = defaultdict(list)
    for d in deltas:
        ds_deltas[d["ds"]].append(d)

    print(f"\n{'Dataset':<20} {'N':>3} {'Up':>3} {'Down':>4} {'Same':>4} {'OldAvg':>7} {'NewAvg':>7} {'Delta':>7}")
    print("-" * 60)
    for ds_name in sorted(ds_deltas):
        dd = ds_deltas[ds_name]
        up = sum(1 for d in dd if d["delta"] > 0)
        down = sum(1 for d in dd if d["delta"] < 0)
        same = sum(1 for d in dd if d["delta"] == 0)
        old_avg = sum(d["old_score"] for d in dd) / len(dd)
        new_avg = sum(d["new_score"] for d in dd) / len(dd)
        print(f"{ds_name:<20} {len(dd):>3} {up:>3} {down:>4} {same:>4} {old_avg:>7.1f} {new_avg:>7.1f} {new_avg - old_avg:>+7.1f}")


def print_category_breakdown(deltas: list[dict]) -> None:
    """Print per-category within dataset."""
    cat_deltas: dict[tuple, list] = defaultdict(list)
    for d in deltas:
        cat_deltas[(d["ds"], d["category"])].append(d)

    print(f"\n{'Dataset/Category':<40} {'N':>3} {'Old':>6} {'New':>6} {'Delta':>7}")
    print("-" * 65)
    for (ds, cat) in sorted(cat_deltas):
        dd = cat_deltas[(ds, cat)]
        old_avg = sum(d["old_score"] for d in dd) / len(dd)
        new_avg = sum(d["new_score"] for d in dd) / len(dd)
        delta_avg = new_avg - old_avg
        marker = "  !!!" if delta_avg < -1.5 else ("  +++" if delta_avg > 1.5 else "")
        print(f"  {ds}/{cat:<35} {len(dd):>3} {old_avg:>6.1f} {new_avg:>6.1f} {delta_avg:>+7.1f}{marker}")


def print_regressions(deltas: list[dict], threshold: int = 0) -> None:
    """Print all regressions (score decreased)."""
    regressed = [d for d in deltas if d["delta"] < -threshold]
    if not regressed:
        print("\nNo regressions found.")
        return
    print(f"\n{'=' * 80}")
    print(f"REGRESSIONS ({len(regressed)} questions):")
    print("=" * 80)
    for d in regressed:
        print(f"  [{d['ds']}] {d['delta']:+d} ({d['old_score']}→{d['new_score']}) cat={d['category']}")
        print(f"    Q: {d['query'][:100]}")
        print(f"    Expected: {d['expected'][:140]}")
        print(f"    Reason: {d['new_reason'][:200]}")
        print()


def print_improvements(deltas: list[dict], threshold: int = 0) -> None:
    """Print all improvements."""
    improved = [d for d in deltas if d["delta"] > threshold]
    if not improved:
        print("\nNo improvements found.")
        return
    print(f"\n{'=' * 80}")
    print(f"IMPROVEMENTS ({len(improved)} questions):")
    print("=" * 80)
    for d in reversed(improved):
        print(f"  [{d['ds']}] {d['delta']:+d} ({d['old_score']}→{d['new_score']}) cat={d['category']}")
        print(f"    Q: {d['query'][:100]}")
        print()


def print_failures(deltas: list[dict], max_score: int = 4) -> None:
    """Print questions where new score <= max_score."""
    failures = sorted(
        [d for d in deltas if d["new_score"] <= max_score],
        key=lambda x: (x["ds"], x["new_score"]),
    )
    if not failures:
        print(f"\nNo failures (score ≤ {max_score}).")
        return
    print(f"\n{'=' * 80}")
    print(f"REMAINING FAILURES (new score ≤ {max_score}): {len(failures)} questions")
    print("=" * 80)
    for d in failures:
        changed = f" (was {d['old_score']})" if d["old_score"] != d["new_score"] else ""
        print(f"  [{d['ds']}] score={d['new_score']}{changed} cat={d['category']}")
        print(f"    Q: {d['query'][:100]}")
        print(f"    Reason: {d['new_reason'][:200]}")
        print()


def report_to_json(deltas: list[dict], old_label: str, new_label: str) -> dict:
    """Convert delta analysis to JSON."""
    improved = [d for d in deltas if d["delta"] > 0]
    regressed = [d for d in deltas if d["delta"] < 0]
    return {
        "old": old_label,
        "new": new_label,
        "total_questions": len(deltas),
        "improved": len(improved),
        "regressed": len(regressed),
        "unchanged": len(deltas) - len(improved) - len(regressed),
        "mean_delta": round(sum(d["delta"] for d in deltas) / len(deltas), 3) if deltas else 0,
        "questions": deltas,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two benchmark report JSONs")
    p.add_argument("old", help="Path to OLD report JSON")
    p.add_argument("new", help="Path to NEW report JSON")
    p.add_argument("--output", "-o", help="Save JSON delta report to file")
    p.add_argument("--failures-only", action="store_true",
                   help="Only show remaining failures")
    p.add_argument("--threshold", "-t", type=int, default=0,
                   help="Min score change to report (default: 0)")
    p.add_argument("--failure-score", type=int, default=4,
                   help="Max score to count as failure (default: 4)")
    args = p.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    if not old_path.exists():
        print(f"ERROR: {old_path} not found")
        sys.exit(1)
    if not new_path.exists():
        print(f"ERROR: {new_path} not found")
        sys.exit(1)

    old_data = json.loads(old_path.read_text(encoding="utf-8"))
    new_data = json.loads(new_path.read_text(encoding="utf-8"))
    old_label = old_path.stem
    new_label = new_path.stem

    deltas = compute_deltas(old_data, new_data)
    if not deltas:
        print("ERROR: No matching questions between the two reports.")
        sys.exit(1)

    if args.failures_only:
        print_failures(deltas, args.failure_score)
    else:
        print_summary(deltas, old_label, new_label)
        print_dataset_breakdown(deltas)
        print_category_breakdown(deltas)
        print_regressions(deltas, args.threshold)
        print_improvements(deltas, args.threshold)
        print_failures(deltas, args.failure_score)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report_to_json(deltas, old_label, new_label), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nDelta report saved to: {out_path}")


if __name__ == "__main__":
    main()
