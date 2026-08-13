"""Score adapter-produced HorizonBench JSONL traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Supports both `python file.py` and module execution.
    from .score import read_jsonl, score_results
except ImportError:  # pragma: no cover - direct-script compatibility
    from score import read_jsonl, score_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score JSONL episode outcomes against HorizonBench metrics")
    parser.add_argument("input", type=Path, help="one JSON object per completed benchmark task")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    metrics = score_results(read_jsonl(args.input)).as_dict()
    encoded = json.dumps(metrics, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
