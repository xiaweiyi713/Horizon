"""CLI for fitting Horizon's dependency-free state-decay predictor.

Input is JSONL. Each line must contain:

``{"signals": {...}, "decayed": true}``

Use a run/model-disjoint validation file for reported metrics. This tool writes
only a model artifact; applying the model remains a separate durable runtime
operation through :class:`LearnedInterventionPolicy`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .learned import DecayModelError, DecayTrainingExample, LogisticDecayPredictor


def read_examples(path: Path) -> list[DecayTrainingExample]:
    rows: list[DecayTrainingExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise DecayModelError("invalid JSON on line {} of {}: {}".format(line_number, path, error)) from error
        try:
            rows.append(DecayTrainingExample.from_mapping(raw))
        except DecayModelError as error:
            raise DecayModelError("invalid training row {} of {}: {}".format(line_number, path, error)) from error
    if not rows:
        raise DecayModelError("no training examples found in {}".format(path))
    return rows


def train(
    training_examples: Iterable[DecayTrainingExample],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    model_version: str,
) -> LogisticDecayPredictor:
    return LogisticDecayPredictor.fit(
        training_examples,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        model_version=model_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Horizon's logistic state-decay predictor")
    parser.add_argument("training_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="model artifact JSON path")
    parser.add_argument("--validation-jsonl", type=Path)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--model-version", default="logistic-state-decay-v1")
    args = parser.parse_args()

    training_rows = read_examples(args.training_jsonl)
    predictor = train(
        training_rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        model_version=args.model_version,
    )
    predictor.save(args.output)
    report = {
        "model": predictor.to_mapping(),
        "training_examples": len(training_rows),
    }
    if args.validation_jsonl is not None:
        report["validation"] = predictor.evaluate(read_examples(args.validation_jsonl)).as_dict()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
