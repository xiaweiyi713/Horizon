"""Trainable state-decay prediction and adaptive context-budget control.

The model is deliberately small and dependency-free: it is an offline-trained
logistic classifier over the same observable signals consumed by Horizon's
heuristic policy. This lets experiments compare a learned controller without
moving durable state mutation out of Rust or requiring a heavyweight ML stack.

Training labels are supplied by an experiment adapter after a boundary has an
observable outcome. The model does not claim online learning or causal credit
assignment; hold out runs/models when evaluating it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


Json = Any
STATE_DECAY_FEATURE_SCHEMA = "state_decay_v1"
_FEATURE_NAMES = (
    "anchor_distance",
    "context_pressure",
    "subgoal_switches",
    "recent_failures",
    "recovered_session",
)


class DecayModelError(ValueError):
    """Raised for invalid training data or an incompatible saved model."""


@dataclass(frozen=True)
class DecayFeatures:
    """Normalized, policy-independent inputs to a state-decay predictor."""

    steps_since_anchor: int = 0
    context_pressure: float = 0.0
    subgoal_switches: int = 0
    recent_failures: int = 0
    recovered_session: bool = False

    def __post_init__(self) -> None:
        if self.steps_since_anchor < 0:
            raise DecayModelError("steps_since_anchor cannot be negative")
        if self.subgoal_switches < 0:
            raise DecayModelError("subgoal_switches cannot be negative")
        if self.recent_failures < 0:
            raise DecayModelError("recent_failures cannot be negative")
        _unit_interval(self.context_pressure, "context_pressure")

    @classmethod
    def from_signals(cls, signals: Mapping[str, Json]) -> "DecayFeatures":
        if not isinstance(signals, Mapping):
            raise DecayModelError("state-decay signals must be a mapping")
        return cls(
            steps_since_anchor=_nonnegative_int(signals.get("steps_since_anchor", 0), "steps_since_anchor"),
            context_pressure=_unit_interval(signals.get("context_pressure", 0.0), "context_pressure"),
            subgoal_switches=_nonnegative_int(signals.get("subgoal_switches", 0), "subgoal_switches"),
            recent_failures=_nonnegative_int(signals.get("recent_failures", 0), "recent_failures"),
            recovered_session=bool(signals.get("recovered_session", False)),
        )

    def vector(self) -> tuple[float, ...]:
        """Return stable feature scaling shared by training and inference."""
        return (
            min(1.0, self.steps_since_anchor / 12.0),
            self.context_pressure,
            min(1.0, self.subgoal_switches / 3.0),
            min(1.0, self.recent_failures / 2.0),
            float(self.recovered_session),
        )

    def as_signals(self) -> dict[str, Json]:
        return {
            "steps_since_anchor": self.steps_since_anchor,
            "context_pressure": self.context_pressure,
            "subgoal_switches": self.subgoal_switches,
            "recent_failures": self.recent_failures,
            "recovered_session": self.recovered_session,
        }


@dataclass(frozen=True)
class DecayTrainingExample:
    """One labeled policy boundary for offline state-decay training."""

    features: DecayFeatures
    decayed: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Json]) -> "DecayTrainingExample":
        if not isinstance(value, Mapping):
            raise DecayModelError("state-decay training rows must be JSON objects")
        if "signals" not in value:
            raise DecayModelError("state-decay training row is missing `signals`")
        if "decayed" not in value:
            raise DecayModelError("state-decay training row is missing boolean `decayed`")
        label = value["decayed"]
        if label not in (True, False, 0, 1):
            raise DecayModelError("state-decay training label `decayed` must be boolean")
        return cls(features=DecayFeatures.from_signals(value["signals"]), decayed=bool(label))


@dataclass(frozen=True)
class ClassificationMetrics:
    """Small, transparent held-out metrics for a binary decay predictor."""

    examples: int
    accuracy: float
    precision: Optional[float]
    recall: Optional[float]
    brier_score: float

    def as_dict(self) -> dict[str, Json]:
        return asdict(self)


@dataclass(frozen=True)
class LogisticDecayPredictor:
    """A serializable, regularized logistic state-decay classifier.

    ``weights`` contain an intercept followed by the feature order published in
    :data:`_FEATURE_NAMES`. The trainer uses deterministic full-batch gradient
    descent so identical labeled data and hyperparameters produce the same
    artifact, which is useful for benchmark reproducibility.
    """

    weights: tuple[float, ...]
    model_version: str = "logistic-state-decay-v1"
    feature_schema: str = STATE_DECAY_FEATURE_SCHEMA
    trained_examples: int = 0
    training_log_loss: Optional[float] = None

    def __post_init__(self) -> None:
        if len(self.weights) != len(_FEATURE_NAMES) + 1:
            raise DecayModelError(
                "state-decay model must contain one intercept plus {} feature weights".format(
                    len(_FEATURE_NAMES)
                )
            )
        if self.feature_schema != STATE_DECAY_FEATURE_SCHEMA:
            raise DecayModelError(
                "unsupported state-decay feature schema `{}`".format(self.feature_schema)
            )
        if not self.model_version.strip():
            raise DecayModelError("state-decay model_version cannot be empty")
        if any(not math.isfinite(weight) for weight in self.weights):
            raise DecayModelError("state-decay model weights must be finite")
        if self.training_log_loss is not None and (
            not math.isfinite(self.training_log_loss) or self.training_log_loss < 0
        ):
            raise DecayModelError("state-decay training_log_loss must be finite and non-negative")

    @classmethod
    def fit(
        cls,
        examples: Iterable[DecayTrainingExample],
        *,
        epochs: int = 400,
        learning_rate: float = 0.35,
        l2: float = 0.01,
        model_version: str = "logistic-state-decay-v1",
    ) -> "LogisticDecayPredictor":
        rows = tuple(examples)
        if not rows:
            raise DecayModelError("cannot train a state-decay predictor with no examples")
        if epochs <= 0 or learning_rate <= 0 or l2 < 0:
            raise DecayModelError("epochs/learning_rate must be positive and l2 cannot be negative")

        weights = [0.0] * (len(_FEATURE_NAMES) + 1)
        for _ in range(epochs):
            gradient = [0.0] * len(weights)
            for row in rows:
                vector = (1.0,) + row.features.vector()
                probability = _sigmoid(sum(weight * value for weight, value in zip(weights, vector)))
                error = probability - float(row.decayed)
                for index, value in enumerate(vector):
                    gradient[index] += error * value
            for index in range(1, len(weights)):
                gradient[index] += l2 * weights[index] * len(rows)
            for index in range(len(weights)):
                weights[index] -= learning_rate * gradient[index] / len(rows)

        predictor = cls(
            weights=tuple(weights),
            model_version=model_version,
            trained_examples=len(rows),
            training_log_loss=_log_loss(rows, tuple(weights)),
        )
        return predictor

    def predict_proba(self, features: DecayFeatures) -> float:
        vector = (1.0,) + features.vector()
        return _sigmoid(sum(weight * value for weight, value in zip(self.weights, vector)))

    def evaluate(
        self, examples: Iterable[DecayTrainingExample], *, threshold: float = 0.5
    ) -> ClassificationMetrics:
        rows = tuple(examples)
        if not rows:
            raise DecayModelError("cannot evaluate a state-decay predictor with no examples")
        threshold = _unit_interval(threshold, "threshold")
        true_positive = false_positive = false_negative = correct = 0
        brier = 0.0
        for row in rows:
            probability = self.predict_proba(row.features)
            predicted = probability >= threshold
            correct += predicted == row.decayed
            true_positive += int(predicted and row.decayed)
            false_positive += int(predicted and not row.decayed)
            false_negative += int(not predicted and row.decayed)
            brier += (probability - float(row.decayed)) ** 2
        return ClassificationMetrics(
            examples=len(rows),
            accuracy=correct / len(rows),
            precision=(
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else None
            ),
            recall=(
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else None
            ),
            brier_score=brier / len(rows),
        )

    def to_mapping(self) -> dict[str, Json]:
        return {
            "kind": "logistic_state_decay",
            "feature_schema": self.feature_schema,
            "model_version": self.model_version,
            "weights": list(self.weights),
            "trained_examples": self.trained_examples,
            "training_log_loss": self.training_log_loss,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Json]) -> "LogisticDecayPredictor":
        if not isinstance(value, Mapping) or value.get("kind") != "logistic_state_decay":
            raise DecayModelError("not a Horizon logistic_state_decay model")
        weights = value.get("weights")
        if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
            raise DecayModelError("state-decay model weights must be an array")
        try:
            numeric_weights = tuple(float(weight) for weight in weights)
            training_log_loss = value.get("training_log_loss")
            model_version = value.get("model_version", "")
            feature_schema = value.get("feature_schema", "")
            if not isinstance(model_version, str) or not isinstance(feature_schema, str):
                raise DecayModelError("state-decay model version and feature schema must be strings")
            return cls(
                weights=numeric_weights,
                model_version=model_version,
                feature_schema=feature_schema,
                trained_examples=_nonnegative_int(value.get("trained_examples", 0), "trained_examples"),
                training_log_loss=(
                    float(training_log_loss) if training_log_loss is not None else None
                ),
            )
        except (TypeError, ValueError) as error:
            raise DecayModelError("invalid state-decay model artifact: {}".format(error)) from error

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LogisticDecayPredictor":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DecayModelError("could not load state-decay model {}: {}".format(path, error)) from error
        return cls.from_mapping(value)


@dataclass(frozen=True)
class AdaptiveInterventionDecision:
    """Budget-aware policy output ready for Rust's durable command boundary."""

    features: DecayFeatures
    risk_score: float
    threshold: float
    inject_anchor: bool
    reason: str
    estimated_compact_tokens: int
    estimated_anchor_tokens: int
    context_window_tokens: int
    run_budget_pressure: float

    def as_runtime_assessment(
        self, *, policy_id: str, policy_version: Optional[str]
    ) -> dict[str, Json]:
        risk_score_milli = _score_milli(self.risk_score, "risk_score")
        threshold_milli = _score_milli(self.threshold, "threshold")
        if self.inject_anchor != (risk_score_milli >= threshold_milli):
            raise DecayModelError(
                "intervention decision does not agree with its serialized risk/threshold values"
            )
        return {
            "policy_id": policy_id,
            "policy_version": policy_version,
            "risk_score_milli": risk_score_milli,
            "threshold_milli": threshold_milli,
            "signals": self.features.as_signals(),
            "action": "inject_anchor" if self.inject_anchor else "continue",
            "reason": self.reason,
            "metadata": {
                "controller": "adaptive_context_budget_v1",
                "estimated_compact_tokens": self.estimated_compact_tokens,
                "estimated_anchor_tokens": self.estimated_anchor_tokens,
                "context_window_tokens": self.context_window_tokens,
                "run_budget_pressure_milli": int(round(self.run_budget_pressure * 1000)),
            },
        }


@dataclass(frozen=True)
class AdaptiveContextBudgetController:
    """Choose anchor injection using state-decay risk and token opportunity cost.

    Immediate prompt pressure lowers the intervention threshold because compact
    context is becoming less reliable. Near-exhausted *run* budget and a large
    anchor premium raise it, unless the learned risk crosses the hard safety
    threshold. This makes the trade-off explicit and auditable instead of
    silently injecting anchors at a fixed cadence.
    """

    context_window_tokens: int = 8_192
    base_threshold: float = 0.55
    min_threshold: float = 0.25
    max_threshold: float = 0.90
    hard_risk_threshold: float = 0.85
    context_urgency_weight: float = 0.20
    run_budget_weight: float = 0.15
    anchor_cost_weight: float = 0.25
    chars_per_token: float = 4.0

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0 or self.chars_per_token <= 0:
            raise DecayModelError("context_window_tokens and chars_per_token must be positive")
        for name, value in (
            ("base_threshold", self.base_threshold),
            ("min_threshold", self.min_threshold),
            ("max_threshold", self.max_threshold),
            ("hard_risk_threshold", self.hard_risk_threshold),
        ):
            _unit_interval(value, name)
        for name, value in (
            ("context_urgency_weight", self.context_urgency_weight),
            ("run_budget_weight", self.run_budget_weight),
            ("anchor_cost_weight", self.anchor_cost_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise DecayModelError("{} must be finite and non-negative".format(name))
        if self.min_threshold > self.max_threshold:
            raise DecayModelError("min_threshold cannot exceed max_threshold")

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(math.ceil(len(text) / self.chars_per_token)))

    def decide(
        self,
        *,
        features: DecayFeatures,
        risk_score: float,
        compact_context: str,
        anchor_context: str,
        run_budget_pressure: float,
    ) -> AdaptiveInterventionDecision:
        risk_score = _unit_interval(risk_score, "risk_score")
        run_budget_pressure = _unit_interval(run_budget_pressure, "run_budget_pressure")
        compact_tokens = self.estimate_tokens(compact_context)
        anchor_tokens = self.estimate_tokens(anchor_context)
        immediate_context_pressure = min(1.0, compact_tokens / self.context_window_tokens)
        state_pressure = max(features.context_pressure, immediate_context_pressure)
        anchor_premium = max(0, anchor_tokens - compact_tokens) / self.context_window_tokens
        threshold = self.base_threshold
        threshold -= self.context_urgency_weight * state_pressure
        threshold += self.run_budget_weight * run_budget_pressure
        threshold += self.anchor_cost_weight * min(1.0, anchor_premium)
        threshold = min(self.max_threshold, max(self.min_threshold, threshold))
        hard_risk = risk_score >= self.hard_risk_threshold
        # The persisted threshold must remain consistent with an emergency
        # injection: Rust rejects an anchor action below its declared threshold.
        if hard_risk:
            threshold = min(threshold, self.hard_risk_threshold)
        # Rust persists scores as integer milli-units and validates the action
        # against those values. Decide in the same quantized space so a float
        # comparison near a rounding boundary cannot produce an invalid event.
        inject_anchor = _score_milli(risk_score, "risk_score") >= _score_milli(
            threshold, "threshold"
        )
        if hard_risk:
            reason = "learned state-decay risk crossed the hard safety threshold"
        elif inject_anchor:
            reason = "learned state-decay risk crossed the adaptive context-budget threshold"
        else:
            reason = "learned state-decay risk stayed below the adaptive context-budget threshold"
        return AdaptiveInterventionDecision(
            features=features,
            risk_score=risk_score,
            threshold=threshold,
            inject_anchor=inject_anchor,
            reason=reason,
            estimated_compact_tokens=compact_tokens,
            estimated_anchor_tokens=anchor_tokens,
            context_window_tokens=self.context_window_tokens,
            run_budget_pressure=run_budget_pressure,
        )


@dataclass(frozen=True)
class LearnedInterventionPolicy:
    """Connect an offline-trained predictor to the adaptive budget controller."""

    predictor: LogisticDecayPredictor
    controller: AdaptiveContextBudgetController = field(default_factory=AdaptiveContextBudgetController)
    policy_id: str = "logistic_state_decay"

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise DecayModelError("learned intervention policy_id cannot be empty")

    def decide(
        self,
        signals: Mapping[str, Json],
        *,
        compact_context: str,
        anchor_context: str,
        run_budget_pressure: float,
    ) -> AdaptiveInterventionDecision:
        features = DecayFeatures.from_signals(signals)
        return self.controller.decide(
            features=features,
            risk_score=self.predictor.predict_proba(features),
            compact_context=compact_context,
            anchor_context=anchor_context,
            run_budget_pressure=run_budget_pressure,
        )


def _nonnegative_int(value: Json, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise DecayModelError("{} must be an integer".format(field_name)) from error
    if result < 0:
        raise DecayModelError("{} cannot be negative".format(field_name))
    return result


def _unit_interval(value: Json, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DecayModelError("{} must be numeric".format(field_name)) from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise DecayModelError("{} must be finite and between 0 and 1".format(field_name))
    return result


def _score_milli(value: Json, field_name: str) -> int:
    return int(round(_unit_interval(value, field_name) * 1_000))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _log_loss(rows: Sequence[DecayTrainingExample], weights: tuple[float, ...]) -> float:
    total = 0.0
    epsilon = 1e-12
    for row in rows:
        vector = (1.0,) + row.features.vector()
        probability = _sigmoid(sum(weight * value for weight, value in zip(weights, vector)))
        probability = min(1.0 - epsilon, max(epsilon, probability))
        label = float(row.decayed)
        total -= label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability)
    return total / len(rows)
