from __future__ import annotations

from collections import Counter
from typing import Any

from .utils import extract_severity, safe_perplexity


def eval_loss_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(metrics)
    for key in ("eval_loss", "test_loss"):
        if key in enriched:
            enriched[f"{key.removesuffix('_loss')}_perplexity"] = safe_perplexity(float(enriched[key]))
    return enriched


def generation_metrics(predictions: list[str], references: list[str]) -> dict[str, Any]:
    pred_scores = [extract_severity(text) for text in predictions]
    ref_scores = [extract_severity(text) for text in references]
    comparable = [(pred, ref) for pred, ref in zip(pred_scores, ref_scores) if pred is not None and ref is not None]
    exact_text = sum(_normalize(pred) == _normalize(ref) for pred, ref in zip(predictions, references))
    metrics: dict[str, Any] = {
        "samples": len(predictions),
        "exact_text_match": exact_text / max(len(predictions), 1),
        "severity_comparable": len(comparable),
    }
    if comparable:
        correct = sum(pred == ref for pred, ref in comparable)
        abs_errors = [abs(pred - ref) for pred, ref in comparable]
        metrics["severity_accuracy"] = correct / len(comparable)
        metrics["severity_mae"] = sum(abs_errors) / len(abs_errors)
        metrics["pred_severity_distribution"] = dict(Counter(pred for pred, _ in comparable))
        metrics["ref_severity_distribution"] = dict(Counter(ref for _, ref in comparable))
    return metrics


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())

