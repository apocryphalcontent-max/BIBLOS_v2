"""
BIBLOS v2 - Evaluation Metrics

Specialized metrics for evaluating biblical text extraction and
cross-reference discovery quality.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# BASIC METRICS
# =============================================================================


def compute_precision(
    predicted: List[Any],
    ground_truth: List[Any],
) -> float:
    """
    Compute precision: correct predictions / total predictions.

    Args:
        predicted: List of predicted items
        ground_truth: List of ground truth items

    Returns:
        Precision score (0.0 - 1.0)
    """
    if not predicted:
        return 0.0

    predicted_set = set(map(str, predicted))
    truth_set = set(map(str, ground_truth))

    true_positives = len(predicted_set & truth_set)
    return true_positives / len(predicted_set)


def compute_recall(
    predicted: List[Any],
    ground_truth: List[Any],
) -> float:
    """
    Compute recall: correct predictions / total ground truth.

    Args:
        predicted: List of predicted items
        ground_truth: List of ground truth items

    Returns:
        Recall score (0.0 - 1.0)
    """
    if not ground_truth:
        return 0.0

    predicted_set = set(map(str, predicted))
    truth_set = set(map(str, ground_truth))

    true_positives = len(predicted_set & truth_set)
    return true_positives / len(truth_set)


def compute_f1(
    predicted: List[Any],
    ground_truth: List[Any],
) -> float:
    """
    Compute F1 score: harmonic mean of precision and recall.

    Args:
        predicted: List of predicted items
        ground_truth: List of ground truth items

    Returns:
        F1 score (0.0 - 1.0)
    """
    precision = compute_precision(predicted, ground_truth)
    recall = compute_recall(predicted, ground_truth)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# =============================================================================
# CROSS-REFERENCE METRICS
# =============================================================================


def compute_cross_reference_accuracy(
    predicted_refs: List[Dict[str, Any]],
    ground_truth_refs: List[Dict[str, Any]],
    matching_threshold: float = 0.8,
) -> Dict[str, float]:
    """
    Compute comprehensive cross-reference accuracy metrics.

    Evaluates:
    - Overall cross-reference discovery accuracy
    - Connection type classification accuracy
    - Strength rating accuracy
    - Confidence calibration

    Args:
        predicted_refs: List of predicted cross-references
        ground_truth_refs: List of ground truth cross-references
        matching_threshold: Minimum overlap to consider a match

    Returns:
        Dictionary of metric scores
    """
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "type_accuracy": 0.0,
        "strength_accuracy": 0.0,
        "confidence_calibration": 0.0,
    }

    if not ground_truth_refs:
        return metrics

    # Build lookup for ground truth
    truth_lookup: Dict[str, Dict[str, Any]] = {}
    for ref in ground_truth_refs:
        key = f"{ref.get('source_ref', '')}_{ref.get('target_ref', '')}"
        truth_lookup[key] = ref

    # Match predictions to ground truth
    matched = 0
    type_correct = 0
    strength_correct = 0
    confidence_errors = []

    for pred in predicted_refs:
        pred_key = f"{pred.get('source_ref', '')}_{pred.get('target_ref', '')}"
        # Also check reverse direction
        pred_key_rev = f"{pred.get('target_ref', '')}_{pred.get('source_ref', '')}"

        truth = truth_lookup.get(pred_key) or truth_lookup.get(pred_key_rev)

        if truth:
            matched += 1

            # Check connection type
            if pred.get("connection_type") == truth.get("connection_type"):
                type_correct += 1

            # Check strength
            if pred.get("strength") == truth.get("strength"):
                strength_correct += 1

            # Track confidence calibration
            pred_conf = pred.get("confidence", 0.5)
            # Higher confidence should correlate with truth confidence
            truth_conf = truth.get("confidence", 1.0)
            confidence_errors.append(abs(pred_conf - truth_conf))

    # Calculate metrics
    if predicted_refs:
        metrics["precision"] = matched / len(predicted_refs)
    if ground_truth_refs:
        metrics["recall"] = matched / len(ground_truth_refs)

    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (
            metrics["precision"] + metrics["recall"]
        )

    if matched > 0:
        metrics["type_accuracy"] = type_correct / matched
        metrics["strength_accuracy"] = strength_correct / matched
        metrics["confidence_calibration"] = 1.0 - np.mean(confidence_errors) if confidence_errors else 0.0

    return metrics


def compute_type_classification_metrics(
    predicted_types: List[str],
    ground_truth_types: List[str],
) -> Dict[str, float]:
    """
    Compute metrics for connection type classification.

    Args:
        predicted_types: List of predicted connection types
        ground_truth_types: List of ground truth connection types

    Returns:
        Dictionary with accuracy, per-type precision/recall
    """
    if len(predicted_types) != len(ground_truth_types):
        raise ValueError("Predicted and ground truth must have same length")

    if not predicted_types:
        return {"accuracy": 0.0}

    # Overall accuracy
    correct = sum(
        1 for p, g in zip(predicted_types, ground_truth_types) if p == g
    )
    accuracy = correct / len(predicted_types)

    # Per-type metrics
    all_types = set(predicted_types) | set(ground_truth_types)
    per_type_metrics = {}

    for conn_type in all_types:
        type_predicted = [i for i, p in enumerate(predicted_types) if p == conn_type]
        type_truth = [i for i, g in enumerate(ground_truth_types) if g == conn_type]

        tp = len(set(type_predicted) & set(type_truth))
        fp = len(set(type_predicted) - set(type_truth))
        fn = len(set(type_truth) - set(type_predicted))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_type_metrics[conn_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accuracy": accuracy,
        "per_type": per_type_metrics,
    }


def compute_strength_correlation(
    predicted_strengths: List[str],
    ground_truth_strengths: List[str],
) -> float:
    """
    Compute correlation between predicted and ground truth strength ratings.

    Maps strength levels to ordinal values:
    - strong: 3
    - moderate: 2
    - weak: 1

    Args:
        predicted_strengths: List of predicted strength ratings
        ground_truth_strengths: List of ground truth strength ratings

    Returns:
        Spearman correlation coefficient (-1.0 to 1.0)
    """
    strength_map = {"strong": 3, "moderate": 2, "weak": 1}

    pred_values = [strength_map.get(s.lower(), 2) for s in predicted_strengths]
    truth_values = [strength_map.get(s.lower(), 2) for s in ground_truth_strengths]

    if len(pred_values) < 2:
        return 0.0

    correlation, _ = spearmanr(pred_values, truth_values)
    return correlation if not np.isnan(correlation) else 0.0


# =============================================================================
# EXTRACTION QUALITY METRICS
# =============================================================================


def compute_extraction_completeness(
    extraction: Dict[str, Any],
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute completeness metrics for an extraction.

    Args:
        extraction: Extraction data dictionary
        required_fields: List of required field names
        optional_fields: List of optional field names

    Returns:
        Dictionary with completeness scores
    """
    metrics = {
        "required_completeness": 0.0,
        "optional_completeness": 0.0,
        "overall_completeness": 0.0,
    }

    # Required fields
    if required_fields:
        filled_required = sum(
            1 for f in required_fields
            if f in extraction and extraction[f] is not None
        )
        metrics["required_completeness"] = filled_required / len(required_fields)

    # Optional fields
    if optional_fields:
        filled_optional = sum(
            1 for f in optional_fields
            if f in extraction and extraction[f] is not None
        )
        metrics["optional_completeness"] = filled_optional / len(optional_fields)

    # Overall
    all_fields = required_fields + (optional_fields or [])
    if all_fields:
        filled_all = sum(
            1 for f in all_fields
            if f in extraction and extraction[f] is not None
        )
        metrics["overall_completeness"] = filled_all / len(all_fields)

    return metrics


def compute_confidence_calibration(
    confidences: List[float],
    accuracies: List[float],
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute confidence calibration metrics.

    A well-calibrated model should have confidence scores that match
    actual accuracy (e.g., 80% confident predictions should be correct 80% of the time).

    Args:
        confidences: List of confidence scores
        accuracies: List of binary accuracy indicators (0 or 1)
        num_bins: Number of bins for calibration

    Returns:
        Dictionary with calibration metrics
    """
    if len(confidences) != len(accuracies):
        raise ValueError("Confidences and accuracies must have same length")

    if not confidences:
        return {"ece": 0.0, "mce": 0.0}

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    # Bin predictions
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

    # Calculate per-bin accuracy and confidence
    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error

    for bin_idx in range(num_bins):
        in_bin = bin_indices == bin_idx
        if np.sum(in_bin) > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_size = np.sum(in_bin) / len(confidences)

            calibration_error = abs(bin_accuracy - bin_confidence)
            ece += bin_size * calibration_error
            mce = max(mce, calibration_error)

    return {
        "ece": ece,  # Lower is better
        "mce": mce,  # Lower is better
    }


# =============================================================================
# LATENCY METRICS
# =============================================================================


def compute_latency_metrics(
    latencies_ms: List[float],
) -> Dict[str, float]:
    """
    Compute comprehensive latency metrics.

    Args:
        latencies_ms: List of latency measurements in milliseconds

    Returns:
        Dictionary with latency statistics
    """
    if not latencies_ms:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    latencies = np.array(latencies_ms)

    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p90": float(np.percentile(latencies, 90)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }


# =============================================================================
# THEOLOGICAL QUALITY METRICS
# =============================================================================


def compute_patristic_coverage(
    citations_found: List[Dict[str, Any]],
    expected_fathers: Optional[List[str]] = None,
    expected_schools: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute patristic citation coverage metrics.

    Args:
        citations_found: List of found patristic citations
        expected_fathers: Optional list of expected Church Fathers
        expected_schools: Optional list of expected patristic schools

    Returns:
        Dictionary with coverage metrics
    """
    metrics = {
        "total_citations": len(citations_found),
        "unique_fathers": 0,
        "schools_represented": 0,
        "father_coverage": 0.0,
        "school_coverage": 0.0,
    }

    if not citations_found:
        return metrics

    # Extract unique fathers and schools
    fathers = set(c.get("father", "") for c in citations_found if c.get("father"))
    schools = set(c.get("school", "") for c in citations_found if c.get("school"))

    metrics["unique_fathers"] = len(fathers)
    metrics["schools_represented"] = len(schools)

    # Coverage against expected
    if expected_fathers:
        found_fathers = set(f.lower() for f in fathers)
        expected_set = set(f.lower() for f in expected_fathers)
        metrics["father_coverage"] = len(found_fathers & expected_set) / len(expected_set)

    if expected_schools:
        found_schools = set(s.lower() for s in schools)
        expected_set = set(s.lower() for s in expected_schools)
        metrics["school_coverage"] = len(found_schools & expected_set) / len(expected_set)

    return metrics


def compute_typological_accuracy(
    predicted_types: List[Dict[str, Any]],
    ground_truth_types: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute accuracy metrics for typological connections.

    Evaluates type/antitype pairing accuracy and category classification.

    Args:
        predicted_types: List of predicted typological connections
        ground_truth_types: List of ground truth typological connections

    Returns:
        Dictionary with accuracy metrics
    """
    metrics = {
        "pairing_precision": 0.0,
        "pairing_recall": 0.0,
        "pairing_f1": 0.0,
        "category_accuracy": 0.0,
    }

    if not ground_truth_types:
        return metrics

    # Build lookup for ground truth
    truth_lookup = {}
    for t in ground_truth_types:
        key = f"{t.get('type_ref', '')}_{t.get('antitype_ref', '')}"
        truth_lookup[key] = t

    matched = 0
    category_correct = 0

    for pred in predicted_types:
        key = f"{pred.get('type_ref', '')}_{pred.get('antitype_ref', '')}"
        truth = truth_lookup.get(key)

        if truth:
            matched += 1
            if pred.get("category") == truth.get("category"):
                category_correct += 1

    if predicted_types:
        metrics["pairing_precision"] = matched / len(predicted_types)
    if ground_truth_types:
        metrics["pairing_recall"] = matched / len(ground_truth_types)

    if metrics["pairing_precision"] + metrics["pairing_recall"] > 0:
        metrics["pairing_f1"] = (
            2 * metrics["pairing_precision"] * metrics["pairing_recall"] /
            (metrics["pairing_precision"] + metrics["pairing_recall"])
        )

    if matched > 0:
        metrics["category_accuracy"] = category_correct / matched

    return metrics
