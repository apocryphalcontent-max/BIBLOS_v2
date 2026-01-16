"""
BIBLOS v2 - Ensemble Inference

Multi-model ensemble inference combining different approaches.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

import numpy as np


class EnsembleStrategy(Enum):
    """Strategies for combining ensemble predictions."""
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted"
    MAX_CONFIDENCE = "max"
    VOTING = "voting"
    STACKING = "stacking"


@dataclass
class ModelPrediction:
    """Prediction from a single model."""
    model_name: str
    predictions: List[Dict[str, Any]]
    confidence: float
    processing_time: float


@dataclass
class EnsembleConfig:
    """Configuration for ensemble inference."""
    strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE
    min_agreement: float = 0.5  # Minimum model agreement for voting
    confidence_threshold: float = 0.4
    model_weights: Dict[str, float] = field(default_factory=dict)


class EnsembleInference:
    """
    Ensemble inference combining multiple model predictions.

    Supports various combination strategies:
    - Average: Simple average of predictions
    - Weighted Average: Weighted by model performance
    - Max Confidence: Take highest confidence prediction
    - Voting: Majority vote among models
    - Stacking: Use meta-model to combine predictions
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.logger = logging.getLogger("biblos.ml.inference.ensemble")
        self._models: Dict[str, Any] = {}
        self._model_weights: Dict[str, float] = {}
        self._meta_model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize ensemble components."""
        self.logger.info("Initializing ensemble inference")

        # Set default weights
        self._model_weights = {
            "embedding_similarity": 0.3,
            "semantic_classifier": 0.25,
            "gnn_discovery": 0.25,
            "rule_based": 0.2
        }

        # Override with config weights
        self._model_weights.update(self.config.model_weights)

        self._initialized = True
        self.logger.info(f"Ensemble initialized with {len(self._model_weights)} models")

    def register_model(
        self,
        name: str,
        model: Any,
        weight: float = 1.0
    ) -> None:
        """Register a model for ensemble predictions."""
        self._models[name] = model
        self._model_weights[name] = weight
        self.logger.info(f"Registered model: {name} (weight: {weight})")

    async def predict(
        self,
        source_verse: str,
        source_text: str,
        candidates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run ensemble prediction on candidates.

        Args:
            source_verse: Source verse ID
            source_text: Source verse text
            candidates: List of candidate targets
            context: Optional pipeline context

        Returns:
            List of scored candidates with ensemble predictions
        """
        if not self._initialized:
            await self.initialize()

        # Collect predictions from all models
        model_predictions = await self._collect_predictions(
            source_verse,
            source_text,
            candidates,
            context
        )

        # Combine predictions based on strategy
        combined = await self._combine_predictions(
            model_predictions,
            candidates
        )

        return combined

    async def _collect_predictions(
        self,
        source_verse: str,
        source_text: str,
        candidates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> List[ModelPrediction]:
        """Collect predictions from all registered models."""
        predictions = []

        # Run models in parallel
        tasks = []
        model_names = []

        for name, model in self._models.items():
            if hasattr(model, "predict"):
                tasks.append(
                    self._run_model(name, model, source_verse, source_text, candidates, context)
                )
                model_names.append(name)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Model {model_names[i]} failed: {result}")
                elif result:
                    predictions.append(result)

        # Add rule-based predictions if no models available
        if not predictions:
            predictions.append(await self._rule_based_predictions(
                source_verse, source_text, candidates, context
            ))

        return predictions

    async def _run_model(
        self,
        name: str,
        model: Any,
        source_verse: str,
        source_text: str,
        candidates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Optional[ModelPrediction]:
        """Run a single model for prediction."""
        import time

        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(model.predict):
                results = await model.predict(source_verse, source_text, candidates)
            else:
                results = model.predict(source_verse, source_text, candidates)

            # Normalize predictions
            predictions = []
            for i, candidate in enumerate(candidates):
                if i < len(results):
                    pred = results[i]
                    predictions.append({
                        "target": candidate.get("target_verse") or candidate.get("verse_id"),
                        "confidence": float(pred.get("confidence", 0.5)),
                        "type": pred.get("type", "thematic"),
                        "score": float(pred.get("score", pred.get("confidence", 0.5)))
                    })
                else:
                    predictions.append({
                        "target": candidate.get("target_verse") or candidate.get("verse_id"),
                        "confidence": 0.5,
                        "type": "thematic",
                        "score": 0.5
                    })

            avg_confidence = np.mean([p["confidence"] for p in predictions]) if predictions else 0.5

            return ModelPrediction(
                model_name=name,
                predictions=predictions,
                confidence=float(avg_confidence),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            self.logger.error(f"Model {name} error: {e}")
            return None

    async def _rule_based_predictions(
        self,
        source_verse: str,
        source_text: str,
        candidates: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> ModelPrediction:
        """Generate rule-based predictions as fallback."""
        import time

        start_time = time.time()
        predictions = []

        source_words = set(source_text.lower().split())

        for candidate in candidates:
            target_verse = candidate.get("target_verse") or candidate.get("verse_id", "")
            target_text = candidate.get("text", "")

            # Simple word overlap scoring
            target_words = set(target_text.lower().split())
            shared_words = source_words & target_words
            overlap_score = len(shared_words) / max(len(source_words), 1)

            # Book proximity scoring
            source_book = source_verse.split(".")[0] if "." in source_verse else ""
            target_book = target_verse.split(".")[0] if "." in target_verse else ""
            same_book_bonus = 0.1 if source_book == target_book else 0

            # Testament scoring (OT-NT connections are valuable)
            ot_books = {"GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
                       "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
                       "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
                       "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
                       "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"}
            source_ot = source_book in ot_books
            target_ot = target_book in ot_books
            cross_testament_bonus = 0.15 if source_ot != target_ot else 0

            confidence = min(1.0, overlap_score * 0.6 + same_book_bonus + cross_testament_bonus + 0.3)

            # Determine type based on heuristics
            connection_type = "thematic"
            if overlap_score > 0.3:
                connection_type = "verbal"
            elif source_ot and not target_ot:
                connection_type = "typological"

            predictions.append({
                "target": target_verse,
                "confidence": confidence,
                "type": connection_type,
                "score": confidence
            })

        return ModelPrediction(
            model_name="rule_based",
            predictions=predictions,
            confidence=0.6,
            processing_time=time.time() - start_time
        )

    async def _combine_predictions(
        self,
        model_predictions: List[ModelPrediction],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine predictions based on selected strategy."""
        strategy = self.config.strategy

        if strategy == EnsembleStrategy.AVERAGE:
            return self._combine_average(model_predictions, candidates)
        elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return self._combine_weighted(model_predictions, candidates)
        elif strategy == EnsembleStrategy.MAX_CONFIDENCE:
            return self._combine_max(model_predictions, candidates)
        elif strategy == EnsembleStrategy.VOTING:
            return self._combine_voting(model_predictions, candidates)
        elif strategy == EnsembleStrategy.STACKING:
            return await self._combine_stacking(model_predictions, candidates)
        else:
            return self._combine_weighted(model_predictions, candidates)

    def _combine_average(
        self,
        model_predictions: List[ModelPrediction],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simple average of model predictions."""
        results = []

        for i, candidate in enumerate(candidates):
            scores = []
            types = []

            for mp in model_predictions:
                if i < len(mp.predictions):
                    pred = mp.predictions[i]
                    scores.append(pred["confidence"])
                    types.append(pred["type"])

            avg_score = np.mean(scores) if scores else 0.5

            # Most common type
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            most_common = max(type_counts, key=type_counts.get) if type_counts else "thematic"

            results.append({
                "target_verse": candidate.get("target_verse") or candidate.get("verse_id"),
                "confidence": float(avg_score),
                "connection_type": most_common,
                "model_scores": {mp.model_name: mp.predictions[i]["confidence"]
                               for mp in model_predictions if i < len(mp.predictions)},
                "strategy": "average"
            })

        return results

    def _combine_weighted(
        self,
        model_predictions: List[ModelPrediction],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Weighted average based on model weights."""
        results = []

        for i, candidate in enumerate(candidates):
            weighted_sum = 0.0
            weight_total = 0.0
            types = []

            for mp in model_predictions:
                weight = self._model_weights.get(mp.model_name, 1.0)

                if i < len(mp.predictions):
                    pred = mp.predictions[i]
                    weighted_sum += pred["confidence"] * weight
                    weight_total += weight
                    types.append((pred["type"], weight))

            final_score = weighted_sum / weight_total if weight_total > 0 else 0.5

            # Weighted most common type
            type_weights = {}
            for t, w in types:
                type_weights[t] = type_weights.get(t, 0) + w
            most_common = max(type_weights, key=type_weights.get) if type_weights else "thematic"

            results.append({
                "target_verse": candidate.get("target_verse") or candidate.get("verse_id"),
                "confidence": float(final_score),
                "connection_type": most_common,
                "model_scores": {mp.model_name: mp.predictions[i]["confidence"]
                               for mp in model_predictions if i < len(mp.predictions)},
                "strategy": "weighted_average"
            })

        return results

    def _combine_max(
        self,
        model_predictions: List[ModelPrediction],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Take highest confidence prediction."""
        results = []

        for i, candidate in enumerate(candidates):
            best_score = 0.0
            best_type = "thematic"
            best_model = None

            for mp in model_predictions:
                if i < len(mp.predictions):
                    pred = mp.predictions[i]
                    if pred["confidence"] > best_score:
                        best_score = pred["confidence"]
                        best_type = pred["type"]
                        best_model = mp.model_name

            results.append({
                "target_verse": candidate.get("target_verse") or candidate.get("verse_id"),
                "confidence": float(best_score),
                "connection_type": best_type,
                "best_model": best_model,
                "strategy": "max_confidence"
            })

        return results

    def _combine_voting(
        self,
        model_predictions: List[ModelPrediction],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Majority voting among models."""
        results = []

        for i, candidate in enumerate(candidates):
            votes = {"positive": 0, "negative": 0}
            types = []
            scores = []

            for mp in model_predictions:
                if i < len(mp.predictions):
                    pred = mp.predictions[i]
                    if pred["confidence"] >= self.config.confidence_threshold:
                        votes["positive"] += 1
                        types.append(pred["type"])
                    else:
                        votes["negative"] += 1
                    scores.append(pred["confidence"])

            total_votes = votes["positive"] + votes["negative"]
            agreement = votes["positive"] / total_votes if total_votes > 0 else 0

            # Compute final confidence based on agreement
            avg_score = np.mean(scores) if scores else 0.5
            final_confidence = avg_score * agreement

            # Most voted type
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            most_common = max(type_counts, key=type_counts.get) if type_counts else "thematic"

            results.append({
                "target_verse": candidate.get("target_verse") or candidate.get("verse_id"),
                "confidence": float(final_confidence),
                "connection_type": most_common,
                "agreement": float(agreement),
                "votes": votes,
                "strategy": "voting"
            })

        return results

    async def _combine_stacking(
        self,
        model_predictions: List[ModelPrediction],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use meta-model to combine predictions (stacking)."""
        if self._meta_model is None:
            # Fall back to weighted average if no meta-model
            return self._combine_weighted(model_predictions, candidates)

        results = []

        for i, candidate in enumerate(candidates):
            # Build feature vector from model predictions
            features = []

            for mp in model_predictions:
                if i < len(mp.predictions):
                    features.append(mp.predictions[i]["confidence"])
                else:
                    features.append(0.5)

            # Run meta-model
            try:
                features_array = np.array(features).reshape(1, -1)
                meta_prediction = self._meta_model.predict(features_array)
                final_confidence = float(meta_prediction[0])
            except Exception:
                # Fallback
                final_confidence = np.mean(features)

            # Get type from highest-confidence model
            types = []
            for mp in model_predictions:
                if i < len(mp.predictions):
                    types.append((mp.predictions[i]["type"], mp.predictions[i]["confidence"]))
            best_type = max(types, key=lambda x: x[1])[0] if types else "thematic"

            results.append({
                "target_verse": candidate.get("target_verse") or candidate.get("verse_id"),
                "confidence": final_confidence,
                "connection_type": best_type,
                "strategy": "stacking"
            })

        return results

    def set_meta_model(self, model: Any) -> None:
        """Set the meta-model for stacking ensemble."""
        self._meta_model = model
        self.logger.info("Meta-model set for stacking ensemble")

    async def cleanup(self) -> None:
        """Cleanup ensemble resources."""
        self._models.clear()
        self._meta_model = None
        self._initialized = False
        self.logger.info("Ensemble inference cleaned up")
