"""
BIBLOS v2 - Finalization Pipeline Phase

Creates golden record and exports results.
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from pipeline.base import (
    BasePipelinePhase,
    PhaseConfig,
    PhaseResult,
    PhaseStatus
)


class FinalizationPhase(BasePipelinePhase):
    """
    Finalization phase.

    Performs:
    - Golden record assembly
    - Result export preparation
    - Metadata generation
    - Quality certification
    """

    def __init__(self, config: Optional[PhaseConfig] = None):
        if config is None:
            config = PhaseConfig(
                name="finalization",
                agents=[],  # No extraction agents
                parallel=False,
                timeout_seconds=60,
                min_confidence=0.7,
                dependencies=["linguistic", "theological", "intertextual", "validation"]
            )
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize finalization phase."""
        self.logger.info("Initializing finalization phase")

    async def execute(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute finalization."""
        start_time = time.time()

        try:
            # Validate dependencies
            if not await self.validate_dependencies(context):
                return self._create_result(
                    PhaseStatus.SKIPPED,
                    {},
                    start_time,
                    error="Dependencies not satisfied"
                )

            # Build golden record
            golden_record = self._build_golden_record(verse_id, text, context)

            # Generate metadata
            metadata = self._generate_metadata(context)

            # Calculate quality certification
            certification = self._calculate_certification(context)

            # Prepare export data
            export_data = self._prepare_export(golden_record, metadata, certification)

            agent_results = {
                "golden_record": golden_record,
                "metadata": metadata,
                "certification": certification,
                "export_ready": export_data
            }

            metrics = {
                "phase_confidence": certification.get("overall_score", 0.0),
                "total_fields": len(golden_record.get("data", {})),
                "certification_level": certification.get("level", "unknown"),
                "export_size_estimate": len(str(export_data))
            }

            return self._create_result(
                PhaseStatus.COMPLETED,
                agent_results,
                start_time,
                metrics=metrics
            )

        except Exception as e:
            self.logger.error(f"Finalization phase failed: {e}")
            return self._create_result(
                PhaseStatus.FAILED,
                {},
                start_time,
                error=str(e)
            )

    def _build_golden_record(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the golden record from all results."""
        agent_results = context.get("agent_results", {})

        # Get harmonized data if available
        harmonized = context.get("harmonized", {})
        golden_data = harmonized.get("golden_record", {}).get("verse_data", {})

        if not golden_data:
            # Build from raw results
            golden_data = self._merge_agent_data(agent_results)

        return {
            "verse_id": verse_id,
            "text": text,
            "data": golden_data,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0"
        }

    def _merge_agent_data(
        self,
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge data from all agents."""
        merged = {}

        # Linguistic data
        if "grammateus" in agent_results:
            merged["structural"] = agent_results["grammateus"].get("data", {})
        if "morphologos" in agent_results:
            merged["morphological"] = agent_results["morphologos"].get("data", {})
        if "syntaktikos" in agent_results:
            merged["syntactic"] = agent_results["syntaktikos"].get("data", {})
        if "semantikos" in agent_results:
            merged["semantic"] = agent_results["semantikos"].get("data", {})

        # Theological data
        if "patrologos" in agent_results:
            merged["patristic"] = agent_results["patrologos"].get("data", {})
        if "typologos" in agent_results:
            merged["typological"] = agent_results["typologos"].get("data", {})
        if "theologos" in agent_results:
            merged["theological"] = agent_results["theologos"].get("data", {})
        if "liturgikos" in agent_results:
            merged["liturgical"] = agent_results["liturgikos"].get("data", {})
        if "dogmatikos" in agent_results:
            merged["dogmatic"] = agent_results["dogmatikos"].get("data", {})

        # Intertextual data
        if "syndesmos" in agent_results:
            merged["cross_references"] = agent_results["syndesmos"].get("data", {})
        if "allographos" in agent_results:
            merged["quotations"] = agent_results["allographos"].get("data", {})
        if "harmonikos" in agent_results:
            merged["parallels"] = agent_results["harmonikos"].get("data", {})
        if "topos" in agent_results:
            merged["motifs"] = agent_results["topos"].get("data", {})

        return merged

    def _generate_metadata(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate metadata for the golden record."""
        phase_results = context.get("phase_results", {})

        # Collect execution times
        execution_times = {}
        for phase_name, result in phase_results.items():
            if isinstance(result, dict):
                execution_times[phase_name] = result.get("duration", 0)

        return {
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "2.0.0",
            "phases_executed": list(phase_results.keys()),
            "execution_times": execution_times,
            "total_execution_time": sum(execution_times.values()),
            "agent_count": len(context.get("agent_results", {}))
        }

    def _calculate_certification(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate quality certification level."""
        # Get validation results
        elenktikos = context.get("agent_results", {}).get("elenktikos", {})
        kritikos = context.get("agent_results", {}).get("kritikos", {})

        validation_passed = elenktikos.get("data", {}).get("validation_passed", False)
        quality_passed = kritikos.get("data", {}).get("quality_passed", False)

        overall_rating = kritikos.get("data", {}).get("overall_rating", {})
        overall_score = overall_rating.get("score", 0.5)

        # Determine certification level
        if validation_passed and quality_passed and overall_score >= 0.8:
            level = "gold"
        elif validation_passed and overall_score >= 0.6:
            level = "silver"
        elif overall_score >= 0.4:
            level = "bronze"
        else:
            level = "provisional"

        return {
            "level": level,
            "overall_score": overall_score,
            "validation_passed": validation_passed,
            "quality_passed": quality_passed,
            "certified_at": datetime.now(timezone.utc).isoformat()
        }

    def _prepare_export(
        self,
        golden_record: Dict[str, Any],
        metadata: Dict[str, Any],
        certification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for export."""
        return {
            "golden_record": golden_record,
            "metadata": metadata,
            "certification": certification,
            "export_format": "json",
            "schema_version": "2.0.0"
        }

    async def cleanup(self) -> None:
        """Cleanup phase resources."""
        self.logger.info("Finalization phase cleaned up")
