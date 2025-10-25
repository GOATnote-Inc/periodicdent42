"""Autonomous campaign runner for integrating instruments with the Experiment OS."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from configs.data_schema import Experiment, ExperimentStatus, Measurement, Prediction, Protocol, Result
from src.data import UvVisReferenceLibrary, load_reference_library
from src.drivers import MeasurementPayload, SafetyInterlockError, UvVisConfiguration, UvVisSpectrometer
from src.services import db
from src.services.storage import LocalStorageBackend, get_storage

logger = logging.getLogger(__name__)


@dataclass
class CampaignReport:
    """Summary returned after a campaign completes."""

    campaign_id: str
    instrument_id: str
    started_at: datetime
    completed_at: datetime
    experiments_requested: int
    experiments_completed: int
    failures: List[str]
    storage_uris: List[str]


class AutonomousCampaignRunner:
    """Coordinates autonomous experiments against a UV-Vis spectrometer."""

    def __init__(
        self,
        *,
        driver: Optional[UvVisSpectrometer] = None,
        library: Optional[UvVisReferenceLibrary] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.driver = driver or UvVisSpectrometer()
        self.library = library or load_reference_library()
        self.storage_dir = storage_dir

    def run_campaign(
        self,
        *,
        min_experiments: int = 50,
        max_hours: float = 24.0,
        configuration: Optional[UvVisConfiguration] = None,
    ) -> CampaignReport:
        """Execute an autonomous campaign collecting at least ``min_experiments`` samples."""

        campaign_id = f"uvvis-{uuid.uuid4().hex[:8]}"
        started_at = datetime.utcnow()
        deadline = started_at + timedelta(hours=max_hours)
        configuration = configuration or UvVisConfiguration(
            integration_time_ms=120,
            averages=5,
            lamp_warmup_minutes=2,
            facility="Stanford Nano Shared Facilities",
            safety_contact="lab-safety@periodic.ai",
        )

        self.driver.connect()
        self.driver.initialize(configuration)
        self.driver.force_ready()

        failures: List[str] = []
        storage_uris: List[str] = []
        experiments_completed = 0

        samples = self.library.sample_batch(size=min_experiments)
        storage = get_storage()

        for idx, sample in enumerate(samples, start=1):
            if datetime.utcnow() > deadline:
                logger.warning("Campaign deadline reached before completing all experiments")
                break

            experiment_id = f"{campaign_id}-exp-{idx:03d}"
            try:
                payload = self._execute_sample(experiment_id, sample)
                result = self._build_result(experiment_id, payload)
                uri = self._persist_result(experiment_id, result, storage)
                if uri:
                    storage_uris.append(uri)
                experiments_completed += 1
            except SafetyInterlockError as err:
                logger.exception("Safety interlock triggered - skipping experiment")
                failures.append(f"{experiment_id}: {err}")
            except Exception as err:  # pragma: no cover - defensive
                logger.exception("Unexpected failure running experiment")
                failures.append(f"{experiment_id}: {err}")

        completed_at = datetime.utcnow()
        self.driver.disconnect()

        return CampaignReport(
            campaign_id=campaign_id,
            instrument_id=self.driver.instrument_id,
            started_at=started_at,
            completed_at=completed_at,
            experiments_requested=min_experiments,
            experiments_completed=experiments_completed,
            failures=failures,
            storage_uris=storage_uris,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_sample(self, experiment_id: str, sample) -> MeasurementPayload:
        self.driver.open_lid()
        self.driver.insert_cuvette(sample.sample_id)
        self.driver.close_lid()
        payload = self.driver.perform_measurement(
            experiment_id=experiment_id,
            sample_metadata={
                "sample_id": sample.sample_id,
                "composition": sample.composition,
                "labels": sample.labels,
                "metadata": sample.metadata,
            },
        )
        self.driver.open_lid()
        self.driver.eject_cuvette()
        return payload

    def _build_result(self, experiment_id: str, payload: MeasurementPayload) -> Result:
        protocol = Protocol(
            instrument_id=payload.instrument_id,
            parameters={
                "integration_time_ms": payload.metadata["integration_time_ms"],
                "averages": payload.metadata["averages"],
                "wavelength_start_nm": payload.wavelengths_nm[0],
                "wavelength_end_nm": payload.wavelengths_nm[-1],
                "wavelength_step_nm": payload.wavelengths_nm[1] - payload.wavelengths_nm[0],
            },
            duration_estimate_hours=0.02,
            cost_estimate_usd=45.0,
            safety_checks=[
                "cuvette_inserted",
                "lid_closed",
                "lamp_warm",
            ],
        )

        experiment = Experiment(
            sample_id=payload.metadata["sample_id"],
            protocol=protocol,
            created_by="autonomy-service",
            metadata={
                "instrument_temperature_c": payload.metadata["temperature_c"],
                "composition": payload.metadata["composition"],
            },
        )

        measurement = Measurement(
            value=max(payload.absorbance),
            unit="dimensionless",
            uncertainty=0.02,
            instrument_id=payload.instrument_id,
            experiment_id=experiment_id,
            metadata={
                "wavelength_nm": payload.wavelengths_nm,
                "absorbance": payload.absorbance,
            },
        )

        labels = payload.metadata.get("labels", {})
        bandgap = labels.get("bandgap_ev")
        prediction = Prediction(
            mean=bandgap if bandgap is not None else 0.0,
            std=0.05,
            epistemic=0.02,
            aleatoric=0.03,
        )

        result = Result(
            experiment_id=experiment.id,
            measurements=[measurement],
            derived_properties={"bandgap_ev": prediction},
            analysis_version="phase2-uvvis-v1",
            quality_score=0.92,
            success=True,
            provenance_hash=experiment.protocol.compute_hash(),
            metadata={
                "labels": labels,
                "campaign_experiment_id": experiment_id,
            },
        )

        db.log_instrument_run(
            run_id=experiment_id,
            instrument_id=payload.instrument_id,
            sample_id=payload.metadata["sample_id"],
            campaign_id=result.metadata["campaign_experiment_id"],
            status=ExperimentStatus.COMPLETED.value,
            notes=None,
        )

        return result

    def _persist_result(self, experiment_id: str, result: Result, storage) -> Optional[str]:
        if storage is None:
            if self.storage_dir:
                storage = LocalStorageBackend(root=self.storage_dir)
            else:
                return None
        payload: Dict[str, object] = result.model_dump(mode="json")
        return storage.store_experiment_result(
            experiment_id,
            payload,
            metadata={
                "instrument": result.metadata.get("campaign_experiment_id"),
                "analysis_version": result.analysis_version,
                "quality_score": result.quality_score,
            },
        )


def get_campaign_runner() -> AutonomousCampaignRunner:
    """Factory for dependency injection in API and scripts."""

    return AutonomousCampaignRunner()
