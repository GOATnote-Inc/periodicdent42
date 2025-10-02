"""Unit tests for UV-Vis hardware integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import load_reference_library
from src.drivers import SafetyInterlockError, UvVisConfiguration, UvVisSpectrometer
from src.lab.campaign import AutonomousCampaignRunner
from src.services.storage import LocalStorageBackend


@pytest.fixture(scope="module")
def uvvis_sample():
    library = load_reference_library()
    return library.sample_batch(size=1)[0]


@pytest.fixture()
def driver():
    drv = UvVisSpectrometer()
    drv.connect()
    drv.initialize(UvVisConfiguration(lamp_warmup_minutes=0))
    drv.force_ready()
    yield drv
    drv.disconnect()


def test_uv_vis_driver_enforces_interlocks(driver, uvvis_sample):
    """Measurement should fail when safety interlocks are not satisfied."""

    with pytest.raises(SafetyInterlockError):
        driver.perform_measurement(
            experiment_id="test-exp-001",
            sample_metadata={
                "sample_id": uvvis_sample.sample_id,
                "composition": uvvis_sample.composition,
                "labels": uvvis_sample.labels,
                "metadata": uvvis_sample.metadata,
            },
        )

    driver.insert_cuvette(uvvis_sample.sample_id)

    with pytest.raises(SafetyInterlockError):
        driver.perform_measurement(
            experiment_id="test-exp-001",
            sample_metadata={
                "sample_id": uvvis_sample.sample_id,
                "composition": uvvis_sample.composition,
                "labels": uvvis_sample.labels,
                "metadata": uvvis_sample.metadata,
            },
        )

    driver.close_lid()
    payload = driver.perform_measurement(
        experiment_id="test-exp-001",
        sample_metadata={
            "sample_id": uvvis_sample.sample_id,
            "composition": uvvis_sample.composition,
            "labels": uvvis_sample.labels,
            "metadata": uvvis_sample.metadata,
        },
    )

    assert payload.metadata["sample_id"] == uvvis_sample.sample_id
    assert len(payload.absorbance) == len(payload.wavelengths_nm)
    assert max(payload.absorbance) > 0


def test_campaign_runner_collects_results(tmp_path, monkeypatch):
    """Autonomous campaign should store results for each experiment."""

    library = load_reference_library()
    runner = AutonomousCampaignRunner(library=library)

    def _local_storage():
        return LocalStorageBackend(root=tmp_path)

    monkeypatch.setattr("src.lab.campaign.get_storage", _local_storage)

    report = runner.run_campaign(min_experiments=5, max_hours=1.0)

    assert report.experiments_completed == 5
    assert len(report.storage_uris) == 5
    for uri in report.storage_uris:
        assert Path(uri).exists()
