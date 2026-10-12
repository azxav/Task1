from dataclasses import replace

import pytest

from faq.config import Settings


def test_local_defaults_use_the_ipv4_addresses_exposed_by_compose(monkeypatch):
    from urllib.parse import urlsplit

    for key in ("DATABASE_URL", "QDRANT_URL", "FAQ_API_URL"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.load()
    assert all(
        urlsplit(url).hostname == "127.0.0.1"
        for url in (settings.database_url, settings.qdrant_url, settings.api_url)
    )


def calibrated():
    settings = Settings.load()
    frozen = {
        "profile": settings.profile,
        "model_name": settings.model_name,
        "model_revision": settings.model_revision,
        "device": settings.device,
        "top_k": settings.top_k,
        "min_score": settings.min_score,
        "min_margin": settings.min_margin,
        "registry_digest": "reviewed",
        "corpus_digest": "original",
    }
    return replace(settings, calibration=frozen, allow_uncalibrated=False), {
        "registry_digest": "reviewed",
        "corpus_digest": "original",
    }


@pytest.mark.parametrize(
    "change", [{"device": "cuda"}, {"profile": "raw"}, {"model_revision": "different"}, {"min_score": 0.99}]
)
def test_changed_pipeline_cannot_silently_reuse_calibrated_thresholds(change):
    settings, meta = calibrated()
    with pytest.raises(ValueError, match="calibration"):
        replace(settings, **change).validate_calibration(meta)


def test_changed_source_requires_recalibration_or_explicit_experiment():
    settings, meta = calibrated()
    meta["corpus_digest"] = "changed"
    with pytest.raises(ValueError, match="calibration"):
        settings.validate_calibration(meta)
    assert replace(settings, allow_uncalibrated=True).validate_calibration(meta) == "uncalibrated"
    settings, meta = calibrated()
    assert settings.validate_calibration(meta) == "calibrated"
