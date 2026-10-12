import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    qdrant_url: str
    model_name: str
    model_revision: str
    device: str
    profile: str
    min_score: float
    min_margin: float
    top_k: int
    registry_path: str
    token: str = field(repr=False)
    api_url: str
    batch_size: int = 16
    calibration: dict = field(default_factory=dict, repr=False)
    allow_uncalibrated: bool = False

    def validate_calibration(self, metadata):
        expected = {
            "profile": self.profile,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "device": self.device,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "registry_digest": metadata.get("registry_digest"),
            "corpus_digest": metadata.get("corpus_digest"),
        }
        mismatches = [key for key, value in expected.items() if self.calibration.get(key) != value]
        if mismatches and not self.allow_uncalibrated:
            raise ValueError(
                "Frozen calibration mismatch: "
                + ", ".join(mismatches)
                + "; evaluate this pipeline or set FAQ_ALLOW_UNCALIBRATED=1 for an explicit experiment"
            )
        return "uncalibrated" if mismatches else "calibrated"

    @classmethod
    def load(cls):
        load_dotenv(override=False)
        manifest_path = Path(os.getenv("FAQ_MANIFEST", "evaluation/uzbek/manifest.json"))
        frozen = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return cls(
            os.getenv("DATABASE_URL", "postgresql://faq:faq_local@127.0.0.1:55432/faq"),
            os.getenv("QDRANT_URL", "http://127.0.0.1:56333"),
            os.getenv("MODEL_NAME", "BAAI/bge-m3"),
            os.getenv("MODEL_REVISION", MODEL_REVISION),
            os.getenv("EMBEDDING_DEVICE", "cpu"),
            os.getenv("FAQ_PROFILE", frozen.get("profile", "uz_surface_v1")),
            float(os.getenv("FAQ_MIN_SCORE", frozen.get("min_score", 0.75))),
            float(os.getenv("FAQ_MIN_MARGIN", frozen.get("min_margin", 0.03))),
            int(os.getenv("FAQ_TOP_K", frozen.get("top_k", 30))),
            os.getenv("FAQ_REGISTRY", "evaluation/uzbek/answer_equivalence.json"),
            os.getenv("TELEGRAM_BOT_TOKEN", ""),
            os.getenv("FAQ_API_URL", "http://127.0.0.1:8000"),
            int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            frozen,
            os.getenv("FAQ_ALLOW_UNCALIBRATED", "0") == "1",
        )
