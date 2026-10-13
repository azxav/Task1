import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
# Keys missing from older manifests mean these values.
CALIBRATION_DEFAULTS = {"rerank_enabled": False}


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
    rerank_enabled: bool = False
    reranker_name: str = RERANKER_NAME
    reranker_revision: str = RERANKER_REVISION
    rerank_groups: int = 5
    rerank_floor: float = 0.76
    rerank_min_score: float = 0.5
    rerank_min_margin: float = 0.1

    def calibration_values(self):
        values = {
            "profile": self.profile,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "device": self.device,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "rerank_enabled": self.rerank_enabled,
        }
        if self.rerank_enabled:
            values.update(
                reranker_name=self.reranker_name,
                reranker_revision=self.reranker_revision,
                rerank_groups=self.rerank_groups,
                rerank_floor=self.rerank_floor,
                rerank_min_score=self.rerank_min_score,
                rerank_min_margin=self.rerank_min_margin,
            )
        return values

    def validate_calibration(self, metadata):
        expected = {
            **self.calibration_values(),
            "registry_digest": metadata.get("registry_digest"),
            "corpus_digest": metadata.get("corpus_digest"),
        }
        mismatches = [
            key
            for key, value in expected.items()
            if self.calibration.get(key, CALIBRATION_DEFAULTS.get(key)) != value
        ]
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
        min_score = float(os.getenv("FAQ_MIN_SCORE", frozen.get("min_score", 0.75)))
        rerank_switch = os.getenv("FAQ_RERANK_ENABLED")
        return cls(
            os.getenv("DATABASE_URL", "postgresql://faq:faq_local@127.0.0.1:55432/faq"),
            os.getenv("QDRANT_URL", "http://127.0.0.1:56333"),
            os.getenv("MODEL_NAME", "BAAI/bge-m3"),
            os.getenv("MODEL_REVISION", MODEL_REVISION),
            os.getenv("EMBEDDING_DEVICE", "cpu"),
            os.getenv("FAQ_PROFILE", frozen.get("profile", "uz_surface_v1")),
            min_score,
            float(os.getenv("FAQ_MIN_MARGIN", frozen.get("min_margin", 0.03))),
            int(os.getenv("FAQ_TOP_K", frozen.get("top_k", 30))),
            os.getenv("FAQ_REGISTRY", "evaluation/uzbek/answer_equivalence.json"),
            os.getenv("TELEGRAM_BOT_TOKEN", ""),
            os.getenv("FAQ_API_URL", "http://127.0.0.1:8000"),
            int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            frozen,
            os.getenv("FAQ_ALLOW_UNCALIBRATED", "0") == "1",
            rerank_enabled=(
                frozen.get("rerank_enabled", False) if rerank_switch is None else rerank_switch == "1"
            ),
            reranker_name=os.getenv("RERANKER_NAME", RERANKER_NAME),
            reranker_revision=os.getenv("RERANKER_REVISION", RERANKER_REVISION),
            rerank_groups=int(os.getenv("FAQ_RERANK_GROUPS", frozen.get("rerank_groups", 5))),
            rerank_floor=float(os.getenv("FAQ_RERANK_FLOOR", frozen.get("rerank_floor", min_score))),
            rerank_min_score=float(os.getenv("FAQ_RERANK_MIN_SCORE", frozen.get("rerank_min_score", 0.5))),
            rerank_min_margin=float(os.getenv("FAQ_RERANK_MIN_MARGIN", frozen.get("rerank_min_margin", 0.1))),
        )
