import logging
import threading
from dataclasses import asdict
from time import perf_counter
from uuid import uuid4

from .embeddings import Encoder
from .evaluation.text_profiles import prepare_profile
from .index import VectorIndex
from .search.answer_groups import EquivalenceRegistry, decide, group_answer_candidates
from .storage import Storage
from .text.normalization import TextValidationError

logger = logging.getLogger(__name__)
NO_MATCH = "Savolingizga ishonchli mos javob topilmadi. Iltimos, savolni aniqroq yozing."


class ServiceUnavailable(RuntimeError):
    """Dependency failure or an unpublished/inconsistent snapshot."""


class SearchService:
    def __init__(self, settings, *, storage=None, index=None, encoder=None, registry=None):
        self.settings = settings
        self.storage = storage if storage is not None else Storage(settings.database_url)
        self.index = index if index is not None else VectorIndex(settings.qdrant_url)
        self.registry = registry if registry is not None else EquivalenceRegistry.load(settings.registry_path)
        self.encoder = encoder if encoder is not None else Encoder(settings)
        self.model_lock = threading.Lock()

    def assert_ready(self):
        try:
            meta = self.storage.metadata()
            expected = {
                "profile": self.settings.profile,
                "model_name": self.settings.model_name,
                "model_revision": self.settings.model_revision,
                "registry_digest": self.registry.digest,
            }
            if any(meta.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Prepared index/configuration mismatch")
            if meta.get("embedding_device") != self.settings.device or meta.get("max_tokens") != 512:
                raise RuntimeError("Prepared index/device configuration mismatch")
            self.settings.validate_calibration(meta)
            if self.index.generation() != meta["generation"]:
                raise RuntimeError("Database/index generation mismatch")
            return meta
        except Exception as exc:
            raise ServiceUnavailable("Search service is not ready") from exc

    def search_prepared(self, text: str):
        with self.storage.snapshot_lock():
            meta = self.assert_ready()
            with self.model_lock:
                vector = self.encoder.encode([text])[0]
            return self.index.search(
                vector, expected_generation=meta["generation"], limit=self.settings.top_k
            )

    def search(self, question: str):
        if len(question) > 2000:
            raise TextValidationError("Question exceeds 2,000 characters")
        text = prepare_profile(question, self.settings.profile)
        started, request_id = perf_counter(), str(uuid4())
        try:
            with self.storage.snapshot_lock():
                meta = self.assert_ready()
                with self.model_lock:
                    vector = self.encoder.encode([text])[0]
                candidates = self.index.search(
                    vector, expected_generation=meta["generation"], limit=self.settings.top_k
                )
                decision = decide(
                    group_answer_candidates(candidates, self.registry),
                    min_score=self.settings.min_score,
                    min_margin=self.settings.min_margin,
                )
                result = {
                    **asdict(decision),
                    "answer": None,
                    "matched_question": None,
                    "message": NO_MATCH if decision.status == "no_match" else None,
                    "request_id": request_id,
                }
                if decision.status == "matched":
                    result["answer"] = self.storage.answer(decision.answer_id).answer
                    result["matched_question"] = self.storage.question(decision.question_id)
            logger.info(
                "search request_id=%s status=%s reason=%s elapsed_ms=%.1f",
                request_id,
                decision.status,
                decision.reason,
                1000 * (perf_counter() - started),
            )
            return result
        except (TextValidationError, ServiceUnavailable):
            raise
        except Exception as exc:
            logger.error("search dependency error request_id=%s kind=%s", request_id, type(exc).__name__)
            raise ServiceUnavailable("Search service unavailable") from exc
