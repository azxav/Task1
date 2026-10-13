"""Local cross-encoder for uncertain dense matches; pinned model, never truncates input."""

import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter


class PairTooLong(ValueError):
    """Query plus stored text exceeds the reranker's 512-token input."""


class Reranker:
    max_length = 512

    def __init__(self, settings):
        os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
        import torch
        from sentence_transformers import CrossEncoder

        started = perf_counter()
        self.model = CrossEncoder(
            settings.reranker_name,
            revision=settings.reranker_revision,
            device=settings.device,
            cache_folder=os.environ["HF_HOME"],
            max_length=self.max_length,
            activation_fn=torch.nn.Sigmoid(),
        )
        self.settings = settings
        self.startup_seconds = perf_counter() - started

    def score(self, query: str, texts: list[str], *, cache: bool = False) -> list[float]:
        for text in texts:
            if len(self.model.tokenizer(query, text, add_special_tokens=True)["input_ids"]) > self.max_length:
                raise PairTooLong("Query and stored text exceed the reranker's 512-token limit")
        if not texts:
            return []
        key = hashlib.sha256(
            json.dumps(
                [
                    self.settings.reranker_name,
                    self.settings.reranker_revision,
                    self.settings.device,
                    self.max_length,
                    query,
                    texts,
                ],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        path = Path(".cache/rerank") / f"{key}.json"
        if cache and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        scores = [
            float(value)
            for value in self.model.predict(
                [(query, text) for text in texts],
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        ]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("Non-finite rerank score")
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(scores), encoding="utf-8")
            temporary.replace(path)
        return scores
