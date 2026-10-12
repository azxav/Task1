"""One local BGE-M3 instance; cache keyed by exact text and model settings."""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .text.normalization import TextValidationError


class Encoder:
    def __init__(self, settings):
        os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(min(8, os.cpu_count() or 1))
        started = perf_counter()
        self.model = SentenceTransformer(
            settings.model_name,
            revision=settings.model_revision,
            device=settings.device,
            cache_folder=os.environ["HF_HOME"],
            model_kwargs={"use_safetensors": False},
        )
        self.model.max_seq_length = 512
        if settings.device.startswith("cuda"):
            self.model.half()
        self.settings = settings
        self.startup_seconds = perf_counter() - started

    def encode(self, texts, *, cache=False):
        for text in texts:
            if len(self.model.tokenizer.encode(text, add_special_tokens=True)) > self.model.max_seq_length:
                raise TextValidationError("Question exceeds the model's 512-token search limit")
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        key = hashlib.sha256(
            json.dumps(
                [self.settings.model_name, self.settings.model_revision, self.settings.device, 512, texts],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        path = Path(".cache/embeddings") / f"{key}.npy"
        if cache and path.exists():
            return np.load(path, allow_pickle=False)
        vectors = self.model.encode(
            texts,
            batch_size=self.settings.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, vectors)
            temporary.replace(path)
        return vectors
