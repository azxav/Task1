import argparse
import json
from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from .config import Settings
from .data import load_dataset
from .embeddings import Encoder
from .evaluation.text_profiles import prepare_profile
from .index import VectorIndex
from .search.answer_groups import EquivalenceRegistry
from .storage import Storage


def import_data(source, settings):
    started = perf_counter()
    dataset = load_dataset(source)
    dataset.questions = {
        ident: replace(q, embedding_text=prepare_profile(q.question, settings.profile))
        for ident, q in dataset.questions.items()
    }
    registry = EquivalenceRegistry.load(settings.registry_path, dataset.answers)
    calibration_status = settings.validate_calibration(
        {"registry_digest": registry.digest, "corpus_digest": dataset.canonical_digest}
    )
    encoder = Encoder(settings)
    questions = dataset.indexed
    vectors = encoder.encode([q.embedding_text for q in questions], cache=True)
    index, store = VectorIndex(settings.qdrant_url), Storage(settings.database_url)
    generation = "faq_" + uuid4().hex
    index.create_generation(generation, questions, vectors)
    metadata = {
        "generation": generation,
        "profile": settings.profile,
        "model_name": settings.model_name,
        "model_revision": settings.model_revision,
        "registry_digest": registry.digest,
        "hashes": dataset.hashes,
        "corpus_digest": dataset.canonical_digest,
        "indexed_questions": len(questions),
        "max_tokens": 512,
        "embedding_device": settings.device,
    }
    metadata["calibration_status"] = calibration_status
    store.initialize()
    with store.snapshot_lock(publish=True):
        store.replace(dataset, metadata)
        index.publish(generation)
    return {
        **metadata,
        "warnings": dataset.warnings,
        "elapsed_seconds": perf_counter() - started,
        "model_startup_seconds": encoder.startup_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description="Local semantic FAQ")
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="Validate and publish a complete data snapshot")
    importer.add_argument("source", nargs="?", default="data/source")
    search = commands.add_parser("search", help="Query the published local index")
    search.add_argument("question")
    commands.add_parser("evaluate", help="Calibrate and measure real BGE-M3 retrieval")
    commands.add_parser("bot", help="Run Telegram polling (requires token)")
    args = parser.parse_args()
    settings = Settings.load()
    if args.command == "import":
        result = import_data(args.source, settings)
    elif args.command == "search":
        from .service import SearchService

        result = SearchService(settings).search(args.question)
    elif args.command == "evaluate":
        from .evaluation.runner import run_evaluation

        result = run_evaluation(settings)
    else:
        from .bot import run_bot

        run_bot(settings)
        return
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
