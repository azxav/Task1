# Uzbek normalization implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. This plan schedules implementation; no product normalizer is implemented yet.

**Goal:** Implement and measure a conservative Uzbek text pipeline against the actual Task 1 corpus without erasing answer-changing distinctions.

**Architecture:** Keep original text, embedding text, and comparison keys separate. A pure shared normalizer handles representation differences; separately versioned retrieval experiments handle casing, Cyrillic, and typos. Corpus lint and human-reviewed answer equivalence inform quality measurement.

**Tech Stack:** Python standard library unicodedata/re, pytest, BGE-M3 through sentence-transformers, Qdrant/PostgreSQL integration from the main Task 1 implementation. Use the project's pinned Python environment; no third-party Uzbek NLP dependency is needed for the core profile.

**Spec:** `docs/superpowers/specs/2026-10-12-uzbek-normalization-design.md`. Read alongside `2026-10-12-task1-design.md` and both research notes.

## Global constraints

- Preserve original_text verbatim; return stored answer text verbatim.
- Default embedding profile: uz_surface_v1, preserving case. comparison_key uses NFC after casefold and is not embedding_text.
- Apply only documented character-context transformations; no global transliteration, apostrophe deletion, stemming, negation stripping, or typo correction.
- Unknown profiles fail explicitly. Configured profile/model metadata must match the published index generation.
- No retrieval-quality or speed claim before actual model/service measurements. Tune only on calibration; freeze before final test.

## Review focus

- Word-final g markers and quoted tokens: retain the marker; do not drop it or globally rewrite quotations. Task 1 fixtures.
- Cyrillic homoglyph inside Latin answer 499: lint and preserve original; do not automatically correct all Cyrillic text. Task 2 fixtures.
- Numeric qualifiers/negation: retain distinctions in preprocessing and human labels; model behavior measured separately. Tasks 1 and 3 fixtures.
- Equivalent answer IDs versus partial overlap: only verified global equivalence changes production margin grouping. Task 4 fixtures.
- Query variants leaking into both splits or expansion increasing false acceptance: family-level split and full-pipeline recalibration. Task 3 checks, Task 5 only if needed.

## File map

- `src/faq/text/normalization.py`: pure profile/result/validation API.
- `src/faq/text/scripts.py`: token/script diagnostics without transliteration.
- `tests/text/test_normalization.py`: canonical and preservation contracts.
- `tests/text/test_corpus_lint.py`: source audit and collision behavior.
- `src/faq/evaluation/text_profiles.py`: profile ablation/cached embedding preparation.
- `src/faq/evaluation/labels.py`: query-family and answer-coverage validation.
- `evaluation/uzbek/queries.jsonl`: curated base families and reviewed variants.
- `evaluation/uzbek/manifest.json`: hashes, split, revisions, frozen profile/thresholds.
- `evaluation/uzbek/answer_equivalence.json`: reviewed global classes and rationales.
- `src/faq/search/answer_groups.py`: global equivalence grouping/tie handling.
- `tests/search/test_answer_groups.py`: equivalence and partial-overlap checks.
- `src/faq/text/candidates.py`: optional candidate adapters, only if Task 3 evidence warrants them.
- `docs/evaluation/uzbek-normalization.md`: actual paired metrics and failure analysis.

Main import/API/index files will be named in the overall application plan. This subplan requires the main retrieval interface to expose `search_prepared(text: str) -> list[ScoredQuestion]`, with question_id, answer_id, and cosine score, and a frozen index generation. It does not invent an alternate production entry point.

### Task 1: Shared surface normalization

**Interfaces:** `normalize(text: str, *, profile: str = "uz_surface_v1") -> NormalizationResult`; frozen result fields embedding_text, comparison_key, profile, applied_rules tuple[str, ...], flags tuple[str, ...]. `TextValidationError(ValueError)` for invalid controls/empty normalized input/unknown profile. Script detection in the next task consumes original text, not destructively normalized text.

- [ ] Write failing tests in `tests/text/test_normalization.py`: four typographic markers and contextual backticks canonicalize to ASCII; `bog‘` and `togʻ` retain final markers; `ID‑karta` equals `ID-karta`; quoted `“ID-karta”`, `5′`, `2–3`, `−5`, domains and acronyms remain intact. Assert embedding_text preserves uppercase while comparison_key folds it.
- [ ] Add parameterized preservation pairs `16 yosh/6 yosh`, `mumkin/mumkin emas`, `bo'ladi/bo'lmaydi`, `she'r/sher`, `sur'at/surat`. Assert unequal embedding_text and keys, with original significant tokens retained. Assert doubled markers are flagged and retained, and prime/acute characters are not interpreted as letter markers.
- [ ] Add BOM, NBSP, CRLF/tabs, empty/whitespace-only, NUL, unsupported-control, internal zero-width/format-character, non-Latin and emoji fixtures. Non-Latin alone is valid. Assert idempotent text/key outputs, unknown-profile errors, and no mutation of input objects. Full original strings remain owned by caller.
- [ ] Run `python -m pytest tests/text/test_normalization.py -q`; confirm missing implementation fails before implementation.
- [ ] Implement exact uz_surface_v1 rule order from the spec. Use Unicode category/name-aware Latin letter checks; avoid regex `\w` for apostrophe context. Do not hard-code the observed corpus as an exception.
- [ ] Rerun the test file; require all contracts pass. Commit normalizer and tests together.

### Task 2: Corpus diagnostics and integration contracts

**Interfaces:** `classify_scripts(text: str) -> ScriptDiagnostics`, with aggregate_script (`latin`, `cyrillic`, `mixed`, `other`) and mixed_tokens tuple[str, ...]; `find_collisions(rows: Sequence[QuestionRow]) -> list[Collision]` grouping comparison_key with original question/answer IDs. Define QuestionRow as a TypedDict with id: int, answer_id: int, question: str, and immutable Collision with comparison_key: str, question_ids: tuple[int, ...], answer_ids: tuple[int, ...], cross_answer: bool. Lint never deletes/merges rows. Diagnostic code lives alongside normalizer; the existing `scripts/audit_task1_data.py` remains an independent source audit, not runtime preprocessing.

- [ ] Write failing fixtures for Latin-only, Cyrillic-only, a Latin token with Cyrillic а, Latin domains inside a Cyrillic sentence, same-answer duplicates, cross-answer duplicates, and partially overlapping but nonidentical text. Assert exact counts/IDs and no original-text mutation.
- [ ] Implement script diagnostics and collision report. Import report distinguishes information/warnings from validation failures and includes Unicode code points and row IDs.
- [ ] Verify all 3,048 actual source questions are unchanged by uz_surface_v1; check 2,171 apostrophe-bearing originals under four typographic variants plus contextual backticks, mutating only eligible letter-marker positions. Include real word-final `mablag'` (questions 2419–2424) and separately preserve quote delimiters in `'hayotiy vaziyat'` (2863–2868). Require canonical equality and zero cross-answer collisions under the selected comparison key. Flag answer 499 and preserve it. Run pure tests without loading BGE-M3.
- [ ] Wire the shared normalizer into main importer and search preparation when those modules exist. Store profile and original/prepared texts; prevent comparison_key from reaching embeddings accidentally.
- [ ] Add an integration fixture with importer/search spies confirming both call the same profile and prepare equal text for equivalent typography; a metadata mismatch keeps readiness false. Verify against a real published generation in the main end-to-end run.
- [ ] Commit diagnostics, contracts and tests.

### Task 3: Controlled real-model quality experiment

**Interfaces:** `prepare_profile(text: str, profile: Literal["raw", "uz_surface_v1", "uz_surface_casefold_v1"]) -> str`; `validate_cases(cases: Sequence[EvaluationCase], index_questions: Sequence[QuestionRow]) -> ValidationReport`. Each EvaluationCase has case_id, family_id, split, query, expected_status, acceptable_answer_ids, slice, coverage_rationale, parent_case_id when mutated. Each profile includes model/tokenizer revision and exact prepared-text hash in its cache key.

- [ ] Curate 72 independent answerable families (8/category), split 36/36 by family, and 36 no-match/clarification families split 18/18. Review complete answers, not source question labels alone. For answers 22/144 inspect query-specific coverage; review 380/494 and 181/279 without silently merging source IDs. Record label rationales in JSONL.
- [ ] Add reviewed typography/whitespace, casing/acronym, missing-apostrophe, typo, colloquial/morphological, script, and numeric/negation slices. Do not assume a changed qualifier retains its parent label. Keep every parent's variants in its split.
- [ ] Write failing validation tests: family split overlap, duplicate IDs, missing/inactive expected answers, normalized exact index overlap in the independent benchmark, and a mutated query with no rationale/parent. Diagnostic holdout overlap must be disclosed separately rather than silently accepted.
- [ ] Implement label validation and three preparation profiles. Casefold experiment applies NFC afterward. Raw profile still obeys input validation and length limits but makes no linguistic changes. No profile is implicitly selected by runtime configuration.
- [ ] Inspect actual pinned tokenizer token IDs for character-equivalent cases. Require equal IDs for identical prepared text; report behavior for casing, negation, numbers, apostrophe omission and unsupported marks without prescribing fake tokenizer invariants.
- [ ] Run P0/P1/P2 on calibration with the real model and fixed index snapshots. P0/P1 index text is identical for this corpus; reuse vectors via prepared-text/model hash. Independently calibrate each pipeline's rejection thresholds; choose the simplest useful profile without new incorrect accepted predictions on calibration. Record configuration, caches, latency and failed cases.
- [ ] Freeze selected profile/thresholds and run the final split once. Save per-case raw/grouped ranks, decision, scores, expected labels, timings, channel provenance and generation. Report strict-ID and semantic-label accuracy/recall@5, accepted precision, coverage, false acceptance/rejection, slice counts, and paired changes. Preserve disappointing results; no retuning on final test.
- [ ] Add the separate supplied-variant holdout diagnostic with its template-similarity limitation. Save actual findings in `docs/evaluation/uzbek-normalization.md`; commit only reproducible cases/configuration and measured results, never models or secrets.

### Task 4: Reviewed equivalence and rejection behavior

**Interfaces:** `group_answer_candidates(candidates: Sequence[ScoredQuestion], registry: EquivalenceRegistry) -> list[AnswerGroup]`. AnswerGroup contains group_id, member_answer_ids, best_answer_id, best_score. Registry contains version, validated disjoint classes, source answer IDs, review rationale. Query-specific acceptable-ID labels are not production registry input.

- [ ] Write failing tests: two verified equivalent IDs do not create a false ambiguity margin; answer 22/144 remain separate absent verified global equivalence; unknown IDs/overlapping classes reject invalid registry; equal scores choose lowest numeric answer ID. Repeated question variants do not multiply competing groups.
- [ ] Create the registry only after reviewing actual answer text. Keep unreviewed IDs singleton groups. If no pair is approved, ship an empty registry and record the limitation rather than asserting equivalence.
- [ ] Implement grouping and wire it into the main decision layer. Record registry version in the generation/evaluation manifest; registry changes require recalibration.
- [ ] Rerun grouping tests and selected calibration experiment with the final registry before freezing thresholds for Task 3 final evaluation. Task 3/4 are coordinated; do not publish a final report using a different registry than runtime.
- [ ] Commit grouping and reviewed registry with their tests.

### Task 5: Evidence-gated query adapters

**Interfaces if activated:** `make_query_candidates(text: str, policy: CandidatePolicy) -> tuple[QueryCandidate, ...]`, with prepared_text, channel, policy_version. Original is always included. No adapter runs by default merely because it exists.

- [ ] Classify Task 3 calibration failures. If dense retrieval plus the conservative profile is sufficient, document that and skip candidate code. Otherwise identify a specific failure slice and an independently testable candidate policy; do not start a full morphology/spelling project.
- [ ] For Cyrillic, write contextual transliteration tests and preservation of Latin domains/numbers; cover е/э/ё/ю/я, ц, ъ/ь, ў/ғ/қ/ҳ and mixed-language ambiguity. Russian-looking/unresolved contexts retain the original query instead of claiming a certain Uzbek correction.
- [ ] For narrowly approved typo/alias rules, require reviewed examples and ambiguous counterexamples (missing apostrophe, h/x, pasport versus ID-karta). Original text and channel provenance survive. No global confusable replacements.
- [ ] Union/deduplicate candidates, group answers using Task 4, recalibrate the whole expanded pipeline on calibration, and explicitly measure new false acceptances and latency. Do not reuse single-query thresholds. Select/adopt only with measured evidence; final testing uses an untouched held-out set and frozen manifest.
- [ ] Commit only the justified adapter, tests and measured report. Record unsupported cases clearly if adapters are deferred.

## Sequence and completion

Implement Tasks 1–2 first. Curate Task 3 labels, complete Task 4 registry/grouping, then run Task 3 calibration and final measurements in that order. Task 5 is optional and must be decided before final testing; if it is introduced after inspecting final failures, create a fresh held-out set and report the earlier failed attempt. Core deterministic checks passing is not evidence that semantic retrieval is accurate.

The main application plan must still supply import/search/index services, Telegram, and README. This plan covers only the requested Uzbek quality foundation. Review design and plan before product implementation; execute inline unless the user selects a different method.
