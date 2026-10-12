# Task 2 Local Decision Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract complete, source-grounded document metadata and obligations through a locally running Tev1 decision stack, with measured OCR/model comparisons and a polished web interface.

**Architecture:** Create an independent `task2/` Python application alongside Task 1. Extract coordinate-aware PDF/OCR content, retain structured passages, generate candidates, ask Tev1 narrow contextual questions, and assemble verbatim evidence-backed results. Benchmark before selecting the production OCR and Tev1 size; Qwen remains a comparison baseline.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, httpx, pdfplumber, local Ollama decision API, local OCR adapters, plain HTML/CSS/JavaScript frontend, pytest, ruff. Separate OCR environments avoid conflicts with Task 1's Transformers <5 and CPU PyTorch settings.

**Spec:** `docs/superpowers/specs/2026-10-12-task2-design.md`

## Global Constraints

- Documents and extracted text must remain local; bind application to localhost.
- Scanned pages use OCR before inference; Tev1 and Qwen receive text only.
- Include execution, approval, oversight, reporting, and all other explicit obligations.
- Preserve compound obligations such as draft development plus Council submission as one task.
- Preserve original deadline wording; do not calculate ambiguous deadlines or guess executors.
- Missing fields: `not specified`; ambiguous fields: `needs review` with evidence.
- Every populated field and task must refer to source passage/page; inherited fields also cite their origin.
- One PDF at a time; maximum 20 MB and 20 pages; temporary server files deleted after success/failure.
- Tev1 is production engine; compare 0.8B/4B plus Qwen3.5-4B baseline.
- Models run sequentially within the laptop's 6 GB VRAM budget. Pin tested revisions/runtime settings and verify licenses.
- Do not hardcode sample organization names, task counts, or expected predictions in extraction code.

## File Map

- `task2/pyproject.toml`: separate application dependencies, CLI, tests.
- `task2/src/doc_tasks/schema.py`: source, document, task, decision, and result contracts.
- `task2/src/doc_tasks/pdf.py`, `ocr.py`, `structure.py`: PDF validation, OCR adapters, layout-preserving passages.
- `task2/src/doc_tasks/candidates.py`, `decisions.py`, `extract.py`: span candidates, local inference, orchestration/validation.
- `task2/src/doc_tasks/evaluate.py`, `cli.py`: comparisons, metrics, commands.
- `task2/src/doc_tasks/api.py`, `web/`: upload API and frontend.
- `task2/evaluation/{manifest.json,gold/,results/}`: frozen paired splits, expected results, measured reports.
- `task2/tests/`, `task2/README.md`, `task2/.env.example`: verification and runnable instructions.
- OCR adapters run externally with local file paths; store scripts and environment locks in `task2/ocr-runtime/`.
- Modify root `.gitignore` for Task 2 environments, uploads, extracted raw files, model caches, and inspection artifacts.

## Review Focus

1. Sender vs recipient vs signer vs footer preparer must remain distinguishable (Tasks 1, 3).
2. OCR row shifts and missing words must trigger evidence/review checks rather than cross-link adjacent tasks (Tasks 2, 3).
3. Context defaults, recurrence, working-day and event-dependent wording must survive verbatim (Tasks 1, 3).
4. Multiple executors and unnumbered obligations outside tables must not disappear (Tasks 1, 3).
5. Runtime errors, disconnects and concurrent uploads must release temporary files and the single-job lock (Task 6).

---

### Task 1: Source Contracts and Frozen Expected Results

**Files:** `task2/pyproject.toml`, `src/doc_tasks/schema.py`, `evaluation/manifest.json`, `evaluation/gold/*.json`, `tests/test_schema.py`, `tests/test_manifest.py`; root `.gitignore`.

**Interfaces:** Produce Pydantic `SourceSpan(id, page, text, bbox, parent_id)`, `Passage(id, kind, span_ids, clause_ref, table_cells)`, `Document(spans, passages)`, `FieldValue(text, source_ids, status)`, `TaskRecord(id, kinds, instruction, executors, deadline, clause_ref, review_reasons)`, `AnalysisResult(metadata, tasks, warnings, provenance)`. Metadata keys: `sender`, `number`, `date`, `signer`; field status: `supported`, `not_specified`, `needs_review`.

- [ ] Write `test_result_requires_evidence`: a supported field with `source_ids=[]` fails validation; missing field has `text=None` and `status='not_specified'`. Write `test_multiple_executors_and_verbatim_deadline`: two executors survive serialization, and `deadline.text == 'uch ish kuni ichida'`.
- [ ] Run `uv run --project task2 pytest task2/tests/test_schema.py -q`; confirm failure before implementation.
- [ ] Implement contracts, separate package and exact JSON/UI status mapping; no date arithmetic.
- [ ] Inspect all source pages locally and annotate metadata, all obligations, compound grouping, executors, deadlines and evidence. Text/scanned counterparts get separately checked evidence references.
- [ ] Freeze four development families: NAMUNA_1, NAMUNA_2, xat_1, xat_3. Hold out NAMUNA_3 and xat_2 with their scanned counterparts. Write `test_pair_split_is_disjoint`: each pair has one split, `dev_family_count == 4`, `test_family_count == 2`.
- [ ] Run schema/manifest tests; confirm all pass. Commit only Task 2 contracts and annotation artifacts.

### Task 2: Local PDF/OCR Structure and Comparison

**Files:** `src/doc_tasks/pdf.py`, `ocr.py`, `structure.py`; `ocr-runtime/`; `tests/test_pdf.py`, `test_structure.py`.

**Interfaces:** Consume source contracts. Produce `read_pdf(path: Path, ocr: OCRAdapter) -> Document`; `OCRAdapter.recognize(image_path: Path, page: int) -> OCRPage`, where `OCRPage` carries spans, table cells, reading order and diagnostics. Produce `build_passages(pages: list[OCRPage]) -> Document`. Normalize native extraction into the same page structure.

- [ ] Write `test_mixed_pdf_uses_ocr_only_for_image_page`, `test_table_cells_keep_row_and_column`, `test_invalid_pdf_is_rejected`, `test_missing_ocr_cell_is_diagnostic`. Assert OCR called only for image page; row 1 deadline cannot move into row 2; diagnostics survive passage construction.
- [ ] Run the named tests; confirm failure, then implement coordinate-aware native extraction and layout-preserving structures.
- [ ] Implement and pin local adapters for PP-OCRv5 Latin plus table reconstruction, PaddleOCR-VL-1.6 page parsing, and Tesseract Uzbek. Preflight installation, license and local inference for each; explicitly record blockers, never fabricate comparison results.
- [ ] Benchmark development scans against manually verified references: character errors, critical token errors, missing spans, reading order, table associations, runtime/memory. Retain exact raw output and configuration hashes for reproducibility.
- [ ] Select production OCR from development evidence; document the decision. Final-test OCR output is evaluated later without tuning. Verify text PDFs use native extraction and all dependencies remain local at inference.
- [ ] Run PDF/structure tests; commit extraction/OCR adapters and measured development report.

### Task 3: Tev1 Candidate Selection and Obligation Linking

**Files:** `src/doc_tasks/candidates.py`, `decisions.py`, `extract.py`; `tests/test_candidates.py`, `test_decisions.py`, `test_extract.py`.

**Interfaces:** Produce `Candidate(id, span_ids, role_hint)`, `DecisionQuestion(id, type, instructions, options)`, `DecisionAnswer(selected_id, probabilities, raw_confidence)`. `DecisionClient.evaluate(state: dict, questions: list[DecisionQuestion]) -> dict[str, DecisionAnswer]`. Produce `extract(document: Document, client: DecisionClient) -> AnalysisResult`; every candidate maps to source spans.

- [ ] Write `test_compound_draft_submission_is_one_task`: one clause produces one instruction containing both actions. Write `test_footer_preparer_is_not_signer_or_executor`, `test_reporting_outside_table_is_included`, `test_inherited_deadline_has_two_sources`, `test_multiple_executors_selected`, `test_unknown_candidate_is_reviewed`, `test_generated_field_text_is_rejected`. Assert complete source instructions, correct roles, both task/default evidence, two executors, review status, and rejection of nonexistent span IDs.
- [ ] Run named tests; confirm failure before implementations. Test decision plumbing using explicit fake answers; do not count fake-answer tests as model accuracy evidence.
- [ ] Generate broad source candidates from layout/clauses, dates/identifiers and contiguous noun phrases. Keep contextual subject spans and multi-line names; measure whether gold spans are available before inference. Do not rely on regex matching sample names.
- [ ] Ask separate obligation-presence and type questions, metadata-role questions, task-to-executor links, task-to-deadline links, and default/inherited applicability. Preserve table header meanings; inspect paragraphs before/after tables and appendices. Multiple executor candidates receive independent link decisions.
- [ ] Give each decision relevant passage, parent/clause context, document metadata and applicable default clauses; bound Tev1 requests to its tested short-context behavior. Use <=24 options and include `unknown/none`; shortlist and recheck candidates when broader pools exceed the bound. Never truncate a clause silently.
- [ ] Implement local Ollama `/v1/systemone` adapter with timeouts, validated outputs and persisted model/runtime provenance. Confirm supported API/runtime and weight license; bind only to explicit local addresses.
- [ ] Assemble verbatim fields and instruction text, deduplicate repeated evidence, preserve related actions together, and validate source IDs/pages. Source validation proves grounding, not the factual correctness of a link. Flag unresolved relation or OCR evidence issues.
- [ ] Run tests and real development examples; inspect missed obligations and wrong links. Adjust prompts/candidates on development only; commit decision stack and error ledger.

### Task 4: Decision Benchmarks and Production Selection

**Files:** `src/doc_tasks/evaluate.py`, `cli.py`; `evaluation/results/`; `tests/test_evaluate.py`.

**Interfaces:** Produce `evaluate(gold: list[AnalysisResult], predicted: list[AnalysisResult]) -> Metrics`; `run_comparison(manifest: Path, split: str, input_mode: str) -> Path`. Metrics include matched/missed/spurious tasks, grouping errors, metadata correctness, executor/deadline linkage, review count, candidate coverage, timings and memory.

- [ ] Write `test_metrics_count_missing_task_and_wrong_executor`: dropping one expected task increments missed count by one; replacing executor increments executor error count. Write `test_compare_reuses_input_hash_and_split`: different models consume identical serialized documents/candidates for each mode.
- [ ] Run metric tests to confirm failure, then implement matching by evidence/clause plus manual adjudication of ambiguous grouping; report raw numerator/denominator counts.
- [ ] Implement Tev1-0.8B/4B configurations and text-only Qwen3.5-4B comparison adapter. Test model loading sequentially; record model tag/hash, quantization, runtime versions, settings, latency and peak memory. Persist comparison configuration.
- [ ] Compare all three on clean development text first, then OCR-derived text. Inspect critical errors and tune review thresholds on development only; do not label uncalibrated concentration scores as correctness probability.
- [ ] Select Tev1 size by obligation/field correctness first, speed/memory second. Freeze prompt, candidates, OCR and threshold configuration before evaluating held-out pairs; report uncertainty from only two final-test families.
- [ ] If Tev1 quality is insufficient, document actual failures and proposed improvement; do not silently ship Qwen as the engine or claim successful accuracy. Save per-document results for all twelve inputs with split labels.
- [ ] Run evaluator tests; commit configuration, results, and concise comparison report.

### Task 5: Upload/Results Interface in the Accepted Style

**Files:** `src/doc_tasks/web/index.html`, `styles.css`, `app.js`.

**Interfaces:** Frontend consumes `AnalysisResult` JSON and the API contracts in Task 6. Browser retains local PDF `Blob` for preview; result JSON is downloaded locally. One document selection replaces prior result/preview object URL.

- [ ] Capture the current 2brain reference before implementation; use warm off-white, grid, bold black headings, lime accents, borders, monospace labels and readable spacing. Use local assets/fonts; do not introduce remote document or font services.
- [ ] Implement upload → Analyze → processing stages → metadata/tasks, source/page evidence, review flags, PDF preview and JSON download. Copy deadlines verbatim; render untrusted source strings as text, never HTML. Result editing/Excel export remain outside initial scope.
- [ ] Verify in browser with a representative compound task, multiple executors, missing field and review flag. Assert original strings displayed, source/page accessible, and downloaded JSON agrees with displayed result.
- [ ] Check keyboard operation, narrow and wide layouts, long Uzbek names, and readable table wrapping. Save screenshots for visual review; commit frontend.

### Task 6: Local API, Resource Limits and Cleanup

**Files:** `src/doc_tasks/api.py`; `tests/test_api.py`; `README.md`, `.env.example`; extend `cli.py`.

**Interfaces:** `POST /api/analyze` accepts one multipart PDF, returns `AnalysisResult`; `GET /api/status` returns stage (`idle`, `reading`, `ocr`, `extracting`, `validating`) and busy state without document text; `GET /api/health` reports local dependency readiness. HTTP 413 for size/page excess, 422 for corrupt/encrypted input, 409 while busy, 503 for unavailable local model/OCR. CLI `doc-tasks serve` binds `127.0.0.1`.

- [ ] Write `test_20mb_limit`, `test_20page_limit`, `test_encrypted_pdf_rejected`, `test_parallel_upload_returns_409`, `test_inference_failure_cleans_temp_and_unlocks`, `test_disconnect_cleans_after_worker_finishes`. Assert limits, status codes, no leftover temporary files and lock release only after active analysis ends.
- [ ] Run API tests to confirm failures. Implement streamed upload limit, page check, single-job worker/lock, processing stages, explicit errors and finally-based cleanup. Cleanup stale task-specific temp files at startup; never remove arbitrary user directories.
- [ ] Connect UI to real API; verify browser-local preview survives server upload deletion. No persistent server result/PDF storage and no public deployment.
- [ ] Document separate environment setup, exact tested OCR/model downloads, offline startup, local dependency readiness, limits, commands, JSON schema and errors. Supply minimal configuration, not secrets.
- [ ] Run API tests and confirm localhost binding; commit service and startup docs.

### Task 7: Complete Verification and Delivery

**Files:** `task2/README.md`, `task2/evaluation/results/report.md`; screenshot artifacts and result JSON.

**Interfaces:** Uses complete CLI/API, frozen benchmark and browser flow.

- [ ] Run `uv run --project task2 pytest task2/tests -q` and `uv run --project task2 ruff check task2/src task2/tests`; require passing output.
- [ ] Run existing Task 1 tests to catch shared-root regressions; do not alter Task 1 behavior to accommodate Task 2.
- [ ] Verify real browser → API → local OCR → Tev1 → evidence-backed results → JSON download for paragraph, table and appendix examples, including scans. Verify errors and temp cleanup once; inspect UI screenshots.
- [ ] Run frozen final-test evaluation once after development freeze; keep it separate from development results. If later fixes use final-test feedback, label that set as exposed and do not claim it remains independent.
- [ ] Check outbound behavior: inference requests target local services; no document content in external requests or routine logs. Inspect dependencies/assets and verify inference with internet unavailable after downloads.
- [ ] Complete twelve result files and concise report: chosen method, actual scores/counts, timing/memory, OCR vs decision errors, Uzbek/generalization limits, unfinished work if any. Link code, startup docs and measured results; no unsupported completion claim.
- [ ] Commit reviewed deliverables. Do not push, publish or deploy unless subsequently requested.

## Handoff

Plan self-review: all eight accepted points and the Tev1-primary instruction are covered. Main integration dependency is the source-span contract; implement sequentially and benchmark before choosing runtime configurations. Recommend native/inline execution in this chat, with final independent review if available. User must review this written plan and select execution method before implementation.
