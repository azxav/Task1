# Task 2: local PDF obligation extraction

## Purpose and scope

Build the second job-application task, with decisions and flows understandable to the candidate. Extract sender organization, document number, document date, signer, and every explicit obligation with its responsible party and deadline. Cover paragraphs, numbered clauses, tables, and appendices in text PDFs and scanned PDFs. Include execution, approval, oversight, reporting, and other explicit obligations. Documents and extracted text must remain local.

Deliver a runnable repository, startup README, results for the twelve supplied PDFs, and a short report describing the method, measured accuracy, errors, and limitations. The brief specifies a two-day deadline; no performance or accuracy numbers are promised before measurement. Task 1 remains a separate application flow.

## Accepted extraction rules

- Preserve a compound instruction as one task when its actions form one obligation, such as developing a draft and submitting it to the Council. Split clearly separate obligations with distinct deliverables, executors, or deadlines. Preserve clause or table-row references.
- Preserve task and deadline wording from the source. Do not calculate unclear or ambiguous relative deadlines, working days, recurring schedules, or event-dependent dates. Date calculation is outside the initial scope; preserve document dates as written too.
- Preserve named responsible parties, including multiple units. Missing or unclear executor: `not specified`. Do not substitute the document signer or footer preparer for the task executor.
- Missing fields: `not specified`. Ambiguous fields: `needs review`, with available source evidence. Do not guess.
- Every task and populated metadata field carries page and source passage references. An inherited field also references the clause supplying it.

## Architecture

Use a separate Python/FastAPI document-analysis module and a small browser frontend. No new database is required for the initial single-document workflow. Local OCR and inference services use explicit local addresses and downloaded model files. Serve frontend assets locally.

Text PDF pages use direct extraction with coordinates. Scanned pages use local OCR before any decision model. Inspect pages individually to support mixed PDFs. Preserve paragraphs, clauses, table cell/row associations, reading order, and appendix links. No scanned images are fed into Qwen or Tev1.

Generate candidate source passages for organizations, people, document identifiers, and deadlines. Give decision models an obligation passage, relevant document metadata, surrounding context, and candidate spans. Include a `none/unknown` option. Support multiple executors rather than forcing one selection. Retain the complete source instruction as task text.

Tev1 chooses/classifies source candidates; ordinary Qwen uses text-only structured output referencing source candidates. Code validates references, copies source wording, and assembles the result. Candidate omissions and relation errors are measured separately. Model confidence is not presented as a probability of correctness; review thresholds require development-set evidence.

## Model comparison

Decision candidates: Tev1-0.8B, Tev1-4B, Qwen3.5-4B. Tev1 uses Ollama's local decision API; Qwen uses a local text inference adapter. Verify runtime versions, model weight licenses, memory use, and actual local operation before adopting a release. Pin the tested model revisions and settings. Tev1 contexts stay short and include relevant context explicitly.

OCR candidates: PP-OCRv5 Latin recognition with layout/table reconstruction, PaddleOCR-VL-1.6 document parsing, and Tesseract Uzbek as a CPU baseline. These are candidates, not already demonstrated winners. Benchmark all on the six supplied scans. Text counterparts provide reference material but must be manually checked against scan images.

Run OCR and decision models sequentially on the laptop's 6 GB GPU; do not assume both can remain loaded. Select by task/field accuracy first, then latency, memory, and setup reliability.

## Evaluation

Manually annotate expected metadata and obligations for all twelve PDFs. Freeze development and final-test groups before prompt or threshold tuning; keep each text/scanned pair together. With six document families, report sample counts and limitations explicitly.

First compare decision models using manually checked clean text and the same candidate-generation/context process. Then evaluate OCR and complete extraction. Measure missed/spurious obligations, compound-task grouping, metadata correctness, executor/deadline associations, exact source wording, review outcomes, latency, and peak memory. OCR checks include character errors, names/numbers/dates/apostrophes, missing text, reading order, and table associations. Report final-test results separately from tuning results. No invented success threshold.

## Interface

PDF upload, `Analyze` button, processing state, metadata, task table, PDF preview, page/source evidence, review flags, and JSON download. Preserve original task, executor, and deadline wording. Editing and Excel export are optional future scope.

Visual reference: https://2brainai.tech/ as inspected on 2026-10-12. Use warm off-white, subtle grid, bold black headings, lime accents, thin borders, monospace labels, and ample readable spacing. Adapt the style to document upload and results.

## Operations and limits

Bind to localhost. One document at a time; maximum 20 MB and 20 pages. Validate PDFs and reject unsupported, corrupt, encrypted, or over-limit input with a clear error. Show honest processing stages rather than fabricated percentages. No background document retention or cloud inference.

Delete server-side temporary uploads/intermediate files after analysis succeeds or fails, and clean stale files on startup. Keep the selected PDF as a browser-local object for preview so deleting server files does not break the accepted preview. Results remain in the browser for review/export until replaced or the page closes. Document initial model downloads and startup requirements separately from offline inference.

## Review and next stage

This records the eight discussion points accepted by the user. Review this written spec before the implementation plan. The plan must establish expected results and comparison probes before choosing production OCR/model configurations or implementing the final interface.
