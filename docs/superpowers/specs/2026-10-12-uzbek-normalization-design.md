# Uzbek normalization and retrieval robustness design

## Goal and scope

Make Task 1 robust to Uzbek keyboard and writing variation while retaining the distinctions that determine the correct FAQ answer. This refines the normalization, evaluation, and answer-ambiguity sections of `2026-10-12-task1-design.md`. It governs text preparation and quality experiments, not the rest of the implementation. Source-backed language/Unicode findings: `docs/research/uzbek-normalization-sources.md`. Measured corpus findings: `docs/research/task1-data-findings.md`.

Normalization addresses equivalent representations. Misspellings, missing apostrophes, morphology, synonyms, transliteration, answer coverage, and semantic equivalence require retrieval experiments or label review. They must not become silent global substitutions.

## Corpus-driven priorities

All 3,048 questions are Latin, NFC-stable, whitespace-clean, and use only ASCII U+0027 apostrophes. Apostrophes occur in 2,171 questions (3,859 occurrences). A conservative profile should change zero question strings. Casefold would change 3,039 questions; evaluate it separately rather than assuming BGE-M3 needs lowercasing. 162 questions contain ID-karta, 221 have uppercase acronym-like tokens, and 12 contain my.gov.uz. Preserve those structures.

Question-marker contexts are concrete: 3,841 between letters, 6 word-final markers in `mablag'` (questions 2419–2424), and 12 quotation delimiters in `'hayotiy vaziyat'` (questions 2863–2868). The earlier between-letters-only proposal is insufficient. Both terminal markers and quotation delimiters need separate fixtures.

The single Cyrillic lookalike is in answer 499, not indexed question text. Several question variants are grammatically awkward. Different answer IDs can be equivalent or only partly overlap. These data issues need visible lint/review, not destructive cleanup.

## Three representations

- `original_text`: untouched source or user text. Store imported text verbatim; return stored answers verbatim. Do not log full user questions by default.
- `embedding_text`: case-preserving surface normalization under profile `uz_surface_v1`. Both query and index use the same function and profile. BGE-M3 remains responsible for tokenization.
- `comparison_key`: NFC applied after Unicode casefold of embedding_text, for duplicate detection and lexical evaluation only. Keep apostrophes and other meaningful punctuation. A comparison-key match is not an automatic answer or proof of semantic equivalence.

`NormalizationResult` contains embedding_text, comparison_key, profile, tuple of applied rule names, tuple of diagnostic flags. Diagnostics describe transformations; they are not retrieval confidence. Helpers are pure, deterministic, linear in input length, and independent of database/model/network access.

## Surface profile uz_surface_v1

Apply the following rules in order; repeat application must produce identical text and keys. Applied-rule metadata records only changes on the current invocation, so it need not remain identical on a second invocation.

1. Remove U+FEFF characters within the complete leading whitespace/BOM prefix. This remains stable when whitespace trimming exposes a BOM, including repeated BOMs. Decode import files as UTF-8 with optional BOM before this step. Do not erase internal U+FEFF after the first non-whitespace/non-BOM character.
2. Apply Unicode NFC. Do not apply NFKC, strip combining marks, or infer language from visual character similarity.
3. Map typographic letter markers U+02BB, U+02BC, U+2018, U+2019 to ASCII U+0027 only when they are between Latin letters, or immediately follow ASCII o/O/g/G. The latter covers word-final `bog‘`, `togʻ`, and isolated letter notation. Resolve context from Unicode Latin letters, not `\w`: digits and underscores are not letters. Do not normalize curly quotation marks elsewhere.
4. Map U+0060 grave/backtick under the same letter context as a documented keyboard-tolerance rule. Do not map accents U+00B4/U+0301, primes U+2032/U+02B9, or other lookalikes in v1. Record the backtick rule separately so its effect can be measured. A single marker surrounded by other marker characters is flagged as suspicious punctuation and left unchanged; do not collapse repeated apostrophes.
5. Map U+2010 HYPHEN and U+2011 NON-BREAKING HYPHEN to ASCII hyphen only when immediately between Latin letters (`ID‑karta`). Preserve U+2013/U+2014, U+2212 minus, digit-containing ranges/dates, ordinary hyphens, and all other punctuation. Typography beyond this documented case is not silently rewritten.
6. Replace Unicode whitespace runs with one ASCII space and trim. Use Unicode whitespace classification, not a list limited to ASCII spaces. Do not remove a character and join previously separated words.
7. Leave casing, numbers, letter order, repeated letters, suffixes, domains, acronyms, and remaining punctuation intact. Build comparison_key using NFC after casefold; it must not become embedding_text implicitly.

Other invisible/format characters, mixed scripts within a token, and combining marks that remain after NFC are diagnostic flags. Preserve them in this profile; do not blanket-delete default-ignorable code points. Import/input validation rejects NUL and non-whitespace C0/DEL control characters. Unsupported script or emoji alone is not a validation error; handle through search/no-match. Whitespace-only normalized queries are invalid input.

Typography rules after o/g can also canonicalize a closing single quote after those letters; the marker is retained, and original display text remains intact. This is a bounded ambiguity of the representation, not a claim to distinguish every quotation from every Uzbek letter. Include quoted-token fixtures and assess retrieval rather than promising perfect linguistic parsing.

The chosen ASCII representation matches the supplied corpus. It is an internal matching convention, not a proposed reform of Uzbek spelling. It keeps the marker present in oʻ/gʻ and tutuq forms; it does not reconstruct their original Unicode glyph identity.

## Examples and preservation contracts

| Input | embedding_text |
|---|---|
| `Oʻqish joyidan maʼlumotnoma` | `O'qish joyidan ma'lumotnoma` |
| `G‘isht va bog‘` | `G'isht va bog'` |
| `ro` + backtick + `yxatdan o` + backtick + `tish` | `ro'yxatdan o'tish` |
| `ID‑karta` | `ID-karta` |
| `  ID-karta` + NBSP + `olish  ` | `ID-karta olish` |
| `“ID-karta”` | unchanged |
| `16 yosh`, `6 yosh` | distinct and unchanged |
| `mumkin`, `mumkin emas` | distinct and unchanged |
| `bo'ladi`, `bo'lmaydi` | distinct and unchanged |
| `she'r`, `sher` and `sur'at`, `surat` | distinct and unchanged |
| `my.gov.uz`, `my.soliq.uz`, `ERI`, `YaTT` | unchanged |
| `5′`, `2–3`, `−5` | unchanged |

The last examples are preservation tests; they do not all occur in the supplied data. Test numbers and negation in full questions too. Distinct normalized text alone does not guarantee the embedding model distinguishes it.

## Deferred transformations and candidate experiments

No default stemming, stopword removal, suffix removal, q/k or h/x substitution, repeated-letter collapse, grammar rewriting, dictionary-wide spelling correction, or missing-apostrophe restoration. `-ma` substring occurrence is not a negation detector. Do not globally equate pasport with ID-karta: travel passport and identity-card intents can differ. Do not conflate authentication, electronic signature, and acronym expansions.

Experiments are separately versioned:

- **Casing:** compare raw, conservative surface, and surface-plus-casefold. Both indexed and query text must use the selected embedding profile. Pin model/tokenizer revisions and inspect actual tokenizer outputs; the artifact has no explicit Lowercase stage, but its precompiled character map is not yet decoded or executed here.
- **Cyrillic query candidates:** classify scripts per token, preserve the original query vector, and optionally add an Uzbek-aware transliterated Latin candidate. Use reviewed contextual rules (including ў/ғ/қ/ҳ, е/э/ё/ю/я, ц, ъ/ь); preserve Latin domains/acronyms/numerals. Pure Cyrillic is not automatically Uzbek. Do not reverse-transliterate the Latin corpus, blanket-map confusables, or promise reversible transliteration.
- **Missing apostrophes/typos:** first measure dense retrieval on `oqish`, `malumot`, `royxat`, ID karta and common typos. If failures justify aliases or lexical/hybrid retrieval, add a reviewed candidate channel without replacing canonical text. Reject ambiguous corrections rather than injecting certainty. Dataset-specific corrections, such as ma'lumotnomanini, are reviewable overlays, never edits to originals.

For multiple query channels, retain channel provenance, union candidates, and deduplicate by question ID before answer grouping. Scores from the same model are still affected by query transformation; maximum-score merging can increase false acceptance. Calibrate the whole selected pipeline independently, including no-match cases and conflicts between channels. Do not average unrelated embeddings or borrow thresholds from a different pipeline.

## Answer interpretation and ambiguity

Normalization cannot prove that an answer covers a question. Inspect full answers for evaluation labels. Mark verified globally equivalent answer IDs in a small reviewed registry only when their scopes really agree. Answers 380/494 and 181/279 are candidates for review. Answers 22/144 have partial overlap, so do not globally merge them; a study-certificate test may allow both, while a work-certificate query requires its own coverage assessment.

During retrieval, compute an ambiguity margin between distinct verified equivalence groups, not merely different answer IDs. Keep original answer IDs in responses; within a verified group choose the best-scoring stored answer, breaking score ties by numeric answer ID. A query-specific acceptable-ID set is an evaluation label and must not automatically alter production grouping. Unverified pairs remain separate. Snapshot and version the equivalence registry with the index/configuration; changes require recalibration. No globally selected thresholds are specified before measurements.

## Evaluation design

1. Character-level fixtures prove canonical equivalence and meaning preservation; property tests cover idempotence and retention of significant tokens. Mutate all 2,171 apostrophe-bearing source questions with the four typographic variants and backticks only at eligible letter-marker positions; keep quotation delimiters fixed for these equality checks. Test smart outer quotes separately as retained punctuation. These checks can run without loading the model. Check all 3,048 originals remain byte-for-byte unchanged under uz_surface_v1. Audit exact and normalized duplicates and cross-answer collisions; lint findings never trigger automatic deletion.
2. Curate 72 independently written answerable base-query families: 8 per source category ID, balancing topic IDs, precise intent, short conversational language, and difficult neighboring answers. Split 4 families/category into calibration and 4 into final test before producing any variants. All variants of a family stay in its split. Add 36 no-match/clarification families split 18/18, including unrelated topics and near-domain queries that the full stored answer cannot address. Labels include expected answer IDs, coverage rationale, condition/scope, and acceptable equivalent IDs where reviewed. Numbers describe a manageable initial benchmark, not statistical proof. This deliberately balanced diagnostic is not a representative estimate of unknown production query traffic; report category counts and that limitation.
3. Produce reviewed slices: typography/whitespace, casing/acronyms, missing apostrophes, spelling mistakes, colloquial/morphological variation, Cyrillic, mixed-script artifacts, negation/numeric qualifiers, and overlapping/equivalent answers. Preserve expected labels only when the mutation demonstrably preserves meaning; changed numbers/negation require relabeling. Synthetic mutations test robustness, not independent semantic generalization.
4. Add a separate supplied-variant holdout diagnostic, holding out a variant family where identifiable, with explicit warning about template similarity. Do not mix its score into independently authored-query performance. Avoid exact normalized overlap with the evaluation index; disclose overlap if present in any diagnostic.
5. Compare P0 raw model text, P1 uz_surface_v1, P2 casefold profile, and optional Cyrillic/typo candidates. P0 and P1 should reuse the same indexed embeddings for this unchanged corpus; transformed query text is the expected difference. Cache embeddings by exact prepared-text hash plus model revision, never just answer ID. Evaluate all profiles on calibration, select the simplest profile that addresses observed failures without adding incorrect accepted answers on calibration, then freeze it and its thresholds before final testing. A bad final result is reported; it is not a license to silently tune on the held-out test.
6. Report retrieval top-1 and recall@5 by answer and reviewed equivalent group, accepted-answer precision, answer coverage, false acceptance of no-match cases, false rejection of answerable cases, per-slice counts, paired changes against raw, decision disagreements across same-meaning variants, and p50/p95 warm latency. Preserve strict-ID metrics alongside semantic labels so equivalence handling is transparent. Record sample size and failed cases; no claim of improvement without real BGE-M3 measurements.

Safe-equivalent inputs that normalize to identical embedding text must have identical model token IDs and identical candidate scores under a deterministic, unchanged evaluation snapshot. Rank tie handling must be deterministic. Distinct preserved text must remain distinguishable before tokenization; separately inspect and report if the tokenizer or retrieval fails to preserve that distinction. Do not promise that semantic negative pairs always yield different answers: labels depend on answer coverage.

## Integration and versioning

Put the pure normalizer in `src/faq/text/normalization.py`. Script diagnostics belong in `src/faq/text/scripts.py`; candidate transforms, if justified, in `src/faq/text/candidates.py`. Import and search call the shared normalizer. Store original_text and embedding_text, and write profile/model/tokenizer versions, corpus hashes, and equivalence-registry version into published index metadata. Refuse readiness when configured index text profile/model metadata disagrees with the published generation. Query candidate policy and thresholds belong in the frozen evaluation/runtime manifest.

Answer text remains verbatim. No answer-only cleanup is needed for a question-only embedding index. Changing index preparation requires a new versioned generation; reuse vectors only when exact prepared text and model revision match. Query-only changes require new policy version and threshold evaluation even when no index rebuild is needed.

## Completion boundary

Implementation and measured results now appear in src/faq/text and docs/evaluation/uzbek-normalization.md. Version 2 uses corrected development labels and fresh test families after review of v1. The two-day task favors a small tested surface profile and explicit limitations over a large unverified language-processing subsystem.
