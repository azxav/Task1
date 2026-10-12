# Task 1 corpus findings

Audited 2026-10-12 directly from both CSV members of `C:/Users/aziz/Downloads/Telegram Desktop/Files Task_1.rar`, decoded as UTF-8. No source files changed. Counts concern supplied data, not production user traffic. Audit outputs: `task1-data-audit.json`, `task1-semantic-audit.json`, `task1-targeted-audit.json` in this directory.

## Measured inventory

| Observation | Questions | Answers |
|---|---:|---:|
| Rows | 3,048 | 505 |
| ASCII apostrophe occurrences U+0027 | 3,859 | 1,001 |
| Rows containing ASCII apostrophes | 2,171 | 423 |
| Other apostrophe-shaped characters | 0 | 0 |
| Leading/trailing or repeated whitespace | 0 | 0 |
| Rows changed by NFC or NFKC | 0 | 0 |
| Rows changed by casefold | 3,039 | 502 |
| Latin-only rows | 3,048 | 504 |
| Mixed Latin/Cyrillic rows | 0 | 1 |
| Rows containing digits | 6 | 50 |
| Maximum text length in characters | 101 | 452 |

All IDs are unique, all rows active, and every question references an existing answer. There are 9 category IDs and 19 topic IDs, but no category/topic name lookup tables in the archive. 502 answers have 6 question variants; answers 27, 56, 505 have 12 each. 3,030 questions carry the August 4 timestamp, 18 August 11. These are source metadata, not proof of when facts became legally effective.

No duplicate question/answer strings or cross-answer question collisions were found under the exploratory diagnostic key: NFC, Unicode casefold, apostrophe-shaped marks between letters mapped to ASCII, and whitespace collapse. This key is an audit probe, not the approved embedding policy. Absence of string collisions does not establish distinct meanings.

Question patterns: 415 contain `haqida ma'lumot bering`; 349 contain `bo'yicha ma'lumot kerak`; 279 contain `qanaqa`; 162 contain `ID-karta`; 12 contain `my.gov.uz`; 221 have an uppercase token of at least two ASCII letters/digits. Pattern counts overlap. 184 match an exploratory negative/missing/cancellation regex (`emas`, `bo'lmay`, `yo'q`, `olmay`, `rad et`, `bekor`); this is not a count of grammatically negative sentences.

Of the 3,859 question apostrophes, 3,841 sit between letters, 6 follow word-final g (`mablag'`, questions 2419–2424 / answer 403), and 12 are quotation delimiters around `'hayotiy vaziyat'` (questions 2863–2868). Thus a between-letters-only rule misses a real corpus case, while a global apostrophe replacement confuses letter markers and quotation typography. Normalize eligible letter markers; test quotation delimiters separately.

## Specific issues and implications

1. **Keyboard mismatch is incoming, not a dirty index.** Every supplied apostrophe is ASCII. Query `Oʻqish joyidan maʼlumotnoma...` differs in code points from indexed `O'qish joyidan ma'lumotnoma...`. Context-sensitive canonicalization is justified. It will not change any supplied question under the proposed case-preserving profile.
2. **One mixed-script typo occurs in display data.** Answer 499 contains `egа`, where final `а` is Cyrillic U+0430; the rest of the text is Latin. Flag it in an import lint report; preserve verbatim by default. It has no effect on a question-only vector index. Do not use this one example to justify replacing every Cyrillic lookalike in user queries.
3. **Malformed supplied variants exist.** Questions 129 and 867 contain `ma'lumotnomanini`. Questions such as 15 (`ID-karta kunda tayyor haqida ma'lumot bering`) are mechanically awkward. Keep source text; assess natural, independently authored user queries rather than rewriting all variants and claiming the data was corrected.
4. **Some different answer IDs likely represent equivalent answers.** Answers 380/494 both describe opening a minor's bank card with a legal representative; 181/279 both direct users to my.gov.uz and 1148. Treat these as equivalence-review candidates, not automatically verified classes. A score margin between their IDs could incorrectly reject an answerable query.
5. **Overlap is sometimes only partial.** Questions 127/865 ask about a certificate from study/work versus study only. Answers 22/144 overlap for study certificates, while answer 22's question also mentions work. Equivalence must be query-specific; globally merging these IDs would erase a real scope distinction. Check whether the stored answer actually addresses the question.
6. **General questions can have intentionally broad stored answers.** Questions 3031–3036 are generic tax-payment questions linked to answer 56, which also has entrepreneur-specific questions 331–336. The answer itself gives broad tax-portal guidance. Do not reject this linkage merely because one question family names YaTT; evaluate the full answer and the user's requested specificity. Answer 27 correctly covers both registering with a clinic and changing the assigned doctor.
7. **Numbers are poorly represented in question variants.** All 6 digit-containing questions concern age 16 and map to answer 35. This does not test age, date, fee, percentage, or deadline robustness elsewhere. Preserve numbers and create targeted counterexamples; retrieving the nearest topic is insufficient if the answer cannot address the qualifier.

Character-trigram Jaccard flagged 8 cross-answer question pairs at >=0.82; answer-string SequenceMatcher found 2 pairs at >=0.88. These are lexical audit heuristics, not semantic model scores. Examples and exact IDs are in the JSON outputs. Semantic-equivalence candidates 181/279 required reading despite not passing the answer-string threshold.

## Consequence for evaluation

The observed repeated templates justify concern that a random variant split could inflate quality estimates; the audit does not prove how the questions were generated. Use independently written conversational queries, label against complete answers, split original query families before creating keyboard mutations, and separately report supplied-variant holdout results. Include equivalent-ID ambiguity, partial answer coverage, missing apostrophes, acronyms, numbers, negation, and Cyrillic query input. No retrieval/model benchmark has run yet.

## Provenance

SHA-256 of raw archive member bytes:

- questions.csv: `9539ba7c9d64b5f7e9a8ab9d1646431d6ee25a19ecb8768bbf51a7bec2e37e7f`
- answers.csv: `72f65dad6b083d8eab55117ad69941dafc7fcb473a365f63392d8e394f851759`

Reproduce from repository root with the project's Python runtime:

```powershell
python scripts/audit_task1_data.py 'C:/Users/aziz/Downloads/Telegram Desktop/Files Task_1.rar'
```

The audit records Python and Unicode database versions. It uses archive-member reads, not source extraction or source rewrites.

Archive files contain curated service information. This audit does not verify legal facts, fees, hotline numbers, or their currency.
