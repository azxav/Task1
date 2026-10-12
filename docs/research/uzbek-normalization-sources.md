# Uzbek FAQ normalization: primary-source notes

Checked 2026-10-12. Facts below are source-supported; recommendations are engineering choices requiring retrieval evaluation.

## Corpus constraint

Parent agent's local audit: 3,048 Latin questions; 3,859 apostrophes, exclusively ASCII U+0027, across 2,171 rows. NFC/NFKC leave questions unchanged; no whitespace artifacts. Answers: 504 Latin, one mixed-script answer (`egа`, Cyrillic U+0430 at answer 499); 1,001 ASCII apostrophes. These are local observations, not linguistic claims. Canonicalize to the corpus's ASCII apostrophe for matching; preserve original display text.

## Orthography facts and implications

- **Letters versus separator:** 1995 rules §5 calls `oʻ` a letter; §28 does likewise for `gʻ`. §32 describes tutuq separately: vowel length/separation in `eʼlon`, `sanʼat`; §17 uses it to distinguish separate s+h in `Isʼhoq`. Deleting every apostrophe therefore erases orthographic information. [Official rules, LexUZ](https://lex.uz/docs/-1625271)
- **Hyphens are structural:** §§51–56 cover paired/repeated words, particles, compounds, and numeric ordinals/dates (`7-sinf`). §53's `-ma` in `uyma-uy` is a linking element; its spelling alone does not establish verbal negation. [Official rules](https://lex.uz/docs/-1625271)
- **Negation:** Rahmatullayev's university textbook describes verbal `-ma`, `emas`, and restricted `yoʻq` constructions; their location in compound verb forms changes interpretation (printed pp.161–162). Preserve them; do not strip suffixes/stopwords or infer sentence polarity from substring `ma`. [Author's textbook, 2006, Ministry-recommended](https://api.ziyonet.uz/uploads/books/761935/5b150f5bb2f42.pdf#page=161)
- **Scope uncertainty:** government news dated 2026-09-10 reports Senate approval of reform (`Ğ`, `Ö`, `Ş`, `Ç`), while the inspected LexUZ alphabet law still displays the 1995 inventory. Effective enactment/date unresolved here. This report targets existing corpus spelling; do not describe 1995 spelling as the conclusively unchanged current alphabet. [Government announcement](https://gov.uz/oz/cirns/news/view/217333), [inspected alphabet law](https://lex.uz/uz/acts/-112286)

## Unicode facts

| Character | Unicode identity | Normalization consequence |
|---|---|---|
| `'` U+0027 | APOSTROPHE | No decomposition |
| `ʻ` U+02BB | MODIFIER LETTER TURNED COMMA | No decomposition |
| `ʼ` U+02BC | MODIFIER LETTER APOSTROPHE | No decomposition |
| `‘` U+2018 / `’` U+2019 | LEFT / RIGHT SINGLE QUOTATION MARK | No decomposition |
| `` ` `` U+0060 | GRAVE ACCENT | No decomposition |
| `´` U+00B4 | ACUTE ACCENT | NFKC expands to space + combining acute |
| `＇` U+FF07 | FULLWIDTH APOSTROPHE | NFKC maps to U+0027 |

Character identities/decompositions: [Unicode 17 UCD](https://unicode.org/Public/17.0.0/ucd/UnicodeData.txt). Thus NFC **and NFKC** do not automatically equate U+0027/02BB/02BC/2018/2019. Their visual similarity is not canonical equivalence. Normalization is not transliteration.

NFC preserves compatibility distinctions; NFKC folds them and can erase distinctions relevant to meaning. Unicode explicitly cautions against indiscriminate NFKC on arbitrary text. [UAX #15 §§1.2–1.3](https://www.unicode.org/reports/tr15/)

Casefold is for caseless comparison, differs from lowercase, can expand characters, and may break normalization; normalize again afterward. Unicode recommends retaining original text alongside folded keys. Optional Turkic I folding is a separate policy; do not assume Turkish casing just because Uzbek is Turkic. [Unicode 17 §5.18.4–5](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-5/)

`NFKC_Casefold` also removes default-ignorable characters, unlike ordinary casefold. That identifier-oriented behavior is not automatically suitable for natural-language embedding input. [Unicode 17 §3.13.3](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/)

Default-ignorable includes soft hyphen, zero-width space/joiners, bidi controls, and variation selectors; it is not synonymous with General_Category `Cf`. Joiners/variation selectors can affect presentation. Avoid blanket removal; inspect actual corpus/query code points and test a documented allowlist. [UCD property data](https://unicode.org/Public/17.0.0/ucd/DerivedCoreProperties.txt), [Unicode 17 §5.21.4](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-5/)

## Transliteration limitations

CLDR has Uzbek-specific, context-dependent rules: `е`→`e`/`ye`, `ў`→`oʻ`, `ғ`→`gʻ`, `ъ`→`ʼ`; `ь` can disappear. A generic Cyrillic replacement table is not equivalent to this transform. Its `ц`→`ts` also differs from some orthographic conversions discussed by Uzbek transliteration researchers. [CLDR transform source](https://raw.githubusercontent.com/unicode-org/cldr/main/common/transforms/uz_Cyrl-uz_Latn.xml)

Mansurov & Mansurov document irreversible information loss (`октябрь`→`oktabr`) and context/lexical ambiguity (`ц` can become `s`/`ts`, Latin `s` can originate from `с`/`ц`). Their test results concern dictionary-word transliteration, not FAQ retrieval or arbitrary mixed Russian/Uzbek text. [Original research, §§1.1–2.1](https://arxiv.org/pdf/2101.05162)

Recommendation: test Cyrillic→Latin as an additional query candidate; retain original query, flag script, and evaluate separately. Do not promise reversibility or silently transliterate every Cyrillic span as Uzbek. No blanket confusable-character mapping: corpus answer 499 needs a targeted, reviewable correction.

## BGE-M3 facts and evaluation boundary

- BAAI card states multilingual support (100+ languages), dense/sparse/multi-vector retrieval, maximum 8,192 tokens. This is **not evidence of Uzbek FAQ quality**, or immunity to apostrophe/casing variants. [Official model card](https://huggingface.co/BAAI/bge-m3)
- Inspected revision `5617a9f61b028005a4858fdac845db406aefb181`: config selects `XLMRobertaTokenizer`; tokenizer JSON declares Unigram model, Metaspace pre-tokenization, and `Sequence(Precompiled, Replace / {2,}/ → " ")`. No explicit `Lowercase` normalizer is declared. The precompiled character map was not decoded here; its precise mappings remain unresolved. [Pinned config](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/tokenizer_config.json), [pinned tokenizer artifact](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json)
- FlagEmbedding loads `AutoTokenizer.from_pretrained`. SentencePiece documentation says its default training rule is `nmt_nfkc`, but allows other rules/custom maps: that default alone does not prove the exact BGE-M3 artifact's full normalization behavior. [Official inference code](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/m3.py), [SentencePiece normalization docs](https://github.com/google/sentencepiece/blob/master/doc/normalization.md)

## Recommended experiment, not established optimum

1. Baseline: raw corpus/model tokenizer; consistent query/index policy.
2. Conservative variant: NFC + trim/collapse whitespace + **word-internal** U+02BB/02BC/2018/2019→ASCII U+0027; retain apostrophe presence, hyphens, numbers, negation, case. Keep outer quote punctuation. Do not globally map grave/acute accents; introduce only as separately measured typo tolerance. Context rules remain imperfect; ASCII cannot reconstruct the distinct source glyph roles.
3. Ablate casefold separately; NFC after fold. Preserve raw display strings. Do not claim casing gains without held-out query evaluation.
4. Separate Cyrillic candidate experiment. Test `ў/ғ/қ/ҳ`, `е/э/ё/ю/я`, `ц`, `ъ/ь`, mixed-script tokens and Russian borrowings.
5. Test tokenizer IDs and top-k answer accuracy on apostrophe variants, positive/negative minimal pairs, dates/hyphens, casing, and quoted text. Pin model/tokenizer/library revisions and normalization version; re-embed when index policy changes. Report collisions and accuracy deltas, not cosmetic cleanup counts.
