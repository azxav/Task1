"""Conservative, corpus-aligned surface preparation; never a spelling corrector."""

import unicodedata as ud
from dataclasses import dataclass

PROFILE = "uz_surface_v1"
MARKERS = frozenset("'ʻʼ‘’`")


class TextValidationError(ValueError):
    """Input cannot be searched safely as text."""


@dataclass(frozen=True)
class NormalizationResult:
    embedding_text: str
    comparison_key: str
    profile: str
    applied_rules: tuple[str, ...]
    flags: tuple[str, ...]


def latin_letter(char: str) -> bool:
    return bool(char) and ud.category(char).startswith("L") and "LATIN" in ud.name(char, "")


def letter_marker_context(text: str, pos: int) -> bool:
    before = text[pos - 1] if pos else ""
    after = text[pos + 1] if pos + 1 < len(text) else ""
    if before in MARKERS or after in MARKERS:
        return False
    return (latin_letter(before) and latin_letter(after)) or before in ("o", "O", "g", "G")


def validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TextValidationError("Question must be text")
    if any((ord(c) < 32 and c not in "\t\n\r") or ord(c) == 127 for c in text):
        raise TextValidationError("Unsupported control character")
    if not text.lstrip("\ufeff").strip():
        raise TextValidationError("Question cannot be empty")


def normalize(text: str, *, profile: str = PROFILE) -> NormalizationResult:
    if profile != PROFILE:
        raise TextValidationError(f"Unknown normalization profile: {profile}")
    validate_text(text)
    rules, flags = [], set()
    # Remove BOMs from the entire leading whitespace/BOM prefix; whitespace
    # trimming must never expose another removable BOM on a second invocation.
    prefix_end = 0
    while prefix_end < len(text) and (text[prefix_end].isspace() or text[prefix_end] == "\ufeff"):
        prefix_end += 1
    prepared = text[:prefix_end].replace("\ufeff", "") + text[prefix_end:]
    if prepared != text:
        rules.append("leading_bom")
    canonical = ud.normalize("NFC", prepared)
    if canonical != prepared:
        rules.append("nfc")
    out = []
    for pos, char in enumerate(canonical):
        replacement = char
        if char in MARKERS:
            before = canonical[pos - 1] if pos else ""
            after = canonical[pos + 1] if pos + 1 < len(canonical) else ""
            if before in MARKERS or after in MARKERS:
                flags.add("suspicious_markers")
            elif char != "'" and letter_marker_context(canonical, pos):
                replacement = "'"
                rules.append("backtick_marker" if char == "`" else "letter_marker")
        elif char in "‐‑" and pos and pos + 1 < len(canonical):
            if latin_letter(canonical[pos - 1]) and latin_letter(canonical[pos + 1]):
                replacement = "-"
                rules.append("letter_hyphen")
        if ud.category(char) == "Cf":
            flags.add("format_character")
        if ud.category(char).startswith("M"):
            flags.add("combining_mark")
        out.append(replacement)
    prepared = "".join(out)
    whitespace = " ".join(prepared.split())
    if whitespace != prepared:
        rules.append("whitespace")
    validate_text(whitespace)
    return NormalizationResult(
        whitespace,
        ud.normalize("NFC", whitespace.casefold()),
        profile,
        tuple(dict.fromkeys(rules)),
        tuple(sorted(flags)),
    )
