import pytest

from faq.text.normalization import TextValidationError, normalize


@pytest.mark.parametrize("mark", ["ʻ", "ʼ", "‘", "’", "`"])
def test_keyboard_markers_keep_letter_information(mark):
    source = f"O{mark}qish ma{mark}lumot mablag{mark}"
    result = normalize(source)
    assert result.embedding_text == "O'qish ma'lumot mablag'"
    assert result.comparison_key == "o'qish ma'lumot mablag'"
    assert source.endswith(mark)


def test_typography_whitespace_and_case_are_separate():
    result = normalize("\ufeff  ID‑karta\u00a0\tOLISH\r\n")
    assert result.embedding_text == "ID-karta OLISH"
    assert result.comparison_key == "id-karta olish"


@pytest.mark.parametrize(
    "text",
    [
        "“ID-karta”",
        "‘hayotiy vaziyat’",
        "5′",
        "2–3",
        "−5",
        "my.gov.uz",
        "YaTT",
        "ERI",
        "ma´lumot",
        "e\u0301",
        "🙂",
        "ўқиш",
        "o''z",
        "o‘‘z",
    ],
)
def test_preserve_unapproved_distinctions(text):
    expected = "é" if text == "e\u0301" else text
    assert normalize(text).embedding_text == expected


@pytest.mark.parametrize(
    "left,right",
    [
        ("16 yosh", "6 yosh"),
        ("mumkin", "mumkin emas"),
        ("bo'ladi", "bo'lmaydi"),
        ("she'r", "sher"),
        ("sur'at", "surat"),
    ],
)
def test_meaning_changing_tokens_are_not_erased(left, right):
    a, b = normalize(left), normalize(right)
    assert a.embedding_text == left and b.embedding_text == right
    assert a.comparison_key != b.comparison_key


@pytest.mark.parametrize("text", ["", " \t\n", "a\x00b", "a\x01b", "a\x7fb"])
def test_invalid_text_rejected(text):
    with pytest.raises(TextValidationError):
        normalize(text)


def test_unknown_profile_rejected():
    with pytest.raises(TextValidationError):
        normalize("savol", profile="unknown")


def test_invisible_characters_flagged_not_silently_joined():
    result = normalize("ish\u200bhaqi")
    assert result.embedding_text == "ish\u200bhaqi"
    assert "format_character" in result.flags
    assert "suspicious_markers" in normalize("o‘‘z").flags


@pytest.mark.parametrize(
    "text", ["G‘isht va bog‘", "\ufeff OʻZ  maʼlumot", "‘hayotiy vaziyat’", "ID‑karta", "o‘‘z"]
)
def test_text_and_key_are_idempotent(text):
    first = normalize(text)
    second = normalize(first.embedding_text)
    assert first.embedding_text == second.embedding_text
    assert first.comparison_key == second.comparison_key


@pytest.mark.parametrize("text", [" \ufeffabc", "\ufeff\ufeffabc", "\u00a0\ufeff \ufeffabc"])
def test_leading_bom_whitespace_prefix_is_stable(text):
    first = normalize(text)
    assert first.embedding_text == "abc"
    assert normalize(first.embedding_text).embedding_text == first.embedding_text


def test_interior_bom_is_preserved():
    assert normalize("a\ufeffb").embedding_text == "a\ufeffb"
