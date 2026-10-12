import unicodedata as ud

from ..text.normalization import TextValidationError, normalize, validate_text


def prepare_profile(text: str, profile: str) -> str:
    validate_text(text)
    if profile == "raw":
        return text
    result = normalize(text)
    if profile == "uz_surface_v1":
        return result.embedding_text
    if profile == "uz_surface_casefold_v1":
        return ud.normalize("NFC", result.embedding_text.casefold())
    raise TextValidationError(f"Unknown preparation profile: {profile}")
