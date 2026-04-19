from .. import constants


def detect_language(text: str) -> str:
    sample = text[:400].strip()
    if not sample:
        return "en"
    try:
        from langdetect import detect
        code = detect(sample)
    except Exception:
        return "en"
    return constants.LANGDETECT_TO_HF.get(code, code)
