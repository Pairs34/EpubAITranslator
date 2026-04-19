import sys
from pathlib import Path

BASE_DIR = (
    Path(sys.executable).parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[2]
)
CONFIG_PATH = BASE_DIR / "config.toml"

SKIP_TAGS = frozenset({"pre", "code", "script", "style"})

SEQ2SEQ_ARCHITECTURES = frozenset(
    {"m2m_100", "marian", "mbart", "mbart50", "t5", "mt5", "fsmt", "pegasus"}
)

MULTIMODAL_KEYWORDS = (
    "translategemma",
    "gemma3",
    "gemma-3",
    "paligemma",
    "llava",
    "idefics",
)

DEFAULT_TRANSLATION_PROMPT = (
    "Sen İngilizce-Türkçe teknik çeviri asistanısın. "
    "Verilen İngilizce metni Türkçeye çevir. "
    "Teknik terimleri (Intune, Azure, Microsoft, JSON, API vb.) olduğu gibi koru. "
    "Yalnızca çeviriyi yaz, açıklama veya ek metin ekleme."
)

LANGDETECT_TO_HF = {
    "en": "en", "tr": "tr", "de": "de", "fr": "fr",
    "es": "es", "it": "it", "pt": "pt", "ru": "ru",
    "ar": "ar", "zh-cn": "zh", "zh-tw": "zh", "ja": "ja",
    "ko": "ko", "nl": "nl", "pl": "pl", "sv": "sv",
}

XHTML_EXTENSIONS = frozenset({".xhtml", ".html", ".htm"})
