import logging
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from ..constants import MULTIMODAL_KEYWORDS, SEQ2SEQ_ARCHITECTURES

log = logging.getLogger(__name__)
_PROGRESS_RE = re.compile(r"(\d+)%\|")


@dataclass
class Translator:
    tokenizer: Any
    model: Any
    device: str
    model_kind: str
    forced_bos_id: int | None = None


def parse_hf_model_id(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    if "huggingface.co/" in cleaned:
        parts = cleaned.split("huggingface.co/")[-1].strip("/").split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return cleaned


def auto_detect_model_kind(model_name: str) -> str:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        arch = str(getattr(cfg, "model_type", "")).lower()
    except Exception as exc:
        log.warning("Model tipi belirlenemedi (%s); 'llm' varsayıldı.", exc)
        return "llm"

    if arch in SEQ2SEQ_ARCHITECTURES:
        if arch == "m2m_100" or "nllb" in model_name.lower():
            return "nllb"
        return "seq2seq"
    return "llm"


def is_multimodal_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(k in lower for k in MULTIMODAL_KEYWORDS)


class _StderrTee:
    def __init__(self, original, callback):
        self._original = original
        self._callback = callback

    def write(self, buf):
        result = self._original.write(buf)
        try:
            self._original.flush()
        except Exception:
            pass
        match = _PROGRESS_RE.search(str(buf))
        if match:
            try:
                self._callback(int(match.group(1)))
            except Exception:
                pass
        return result

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass


@contextmanager
def _capture_progress(callback):
    if callback is None:
        yield
        return
    original = sys.stderr
    sys.stderr = _StderrTee(original, callback)
    try:
        yield
    finally:
        sys.stderr = original


def build_translator(
    model_name: str,
    device: str,
    model_kind: str,
    cache_root: Path,
    cpu_threads: int | None = None,
    download_callback: Callable[[int], None] | None = None,
) -> Translator:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    if device == "cpu" and cpu_threads is not None:
        torch.set_num_threads(cpu_threads)

    log.info("Model yükleniyor: %s [%s]", model_name, device.upper())

    cache_dir = cache_root / model_name.replace("/", "--")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    forced_bos_id = None
    kind = model_kind
    used_device = device

    with _capture_progress(download_callback):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            src_lang="eng_Latn" if kind == "nllb" else None,
        )

        if kind in ("nllb", "seq2seq"):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, cache_dir=cache_dir, dtype=dtype
            ).to(device)
            if kind == "nllb":
                forced_bos_id = tokenizer.convert_tokens_to_ids("tur_Latn")

        elif is_multimodal_model(model_name):
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, cache_dir=cache_dir, dtype=dtype, device_map=device
            )
            kind = "multimodal"
            used_device = str(model.device)

        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, cache_dir=cache_dir, dtype=dtype
                ).to(device)
            except Exception as causal_err:
                log.info("CausalLM yüklenemedi (%s), ImageTextToText deneniyor.", causal_err)
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name, cache_dir=cache_dir, dtype=dtype, device_map=device
                )
                kind = "multimodal"
                used_device = str(model.device)

    return Translator(tokenizer, model, used_device, kind, forced_bos_id)
