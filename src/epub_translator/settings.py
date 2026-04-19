import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w

from .constants import BASE_DIR, CONFIG_PATH


@dataclass
class Settings:
    nllb_speed: str = "facebook/nllb-200-distilled-600M"
    nllb_quality: str = "facebook/nllb-200-distilled-1.3B"
    llm_speed: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_quality: str = "Qwen/Qwen2.5-7B-Instruct"
    cache_dir: str = "models"
    custom_models: list[str] = field(default_factory=list)

    batch_size: int = 8
    cpu_usage_percent: int = 50
    gpu_vram_fraction: float = 0.9
    num_beams: int = 1
    max_new_tokens: int = 512

    output_dir: str = "translated"
    hf_token: str = ""

    @property
    def cache_path(self) -> Path:
        return BASE_DIR / self.cache_dir

    @property
    def output_path(self) -> Path:
        return BASE_DIR / self.output_dir

    def cpu_threads(self) -> int:
        cores = os.cpu_count() or 1
        return max(1, cores * self.cpu_usage_percent // 100)


def _parse_custom(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    if not path.exists():
        return Settings()

    with path.open("rb") as f:
        raw = tomllib.load(f)

    s = Settings()
    m = raw.get("model", {})
    s.nllb_speed = m.get("nllb_speed", s.nllb_speed)
    s.nllb_quality = m.get("nllb_quality", s.nllb_quality)
    s.llm_speed = m.get("llm_speed", s.llm_speed)
    s.llm_quality = m.get("llm_quality", s.llm_quality)
    s.cache_dir = m.get("cache_dir", s.cache_dir)
    s.custom_models = _parse_custom(m.get("custom_models"))

    t = raw.get("translator", {})
    s.batch_size = int(t.get("batch_size", s.batch_size))
    s.cpu_usage_percent = int(t.get("cpu_usage_percent", s.cpu_usage_percent))
    s.gpu_vram_fraction = float(t.get("gpu_vram_fraction", s.gpu_vram_fraction))
    s.num_beams = int(t.get("num_beams", s.num_beams))
    s.max_new_tokens = int(t.get("max_new_tokens", s.max_new_tokens))

    s.output_dir = raw.get("output", {}).get("dir", s.output_dir)
    s.hf_token = str(raw.get("auth", {}).get("hf_token", "")).strip()
    return s


def save_settings(s: Settings, path: Path = CONFIG_PATH) -> None:
    payload = {
        "model": {
            "nllb_speed": s.nllb_speed,
            "nllb_quality": s.nllb_quality,
            "llm_speed": s.llm_speed,
            "llm_quality": s.llm_quality,
            "cache_dir": s.cache_dir,
            "custom_models": ",".join(s.custom_models),
        },
        "translator": {
            "batch_size": s.batch_size,
            "cpu_usage_percent": s.cpu_usage_percent,
            "gpu_vram_fraction": s.gpu_vram_fraction,
            "num_beams": s.num_beams,
            "max_new_tokens": s.max_new_tokens,
        },
        "output": {"dir": s.output_dir},
        "auth": {"hf_token": s.hf_token},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(payload, f)


def export_token_to_env(s: Settings) -> None:
    if s.hf_token:
        os.environ["HF_TOKEN"] = s.hf_token
