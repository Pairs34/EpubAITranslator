from .settings import Settings

CUSTOM_PLACEHOLDER = "Özel Model…"


def custom_label(model_id: str) -> str:
    return f"★ {model_id.split('/')[-1]}"


def build_model_options(s: Settings) -> list[tuple[str | None, str | None, str]]:
    options: list[tuple[str | None, str | None, str]] = [
        (s.nllb_speed, "nllb", "NLLB Hız (~2.4 GB)"),
        (s.nllb_quality, "nllb", "NLLB Kalite (~5 GB)"),
        (s.llm_speed, "llm", "LLM Hız (~3 GB) — prompt destekli"),
        (s.llm_quality, "llm", "LLM Kalite (~15 GB) — prompt destekli"),
    ]
    for m in s.custom_models:
        options.append((m, None, custom_label(m)))
    options.append((None, None, CUSTOM_PLACEHOLDER))
    return options


def find_option_by_label(options, label):
    for opt in options:
        if opt[2] == label:
            return opt
    return None


def find_option_by_id(options, model_id):
    for opt in options:
        if opt[0] == model_id:
            return opt
    return None
