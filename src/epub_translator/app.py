import logging
import warnings

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "epub_translator"

import customtkinter as ctk

from .settings import export_token_to_env, load_settings
from .ui.main_window import MainWindow, apply_gpu_memory_limit


def _setup_logging():
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("huggingface_hub", "urllib3", "transformers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def main():
    _setup_logging()

    settings = load_settings()
    export_token_to_env(settings)
    apply_gpu_memory_limit(settings)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    MainWindow(settings).mainloop()


if __name__ == "__main__":
    main()
