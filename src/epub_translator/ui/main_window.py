import logging
import queue
import threading
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
import torch

from ..core.epub_processor import translate_epub
from ..core.model_loader import (
    auto_detect_model_kind,
    build_translator,
    parse_hf_model_id,
)
from ..model_registry import build_model_options, find_option_by_label
from ..settings import Settings
from .tabs.config_tab import ConfigTab
from .tabs.model_tab import ModelTab
from .tabs.translation_tab import TranslationTab

log = logging.getLogger(__name__)
LOG_POLL_MS = 80


class _QueueLogHandler(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink
        self.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"
        ))

    def emit(self, record):
        try:
            self.sink.put_nowait(self.format(record))
        except queue.Full:
            pass


class MainWindow(ctk.CTk):
    def __init__(self, settings: Settings):
        super().__init__()
        self.title("EPUB Çeviri Aracı - Pro")
        self.geometry("860x780")
        self.resizable(False, False)

        self.settings = settings
        self.model_options = build_model_options(settings)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.cancel_event = threading.Event()
        logging.getLogger().addHandler(_QueueLogHandler(self.log_queue))

        self._build_header()
        self._build_tabs()
        self._poll_logs()

    def _build_header(self):
        header = ctk.CTkFrame(self, corner_radius=0, fg_color=("#1a1a2e", "#0d1b2a"))
        header.pack(fill="x")

        ctk.CTkLabel(
            header, text="📚  EPUB Çeviri Aracı",
            font=ctk.CTkFont(size=24, weight="bold"), text_color="#4fc3f7",
        ).pack(anchor="w", padx=24, pady=(16, 2))

        ctk.CTkLabel(
            header,
            text="Yerel AI modeliyle profesyonel düzeyde İngilizce → Türkçe EPUB çevirisi",
            font=ctk.CTkFont(size=12), text_color="#90a4ae",
        ).pack(anchor="w", padx=24, pady=(0, 14))

    def _build_tabs(self):
        self.tabview = ctk.CTkTabview(self, width=800, height=600)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=10)

        translation_frame = self.tabview.add("🚀 Çeviri Merkezi")
        model_frame = self.tabview.add("🧠 Model & Ayarlar")
        config_frame = self.tabview.add("⚙️ Yapılandırma")

        self.translation_tab = TranslationTab(translation_frame, self)
        self.model_tab = ModelTab(model_frame, self)
        self.config_tab = ConfigTab(config_frame, self)

    def get_selected_device(self):
        return self.model_tab.device_var.get()

    def get_system_prompt(self):
        return self.model_tab.get_system_prompt()

    def get_selected_model(self):
        label = self.model_tab.model_var.get()
        opt = find_option_by_label(self.model_options, label)
        if opt and opt[0]:
            kind = opt[1] or self.model_tab.detected_kind or "llm"
            return opt[0], kind

        raw = self.model_tab.custom_id_var.get().strip()
        if not raw:
            messagebox.showerror(
                "Hata",
                "HuggingFace model ID giriniz veya özel bir model araması yapınız.",
            )
            return None

        model_id = parse_hf_model_id(raw)
        kind = self.model_tab.detected_kind or auto_detect_model_kind(model_id)
        return model_id, kind

    def start_translation(self):
        epub_path = Path(self.translation_tab.epub_path_var.get().strip().strip('"'))
        if not epub_path.exists() or epub_path.suffix.lower() != ".epub":
            messagebox.showerror("Hata", "Geçerli bir .epub dosyası seçiniz.")
            return

        selection = self.get_selected_model()
        if selection is None:
            return

        model_id, model_kind = selection
        device = self.get_selected_device()
        prompt = self.get_system_prompt()

        self.cancel_event.clear()
        self.translation_tab.set_running(True)
        self.translation_tab.reset_progress()
        self.translation_tab.set_status("Çalışıyor…", "#4fc3f7")
        self.tabview.set("🚀 Çeviri Merkezi")

        threading.Thread(
            target=self._worker,
            args=(epub_path, model_id, model_kind, device, prompt),
            daemon=True,
        ).start()

    def cancel_translation(self):
        self.cancel_event.set()
        log.warning("İptal isteği gönderildi.")
        self.translation_tab.cancel_btn.configure(state="disabled")
        self.translation_tab.set_status("Durduruluyor...", "#ef5350")

    def _worker(self, epub_path, model_id, model_kind, device, prompt):
        try:
            log.info("Model yükleniyor: %s [%s]", model_id, device.upper())

            def on_download(percent):
                self.after(0, lambda: self.translation_tab.set_progress_percent(percent))
                self.after(0, lambda: self.translation_tab.set_status(
                    f"İndiriliyor... %{percent}", "yellow"
                ))

            tr = build_translator(
                model_id, device, model_kind, self.settings.cache_path,
                cpu_threads=self.settings.cpu_threads(),
                download_callback=on_download,
            )

            log.info("Model hazır → '%s' çevriliyor", epub_path.name)
            self.after(0, self.translation_tab.reset_progress)
            self.after(0, lambda: self.translation_tab.set_status("Çalışıyor…", "#4fc3f7"))

            def on_progress(done, total):
                self.after(0, lambda: self.translation_tab.set_progress(done, total))

            output = translate_epub(
                epub_path, tr, prompt,
                self.settings.batch_size,
                self.settings.max_new_tokens,
                self.settings.num_beams,
                cancel_event=self.cancel_event,
                on_progress=on_progress,
            )

            if output is None:
                self.after(0, lambda: self.translation_tab.set_status("İptal edildi ⏹️", "#ef5350"))
                return

            log.info("Tamamlandı → %s", output)
            self.after(0, lambda: messagebox.showinfo(
                "Mükemmel!",
                f"Çeviri başarıyla tamamlandı!\n\nDosya Yolu:\n{output}",
            ))
            self.after(0, lambda: self.translation_tab.set_status("Tamamlandı ✅", "#81c784"))

        except Exception as exc:
            log.exception("Kritik çeviri hatası")
            msg = str(exc)
            self.after(0, lambda: self.translation_tab.set_status("Hata ❌", "#e57373"))
            self.after(0, lambda: messagebox.showerror("Hata", f"Kritik hata oluştu:\n{msg}"))
        finally:
            self.after(0, lambda: self.translation_tab.set_running(False))

    def _poll_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.translation_tab.append_log(msg)
        except queue.Empty:
            pass
        self.after(LOG_POLL_MS, self._poll_logs)


def apply_gpu_memory_limit(s: Settings):
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(s.gpu_vram_fraction, device=0)
        except Exception as exc:
            log.warning("VRAM limiti uygulanamadı: %s", exc)
