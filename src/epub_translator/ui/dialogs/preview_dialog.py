import logging
import threading
from tkinter import messagebox

import customtkinter as ctk

from ...core.model_loader import build_translator
from ...core.translator import translate_in_batches

log = logging.getLogger(__name__)

PLACEHOLDER = (
    "Welcome to the EPUB translation tool. "
    "This is a quick test for previewing translations."
)


class PreviewDialog(ctk.CTkToplevel):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.title("Çeviri Önizleme")
        self.geometry("600x450")
        self.transient(main_window)
        self.grab_set()
        self.main_window = main_window

        ctk.CTkLabel(self, text="İngilizce Metin:").pack(anchor="w", padx=10, pady=(10, 0))
        self.input = ctk.CTkTextbox(self, height=120)
        self.input.pack(fill="x", padx=10, pady=5)
        self.input.insert("1.0", PLACEHOLDER)

        self.translate_btn = ctk.CTkButton(self, text="Çevir (Önizleme)", command=self._translate)
        self.translate_btn.pack(pady=10)

        ctk.CTkLabel(self, text="Çeviri Sonucu:").pack(anchor="w", padx=10)
        self.output = ctk.CTkTextbox(self, height=120, state="disabled", text_color="#a5d6a7")
        self.output.pack(fill="x", padx=10, pady=5)

    def _translate(self):
        text = self.input.get("1.0", "end").strip()
        if not text:
            return

        selection = self.main_window.get_selected_model()
        if selection is None:
            messagebox.showerror("Hata", "Lütfen önce bir model seçin veya tespit edin.", parent=self)
            return

        model_id, model_kind = selection
        device = self.main_window.get_selected_device()
        prompt = self.main_window.get_system_prompt()
        s = self.main_window.settings

        self.translate_btn.configure(state="disabled", text="Çevriliyor...")
        self._set_output(
            "Model yükleniyor ve çeviri yapılıyor. İlk yükleme vakit alabilir...\nLütfen bekleyin..."
        )

        def task():
            try:
                def progress(p):
                    self.after(0, lambda: self._set_output(
                        f"Model İndiriliyor... %{p} (Konsolu kontrol edin)"
                    ))

                tr = build_translator(
                    model_id, device, model_kind, s.cache_path,
                    cpu_threads=s.cpu_threads(),
                    download_callback=progress,
                )
                result = translate_in_batches(
                    tr, [text], prompt, s.batch_size, s.max_new_tokens, s.num_beams
                )
                self._show(result[0])
            except Exception as exc:
                log.exception("Önizleme hatası")
                self._show(f"Hata oluştu:\n{exc}")

        threading.Thread(target=task, daemon=True).start()

    def _show(self, result):
        if not self.winfo_exists():
            return
        self.after(0, lambda: self._set_output(result))
        self.after(0, lambda: self.translate_btn.configure(state="normal", text="Çevir (Önizleme)"))

    def _set_output(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")
