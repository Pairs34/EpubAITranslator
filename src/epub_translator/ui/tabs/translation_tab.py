from tkinter import filedialog

import customtkinter as ctk


class TranslationTab:
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main_window = main_window
        self.epub_path_var = ctk.StringVar()

        ctk.CTkLabel(
            parent, text="EPUB Dosyası Seçimi",
            font=ctk.CTkFont(weight="bold"), text_color="#b0bec5",
        ).pack(anchor="w", padx=20, pady=(15, 5))

        file_row = ctk.CTkFrame(parent, fg_color="transparent")
        file_row.pack(fill="x", padx=20)

        ctk.CTkEntry(
            file_row, textvariable=self.epub_path_var,
            placeholder_text="Çevrilecek .epub dosyasının yolu...", height=36,
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(file_row, text="📁  Seç", width=100, height=36, command=self._browse).pack(side="left")

        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.pack(fill="x", padx=20, pady=25)

        self.start_btn = ctk.CTkButton(
            actions, text="▶   Çeviriyi Başlat", width=220, height=48,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=main_window.start_translation,
        )
        self.start_btn.pack(side="left")

        self.cancel_btn = ctk.CTkButton(
            actions, text="✕  Durdur", width=120, height=48,
            fg_color="#c62828", hover_color="#b71c1c",
            command=main_window.cancel_translation, state="disabled",
        )
        self.cancel_btn.pack(side="left", padx=(10, 0))

        self.status = ctk.CTkLabel(
            actions, text="Bekleniyor...", text_color="#90a4ae", font=ctk.CTkFont(size=13),
        )
        self.status.pack(side="right", padx=10)

        progress = ctk.CTkFrame(parent, fg_color="transparent")
        progress.pack(fill="x", padx=20, pady=(10, 0))
        self.progress_bar = ctk.CTkProgressBar(progress, height=14)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(progress, text="0 / 0", width=60, anchor="e")
        self.progress_label.pack(side="right")

        ctk.CTkLabel(
            parent, text="İşlem Günlüğü",
            font=ctk.CTkFont(weight="bold"), text_color="#b0bec5",
        ).pack(anchor="w", padx=20, pady=(20, 5))

        self.log_box = ctk.CTkTextbox(
            parent, height=200,
            font=ctk.CTkFont(family="Consolas", size=11),
            state="disabled",
        )
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("EPUB Dosyaları", "*.epub"), ("Tümü", "*.*")]
        )
        if path:
            self.epub_path_var.set(path)

    def append_log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_progress(self, done, total):
        ratio = (done / total) if total else 0
        self.progress_bar.set(ratio)
        self.progress_label.configure(text=f"{done} / {total}")

    def set_progress_percent(self, percent):
        self.progress_bar.set(percent / 100.0)
        self.progress_label.configure(text=f"%{percent}")

    def reset_progress(self):
        self.progress_bar.set(0)
        self.progress_label.configure(text="—")

    def set_status(self, text, color="#90a4ae"):
        self.status.configure(text=text, text_color=color)

    def set_running(self, running):
        if running:
            self.start_btn.configure(state="disabled")
            self.cancel_btn.configure(state="normal")
        else:
            self.start_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
