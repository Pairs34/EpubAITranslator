import logging
import threading
import webbrowser
from tkinter import messagebox

import customtkinter as ctk
import torch

from ...constants import DEFAULT_TRANSLATION_PROMPT
from ...core.model_loader import auto_detect_model_kind, parse_hf_model_id
from ...model_registry import (
    CUSTOM_PLACEHOLDER,
    build_model_options,
    custom_label,
    find_option_by_id,
)
from ...settings import save_settings
from ..dialogs.preview_dialog import PreviewDialog
from ..widgets.model_browser import ModelBrowser

log = logging.getLogger(__name__)
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"


class ModelTab:
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main_window = main_window
        self.detected_kind = None

        self.device_var = ctk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        options = main_window.model_options
        self.model_var = ctk.StringVar(value=options[0][2])
        self.custom_id_var = ctk.StringVar()

        self._build_device()
        self._build_model()
        self._build_prompt()
        self._build_auth()

    def _build_device(self):
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=20, pady=(15, 10), ipady=5)

        ctk.CTkLabel(frame, text="Donanım (Device)", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=15, pady=5
        )

        radios = ctk.CTkFrame(frame, fg_color="transparent")
        radios.pack(fill="x", padx=15, pady=5)

        if torch.cuda.is_available():
            ctk.CTkRadioButton(
                radios, text=f"Ekran Kartı (GPU) — {torch.cuda.get_device_name(0)}",
                variable=self.device_var, value="cuda",
            ).pack(side="left", padx=(0, 20))
            ctk.CTkRadioButton(
                radios, text="İşlemci (CPU)", variable=self.device_var, value="cpu",
            ).pack(side="left")
        else:
            ctk.CTkLabel(
                radios, text="GPU bulunamadı, CPU üzerinden yavaş çeviri yapılacaktır.",
                text_color="#ef5350",
            ).pack(side="left")
            ctk.CTkRadioButton(
                radios, text="CPU İşlemci", variable=self.device_var,
                value="cpu", state="disabled",
            ).pack(side="left", padx=20)

    def _build_model(self):
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=20, pady=10, ipady=5)

        ctk.CTkLabel(frame, text="Yapay Zeka Modeli", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=15, pady=5, sticky="w"
        )

        labels = [opt[2] for opt in self.main_window.model_options]
        self.dropdown = ctk.CTkOptionMenu(
            frame, variable=self.model_var, values=labels,
            command=self._on_model_change, width=300, height=34,
        )
        self.dropdown.grid(row=1, column=0, padx=15, sticky="w")

        self.custom_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ctk.CTkEntry(
            self.custom_frame, textvariable=self.custom_id_var,
            placeholder_text="HuggingFace Model Linki veya ID...",
            width=260, height=32,
        ).grid(row=0, column=0, padx=(0, 6))

        ctk.CTkButton(
            self.custom_frame, text="Algıla", width=60, height=32,
            command=self.detect_custom_model,
        ).grid(row=0, column=1, padx=(0, 4))

        ctk.CTkButton(
            self.custom_frame, text="Model Ara", width=80, height=32,
            command=self._open_browser,
        ).grid(row=0, column=2)

        self.kind_label = ctk.CTkLabel(
            self.custom_frame, text="", text_color="#4fc3f7", font=ctk.CTkFont(size=11),
        )
        self.kind_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))

    def _build_prompt(self):
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=20, pady=10, ipady=5)

        ctk.CTkLabel(frame, text="Sistem Mesajı (Prompt)", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=15, pady=(5, 0)
        )

        ctk.CTkLabel(
            frame,
            text="Modele ne yapması gerektiğini anlatan direktif. "
                 "LLM modellerinde aktif, çeviri modellerinde (NLLB/Seq2Seq) etkisizdir.",
            text_color="gray", font=ctk.CTkFont(size=11),
            wraplength=500, justify="left",
        ).pack(anchor="w", padx=15)

        self.prompt_box = ctk.CTkTextbox(frame, height=80, font=ctk.CTkFont(size=12))
        self.prompt_box.pack(fill="x", padx=15, pady=(5, 0))
        self.prompt_box.insert("1.0", DEFAULT_TRANSLATION_PROMPT)

        ctk.CTkButton(frame, text="🔍 Örnek Çeviri Test Et", command=self._open_preview).pack(
            anchor="e", padx=15, pady=5
        )

    def _build_auth(self):
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=20, pady=(10, 20), ipady=5)

        ctk.CTkLabel(
            frame, text="Hugging Face Kimlik Doğrulama",
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=15, pady=(5, 0))

        ctk.CTkLabel(
            frame,
            text="Erişimi kısıtlı (Google/Meta gibi gated) modelleri indirebilmek için HF Token gerekir.",
            text_color="gray", font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=15)

        token_present = bool(self.main_window.settings.hf_token)
        self.status_label = ctk.CTkLabel(
            frame,
            text="Doğrulanıyor..." if token_present else "Giriş Yapılmadı",
            text_color="yellow" if token_present else "#e57373",
            font=ctk.CTkFont(weight="bold"),
        )
        self.status_label.pack(anchor="w", padx=15, pady=2)

        actions = ctk.CTkFrame(frame, fg_color="transparent")
        actions.pack(fill="x", padx=15, pady=5)

        ctk.CTkButton(
            actions, text="1. Token Al (Tarayıcıda Aç)", width=150,
            fg_color="#455a64", hover_color="#37474f",
            command=lambda: webbrowser.open(HF_TOKEN_URL),
        ).pack(side="left", padx=(0, 10))

        self.token_entry = ctk.CTkEntry(
            actions,
            placeholder_text="2. Token kodunu buraya yapıştırın (hf_...)",
            width=250, show="*",
        )
        self.token_entry.pack(side="left", padx=(0, 4))
        if token_present:
            self.token_entry.insert(0, self.main_window.settings.hf_token)

        self.eye_btn = ctk.CTkButton(
            actions, text="👁", width=36, height=32,
            fg_color="#37474f", hover_color="#455a64",
            command=self._toggle_token_visibility,
        )
        self.eye_btn.pack(side="left", padx=(0, 8))

        ctk.CTkButton(actions, text="Kaydet ve Doğrula", command=self._verify_token).pack(side="left")

        if token_present:
            self._validate_async(self.main_window.settings.hf_token, persist=False, notify=False)

    def _toggle_token_visibility(self):
        if self.token_entry.cget("show") == "*":
            self.token_entry.configure(show="")
            self.eye_btn.configure(text="🙈")
        else:
            self.token_entry.configure(show="*")
            self.eye_btn.configure(text="👁")

    def _on_model_change(self, label):
        if label == CUSTOM_PLACEHOLDER:
            self.custom_frame.grid(row=2, column=0, pady=(10, 0), sticky="w", padx=15)
        else:
            self.custom_frame.grid_forget()

    def detect_custom_model(self):
        raw = self.custom_id_var.get().strip()
        if not raw:
            return

        model_id = parse_hf_model_id(raw)
        self.kind_label.configure(text=f"Algılanıyor: {model_id} ...")

        def task():
            kind = auto_detect_model_kind(model_id)
            self.detected_kind = kind
            self.parent.after(0, lambda: self.kind_label.configure(
                text=f"Algılanan Tip: {kind.upper()} Mode"
            ))
            self.parent.after(0, lambda: self._register_model(model_id))

        threading.Thread(target=task, daemon=True).start()

    def _register_model(self, model_id):
        existing = find_option_by_id(self.main_window.model_options, model_id)
        if existing is not None:
            self.model_var.set(existing[2])
            self._on_model_change(existing[2])
            return

        s = self.main_window.settings
        if model_id not in s.custom_models:
            s.custom_models.append(model_id)
            save_settings(s)

        self.main_window.model_options = build_model_options(s)
        labels = [opt[2] for opt in self.main_window.model_options]
        self.dropdown.configure(values=labels)
        label = custom_label(model_id)
        self.model_var.set(label)
        self._on_model_change(label)

    def _open_browser(self):
        def on_select(model_id):
            self.custom_id_var.set(model_id)
            self.detect_custom_model()

        ModelBrowser(self.parent, on_select)

    def _open_preview(self):
        PreviewDialog(self.main_window)

    def _verify_token(self):
        token = self.token_entry.get().strip()
        if not token:
            messagebox.showwarning("Boş Token", "Lütfen bir token girin.")
            return
        self._validate_async(token, persist=True, notify=True)

    def _validate_async(self, token, persist, notify):
        self.status_label.configure(text="Doğrulanıyor...", text_color="yellow")

        def task():
            try:
                from huggingface_hub import HfApi
                info = HfApi(token=token).whoami()
                username = info.get("name") or info.get("fullname") or "—"
            except Exception as exc:
                log.warning("HF token doğrulama başarısız: %s", exc)
                self.parent.after(0, self._on_token_invalid, exc, notify)
                return

            if persist:
                self.main_window.settings.hf_token = token
                save_settings(self.main_window.settings)
                import os
                os.environ["HF_TOKEN"] = token

            self.parent.after(0, self._on_token_valid, username, notify)

        threading.Thread(target=task, daemon=True).start()

    def _on_token_valid(self, username, notify):
        self.status_label.configure(text=f"✅ Giriş: {username}", text_color="#81c784")
        if notify:
            messagebox.showinfo("Başarılı", f"Hugging Face girişi kaydedildi!\nKullanıcı: {username}")

    def _on_token_invalid(self, exc, notify):
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg or "invalid" in msg:
            short = "❌ Token süresi dolmuş veya geçersiz"
        else:
            short = "❌ Doğrulama başarısız"
        self.status_label.configure(text=short, text_color="#e57373")
        if notify:
            messagebox.showerror(
                "Token Geçersiz",
                "Token süresi dolmuş veya geçersiz görünüyor.\n\n"
                "1. 'Token Al (Tarayıcıda Aç)' tuşundan yeni token oluşturun.\n"
                "2. Yeni tokeni kutuya yapıştırıp 'Kaydet ve Doğrula' tuşuna basın.",
            )

    def get_system_prompt(self):
        return self.prompt_box.get("1.0", "end").strip()
