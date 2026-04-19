import logging
import threading
import webbrowser
from tkinter import messagebox

import customtkinter as ctk

log = logging.getLogger(__name__)


def _friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if "401" in msg or "unauthorized" in low or "invalid" in low and "token" in low:
        return (
            "Hugging Face token geçersiz veya süresi dolmuş.\n\n"
            "Çözüm: 'Model & Ayarlar' sekmesinden yeni bir token alıp kaydedin.\n"
            "(https://huggingface.co/settings/tokens)"
        )
    if "403" in msg or "forbidden" in low:
        return "Bu modele erişim izniniz yok (403 Forbidden)."
    if "rate limit" in low or "429" in msg:
        return "Hugging Face istek limitine takıldınız (429). Birkaç dakika sonra tekrar deneyin."
    if "connection" in low or "timeout" in low or "network" in low:
        return "Ağ bağlantı hatası. İnternet bağlantınızı kontrol edin."
    return f"Beklenmeyen hata:\n{msg}"


class ModelBrowser(ctk.CTkToplevel):
    def __init__(self, parent, on_select):
        super().__init__(parent)
        self.title("Hugging Face Model Tarayıcısı")
        self.geometry("700x500")
        self.transient(parent)
        self.grab_set()
        self.on_select = on_select

        try:
            from huggingface_hub import HfApi
            self.api = HfApi()
        except ImportError:
            ctk.CTkLabel(self, text="huggingface_hub kütüphanesi yüklenemedi!").pack(pady=20)
            self.api = None
            return

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=10)

        self.query_var = ctk.StringVar(value="translate")
        entry = ctk.CTkEntry(top, textvariable=self.query_var, width=300)
        entry.pack(side="left", padx=5)
        entry.bind("<Return>", lambda _e: self.search())

        ctk.CTkButton(top, text="Ara", width=80, command=self.search).pack(side="left", padx=5)

        self.status = ctk.CTkLabel(top, text="")
        self.status.pack(side="left", padx=10)

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.search()

    def search(self):
        query = self.query_var.get().strip()
        if not query or self.api is None:
            return

        for child in self.scroll.winfo_children():
            child.destroy()

        self.status.configure(text="Aranıyor...")
        self.update_idletasks()

        def task():
            try:
                models = list(self.api.list_models(search=query, limit=15, sort="downloads"))
                self.after(0, self._populate, models)
            except Exception as exc:
                log.warning("Model arama hatası: %s", exc)
                self.after(0, self._show_error, exc)

        threading.Thread(target=task, daemon=True).start()

    def _show_error(self, exc):
        self.status.configure(text="Arama başarısız ⚠", text_color="#ef5350")
        messagebox.showwarning("Model Arama Hatası", _friendly_error(exc), parent=self)

    def _populate(self, models):
        self.status.configure(text=f"{len(models)} model bulundu.")
        for m in models:
            self._render_row(m)

    def _render_row(self, model):
        row = ctk.CTkFrame(self.scroll)
        row.pack(fill="x", pady=4)

        ctk.CTkLabel(row, text=model.id, font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=10, pady=(5, 0)
        )

        downloads = getattr(model, "downloads", 0)
        likes = getattr(model, "likes", 0)
        ctk.CTkLabel(
            row,
            text=f"İndirme: {downloads} | Beğeni: {likes}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(anchor="w", padx=10)

        actions = ctk.CTkFrame(row, fg_color="transparent")
        actions.pack(anchor="e", padx=10, pady=(0, 5))

        ctk.CTkButton(
            actions, text="Tarayıcıda Aç", width=100,
            command=lambda mid=model.id: webbrowser.open(f"https://huggingface.co/{mid}"),
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            actions, text="Seç", width=60,
            fg_color="#2e7d32", hover_color="#1b5e20",
            command=lambda mid=model.id: self._select(mid),
        ).pack(side="left")

    def _select(self, model_id):
        self.on_select(model_id)
        self.destroy()
