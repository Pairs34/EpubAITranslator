import logging
import os
from tkinter import messagebox

import customtkinter as ctk
import psutil
import torch

from ...settings import save_settings
from ..widgets.tooltip import ToolTip

log = logging.getLogger(__name__)
RAM_PER_BATCH_GB = 0.05


class ConfigTab:
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main_window = main_window
        s = main_window.settings

        self.batch_var = ctk.StringVar(value=str(s.batch_size))
        self.cpu_var = ctk.StringVar(value=str(s.cpu_usage_percent))
        self.vram_var = ctk.StringVar(value=str(int(s.gpu_vram_fraction * 100)))
        self.beams_var = ctk.StringVar(value=str(s.num_beams))
        self.maxtok_var = ctk.StringVar(value=str(s.max_new_tokens))

        self.batch_load = None
        self.cpu_value = None
        self.cpu_threads = None
        self.cpu_load = None
        self.vram_info = None
        self.vram_value = None
        self.beam_info = None
        self.beam_value = None
        self.maxtok_value = None

        self._build_header()
        self._build_batch()
        self._build_cpu()
        if torch.cuda.is_available():
            self._build_gpu()
            self._refresh_gpu()

        ctk.CTkButton(
            parent, text="💾 Ayarları Kaydet", width=160, height=42,
            font=ctk.CTkFont(weight="bold"), command=self._save,
        ).pack(anchor="e", padx=20, pady=20)

        self._refresh_resources()

    def _build_header(self):
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = os.cpu_count() or 1

        ctk.CTkLabel(
            self.parent, text="Gelişmiş Ayarlar",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=20, pady=(20, 5))

        ctk.CTkLabel(
            self.parent,
            text=f"Sistem: {cpu_count} çekirdek CPU  |  {total_ram:.1f} GB Toplam RAM",
            text_color="#78909c", font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=20, pady=(0, 15))

    def _info_button(self, parent, tip_text):
        btn = ctk.CTkButton(
            parent, text="ℹ️", width=26, height=22,
            fg_color="transparent", hover_color="#37474f",
            font=ctk.CTkFont(size=13),
        )
        ToolTip(btn, tip_text)
        return btn

    def _build_batch(self):
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=20, pady=5)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(anchor="w", padx=10, pady=(10, 0))
        ctk.CTkLabel(header, text="Batch Size (Eşzamanlı Çeviri)",
                     font=ctk.CTkFont(weight="bold")).pack(side="left")
        self._info_button(
            header,
            "Aynı anda kaç metin parçası işleneceğini belirler.\n"
            "• GPU: VRAM'inize göre 4-16 arası seçin.\n"
            "• CPU: 1 veya 2 güvenlidir.\n"
            "• Çok yüksek değer: bellek hatasına (OOM) yol açabilir.",
        ).pack(side="left", padx=(4, 0))

        ctk.CTkLabel(
            frame,
            text="GPU VRAM izin veriyorsa 8, 16 gibi değerler uygundur. "
                 "CPU kullanıyorsanız 1-2 önerilir.",
            text_color="gray", font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=10)

        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(anchor="w", padx=10, pady=(5, 10))
        ctk.CTkEntry(row, textvariable=self.batch_var, width=80).pack(side="left", padx=(0, 10))
        self.batch_load = ctk.CTkLabel(
            row, text="", font=ctk.CTkFont(size=11, weight="bold"),
            width=200, anchor="w",
        )
        self.batch_load.pack(side="left")
        self.batch_var.trace_add("write", self._refresh_resources)

    def _build_cpu(self):
        cpu_count = os.cpu_count() or 1
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=20, pady=5)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(anchor="w", padx=10, pady=(10, 0))
        ctk.CTkLabel(header, text="CPU Kullanım Yüzdesi  (%)",
                     font=ctk.CTkFont(weight="bold")).pack(side="left")
        self._info_button(
            header,
            "CPU çekirdeklerinin kaçta kaçının kullanılacağını belirler.\n"
            "• %25-50: Sistem rahat çalışır, çeviri yavaşlar.\n"
            "• %75: Dengeli — önerilen.\n"
            "• %100: Tüm çekirdekler çeviride, sistem donabilir.",
        ).pack(side="left", padx=(4, 0))

        ctk.CTkLabel(
            frame,
            text=f"Sistem {cpu_count} çekirdeğe sahip. Düşük değer sistemi rahat bırakır, "
                 "yüksek değer çeviriyi hızlandırır.",
            text_color="gray", font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=10)

        slider_row = ctk.CTkFrame(frame, fg_color="transparent")
        slider_row.pack(fill="x", padx=10, pady=(5, 0))
        slider = ctk.CTkSlider(
            slider_row, from_=10, to=100, number_of_steps=18,
            command=self._on_cpu,
        )
        slider.set(int(self.cpu_var.get()))
        slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.cpu_value = ctk.CTkLabel(
            slider_row, text=f"%{self.cpu_var.get()}",
            font=ctk.CTkFont(size=13, weight="bold"), width=45,
        )
        self.cpu_value.pack(side="left")

        detail = ctk.CTkFrame(frame, fg_color="transparent")
        detail.pack(fill="x", padx=10, pady=(2, 10))
        self.cpu_threads = ctk.CTkLabel(detail, text="", font=ctk.CTkFont(size=11), anchor="w")
        self.cpu_threads.pack(side="left")
        self.cpu_load = ctk.CTkLabel(
            detail, text="", font=ctk.CTkFont(size=11, weight="bold"), anchor="e",
        )
        self.cpu_load.pack(side="right", padx=(0, 5))

        self.cpu_var.trace_add("write", self._refresh_resources)

    def _build_gpu(self):
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            frame, text=f"GPU Yük Kontrolü  —  {gpu_name}",
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=(10, 0))

        ctk.CTkLabel(
            frame,
            text=f"Toplam VRAM: {total_vram:.1f} GB  |  "
                 "Düşük limit GPU'yu korur, yüksek limit hızlandırır.",
            text_color="gray", font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=10)

        self.vram_value = self._slider_row(
            frame, "VRAM Kullanım Limiti", self.vram_var,
            from_=20, to=100, steps=16, command=self._on_vram,
            display=lambda v: f"%{int(v)}",
            tooltip=(
                "GPU bellek (VRAM) kullanımını sınırlar.\n"
                "• %50-70: GPU'nun ömrü uzar.\n"
                "• %90-100: En hızlı çeviri.\n"
                "• Çok düşük: model yüklenmeyebilir."
            ),
        )
        self.vram_info = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=11), anchor="w")
        self.vram_info.pack(anchor="w", padx=10, pady=(0, 4))

        self.beam_value = self._slider_row(
            frame, "Beam Search (Kalite/Hız)", self.beams_var,
            from_=1, to=6, steps=5, command=self._on_beam,
            display=lambda v: str(max(1, int(round(v)))),
            tooltip=(
                "Model kaç alternatif çeviri yolu deneyeceğini belirler.\n"
                "• 1 (Greedy): En hızlı.\n"
                "• 2-3: Dengeli.\n"
                "• 4-6: Yüksek kalite, çok yavaş ve VRAM yoğun."
            ),
        )
        self.beam_info = ctk.CTkLabel(
            frame, text="", font=ctk.CTkFont(size=11, weight="bold"), anchor="w",
        )
        self.beam_info.pack(anchor="w", padx=10, pady=(0, 4))

        self.maxtok_value = self._slider_row(
            frame, "Maks. Çıktı Token", self.maxtok_var,
            from_=64, to=1024, steps=15, command=self._on_maxtok,
            display=lambda v: str(int(round(v / 64)) * 64),
            tooltip=(
                "Tek seferde üretilebilen maksimum token sayısı (1 token ≈ 0.75 kelime).\n"
                "• 128-256: Kısa cümleler.\n"
                "• 512: Çoğu EPUB paragrafı için ideal.\n"
                "• 1024: Çok uzun paragraflar."
            ),
        )

        ctk.CTkLabel(
            frame,
            text="Kısa metinler için 256 yeterli. Uzun paragraflar için 512-1024 kullanın.",
            text_color="gray", font=ctk.CTkFont(size=11),
            wraplength=520, justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 10))

    def _slider_row(self, parent, label, var, from_, to, steps, command, display, tooltip):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(8, 2))

        header = ctk.CTkFrame(row, fg_color="transparent")
        ctk.CTkLabel(
            header, text=label, width=155, anchor="w", font=ctk.CTkFont(size=12),
        ).pack(side="left")
        self._info_button(header, tooltip).pack(side="left")
        header.pack(side="left")

        slider = ctk.CTkSlider(row, from_=from_, to=to, number_of_steps=steps, command=command)
        slider.set(float(var.get()))
        slider.pack(side="left", fill="x", expand=True, padx=(8, 8))

        value_label = ctk.CTkLabel(
            row, text=display(float(var.get())),
            font=ctk.CTkFont(size=13, weight="bold"), width=45,
        )
        value_label.pack(side="left")
        return value_label

    def _on_cpu(self, value):
        pct = int(value)
        self.cpu_var.set(str(pct))
        if self.cpu_value:
            self.cpu_value.configure(text=f"%{pct}")

    def _on_vram(self, value):
        pct = int(value)
        self.vram_var.set(str(pct))
        if self.vram_value:
            self.vram_value.configure(text=f"%{pct}")
        self._refresh_gpu()

    def _on_beam(self, value):
        beams = max(1, int(round(value)))
        self.beams_var.set(str(beams))
        if self.beam_value:
            self.beam_value.configure(text=str(beams))
        self._refresh_gpu()

    def _on_maxtok(self, value):
        snapped = int(round(value / 64)) * 64
        self.maxtok_var.set(str(snapped))
        if self.maxtok_value:
            self.maxtok_value.configure(text=str(snapped))

    def _refresh_gpu(self):
        if not torch.cuda.is_available():
            return
        try:
            vram_pct = int(float(self.vram_var.get()))
            beams = int(self.beams_var.get())
        except ValueError:
            return

        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        allocated = total_vram * vram_pct / 100

        if vram_pct <= 50:
            color, msg = "#66bb6a", f"✅ Düşük yük — ~{allocated:.1f} GB VRAM ayrılır"
        elif vram_pct <= 80:
            color, msg = "#ffa726", f"🟠 Orta yük — ~{allocated:.1f} GB VRAM ayrılır"
        else:
            color, msg = "#ef5350", f"🔴 Tam güç — ~{allocated:.1f} GB VRAM ayrılır"

        if self.vram_info:
            self.vram_info.configure(text=msg, text_color=color)
        if self.vram_value:
            self.vram_value.configure(text=f"%{vram_pct}", text_color=color)

        if beams == 1:
            color, msg = "#66bb6a", "✅ Greedy (en hızlı)"
        elif beams <= 3:
            color, msg = "#ffa726", f"🟠 {beams} beam — dengeli kalite/hız"
        else:
            color, msg = "#ef5350", f"🔴 {beams} beam — yüksek kalite, yavaş"

        if self.beam_info:
            self.beam_info.configure(text=msg, text_color=color)
        if self.beam_value:
            self.beam_value.configure(text=str(beams), text_color=color)

    def _refresh_resources(self, *_):
        cpu_count = os.cpu_count() or 1
        try:
            cpu_pct = int(float(self.cpu_var.get()))
            batch = int(self.batch_var.get())
        except ValueError:
            return

        threads = max(1, cpu_count * cpu_pct // 100)

        if cpu_pct <= 40:
            cpu_color, cpu_msg = "#66bb6a", "✅ Hafif yük — sistem rahat"
        elif cpu_pct <= 70:
            cpu_color, cpu_msg = "#ffa726", "🟠 Orta yük — diğer işler yavaşlayabilir"
        else:
            cpu_color, cpu_msg = "#ef5350", "🔴 Yoğun yük — sistem ısınabilir"

        if self.cpu_threads:
            self.cpu_threads.configure(text=f"{threads} / {cpu_count} çekirdek kullanılacak")
        if self.cpu_load:
            self.cpu_load.configure(text=cpu_msg, text_color=cpu_color)
        if self.cpu_value:
            self.cpu_value.configure(text=f"%{cpu_pct}", text_color=cpu_color)

        ram_avail = psutil.virtual_memory().available / (1024 ** 3)
        estimated = batch * RAM_PER_BATCH_GB
        ratio = estimated / max(ram_avail, 0.1)

        if ratio < 0.3:
            color, msg = "#66bb6a", f"✅ Güvenli — ~{estimated:.1f} GB RAM tahmini"
        elif ratio < 0.7:
            color, msg = "#ffa726", f"🟠 Dikkat — ~{estimated:.1f} GB RAM tahmini"
        else:
            color, msg = "#ef5350", f"🔴 Riskli — kullanılabilir: {ram_avail:.1f} GB"

        if self.batch_load:
            self.batch_load.configure(text=msg, text_color=color)

    def _save(self):
        try:
            batch = int(self.batch_var.get().strip())
            cpu_pct = int(float(self.cpu_var.get().strip()))
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli rakamlar girin.")
            return

        if not 1 <= batch <= 64:
            messagebox.showerror("Hata", "Batch Size 1-64 arasında olmalıdır.")
            return
        if not 10 <= cpu_pct <= 100:
            messagebox.showerror("Hata", "CPU yüzdesi 10-100 arasında olmalıdır.")
            return

        s = self.main_window.settings
        s.batch_size = batch
        s.cpu_usage_percent = cpu_pct

        summary = [
            f"Batch: {batch}",
            f"CPU: %{cpu_pct} ({s.cpu_threads()} çekirdek)",
        ]

        if torch.cuda.is_available():
            try:
                vram_fraction = int(float(self.vram_var.get())) / 100
                beams = max(1, int(float(self.beams_var.get())))
                max_tokens = max(64, int(round(float(self.maxtok_var.get()) / 64)) * 64)
            except ValueError:
                messagebox.showerror("Hata", "GPU ayarları geçersiz.")
                return

            s.gpu_vram_fraction = vram_fraction
            s.num_beams = beams
            s.max_new_tokens = max_tokens
            torch.cuda.set_per_process_memory_fraction(vram_fraction, device=0)
            summary.append(f"VRAM: %{int(vram_fraction * 100)} | Beam: {beams} | MaxToken: {max_tokens}")

        save_settings(s)
        messagebox.showinfo("Kaydedildi", "Ayarlar uygulandı.\n" + "  |  ".join(summary))
