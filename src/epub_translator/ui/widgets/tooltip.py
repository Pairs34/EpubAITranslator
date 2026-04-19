import tkinter as tk


class ToolTip:
    BG = "#263238"
    BORDER = "#4fc3f7"
    FG = "#e0f7fa"
    DELAY_MS = 600
    PAD = 6

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4

        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        self.tip.attributes("-topmost", True)

        frame = tk.Frame(
            self.tip, background=self.BG, bd=1, relief="solid",
            highlightbackground=self.BORDER, highlightthickness=1,
        )
        frame.pack()
        tk.Label(
            frame, text=self.text, background=self.BG, foreground=self.FG,
            font=("Segoe UI", 10), wraplength=320, justify="left",
            padx=self.PAD, pady=self.PAD,
        ).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None
