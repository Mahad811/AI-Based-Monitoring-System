#!/usr/bin/env python3
"""
Vital Guardian — Desktop Launcher
A GTK3 splash-screen application that starts the backend server,
shows animated startup progress, then opens the browser when ready.
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, GdkPixbuf, Pango, Gdk

import os
import sys
import time
import subprocess
import threading
import urllib.request
import webbrowser
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
ICON_PATH = ROOT / "appicon_content" / "VitalGuardian_DesktopIcon.png"
VENV      = ROOT / "venv"
SERVER_URL = "http://localhost:8000"
HEALTH_URL = f"{SERVER_URL}/"

# ── Env setup (mirrors run_server.sh) ────────────────────────────────────────
NVIDIA_SITE = VENV / "lib" / "python3.10" / "site-packages" / "nvidia"
_libs = ["cudnn","cublas","cuda_runtime","cufft","cusolver","cusparse",
         "curand","cuda_cupti","nvjitlink","nccl"]
_ld = ":".join(str(NVIDIA_SITE / l / "lib") for l in _libs)
existing = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"]  = f"{_ld}:{existing}" if existing else _ld
os.environ["YOLO_CONFIG_DIR"]  = "/tmp/.ultralytics"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Colours ──────────────────────────────────────────────────────────────────
BG      = "#060910"
PANEL   = "#0d1117"
GOLD    = "#dca54c"
GREEN   = "#34d399"
RED     = "#ef4444"
MUTED   = "#4b5a72"
TEXT    = "#e2e8f0"
BORDER  = "#1a2332"

CSS = f"""
* {{
    font-family: 'Inter', 'Noto Sans', 'Ubuntu', sans-serif;
}}
window, .background {{
    background-color: {BG};
}}
box#splash-box {{
    background-color: {BG};
}}
box#card {{
    background-color: {PANEL};
    border-radius: 18px;
    border: 1px solid {BORDER};
}}
label#app-name {{
    color: {GOLD};
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 3px;
}}
label#app-sub {{
    color: {MUTED};
    font-size: 10px;
    letter-spacing: 1.5px;
}}
label#status-label {{
    color: {TEXT};
    font-size: 12px;
    font-weight: 600;
}}
label#step-label {{
    color: {MUTED};
    font-size: 11px;
}}
progressbar#vg-progress trough {{
    background-color: #111827;
    border-radius: 6px;
    min-height: 7px;
    border: none;
}}
progressbar#vg-progress progress {{
    background-color: {GOLD};
    border-radius: 6px;
    min-height: 7px;
    border: none;
}}
button#btn-cancel {{
    background: transparent;
    background-image: none;
    color: {MUTED};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 5px 18px;
    font-size: 11px;
    box-shadow: none;
}}
button#btn-cancel:hover {{
    color: {RED};
    border-color: {RED};
    background: rgba(239,68,68,0.06);
}}
label#version-lbl {{
    color: {MUTED};
    font-size: 10px;
    opacity: 0.5;
}}
"""

STEPS = [
    "Initializing GPU context…",
    "Loading TensorFlow CUDA libraries…",
    "Starting MoViNet fall classifier…",
    "Loading YOLO person detector…",
    "Warming up vision pipeline…",
    "Initializing Gemini Cognitive Core…",
    "Starting web server on port 8000…",
    "Waiting for server to be ready…",
]


class VitalGuardianLauncher(Gtk.Window):
    def __init__(self):
        super().__init__(title="Vital Guardian")
        self.set_default_size(500, 380)
        self.set_resizable(False)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_decorated(True)

        # Set window icon
        if ICON_PATH.exists():
            try:
                pb = GdkPixbuf.Pixbuf.new_from_file_at_scale(str(ICON_PATH), 64, 64, True)
                self.set_icon(pb)
            except Exception:
                pass

        # Apply CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(CSS.encode())
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self._server_proc  = None
        self._cancelled    = False
        self._step_idx     = 0
        self._progress     = 0.0
        self._pulse_angle  = 0.0

        self._build_ui()
        self.connect("destroy", self._on_destroy)
        self.show_all()

        # Start the launch sequence
        threading.Thread(target=self._launch_server, daemon=True).start()
        GLib.timeout_add(40, self._animate_tick)

    # ── UI Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        outer.set_name("splash-box")
        outer.set_valign(Gtk.Align.FILL)
        outer.set_halign(Gtk.Align.FILL)

        # Card container (centered)
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        card.set_name("card")
        card.set_margin_top(32)
        card.set_margin_bottom(32)
        card.set_margin_start(40)
        card.set_margin_end(40)
        card.set_valign(Gtk.Align.CENTER)

        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        inner.set_margin_top(32)
        inner.set_margin_bottom(28)
        inner.set_margin_start(36)
        inner.set_margin_end(36)

        # ── Logo row ──────────────────────────────────────────────────────────
        logo_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        logo_row.set_halign(Gtk.Align.CENTER)

        # Icon
        if ICON_PATH.exists():
            try:
                pb = GdkPixbuf.Pixbuf.new_from_file_at_scale(str(ICON_PATH), 64, 64, True)
                img = Gtk.Image.new_from_pixbuf(pb)
                logo_row.pack_start(img, False, False, 0)
            except Exception:
                pass

        # Name + sub
        name_col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        name_col.set_valign(Gtk.Align.CENTER)

        name_lbl = Gtk.Label(label="VITAL GUARDIAN")
        name_lbl.set_name("app-name")
        name_lbl.set_halign(Gtk.Align.START)
        name_col.pack_start(name_lbl, False, False, 0)

        sub_lbl = Gtk.Label(label="AI-POWERED ICU MONITORING SYSTEM")
        sub_lbl.set_name("app-sub")
        sub_lbl.set_halign(Gtk.Align.START)
        name_col.pack_start(sub_lbl, False, False, 0)

        logo_row.pack_start(name_col, False, False, 0)
        inner.pack_start(logo_row, False, False, 0)

        # ── Separator ─────────────────────────────────────────────────────────
        sep = Gtk.Box()
        sep.set_size_request(-1, 1)
        sep.override_background_color(Gtk.StateFlags.NORMAL,
            Gdk.RGBA(*[int(BORDER.lstrip('#')[i:i+2], 16)/255 for i in (0,2,4)], 1.0))
        inner.pack_start(sep, False, False, 20)

        # ── Status dot + label ────────────────────────────────────────────────
        status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        status_row.set_halign(Gtk.Align.CENTER)

        self._dot = Gtk.Box()
        self._dot.set_name("dot")
        self._dot.set_size_request(10, 10)
        self._dot.set_valign(Gtk.Align.CENTER)
        status_row.pack_start(self._dot, False, False, 0)

        self._status_lbl = Gtk.Label(label="Starting Vital Guardian…")
        self._status_lbl.set_name("status-label")
        status_row.pack_start(self._status_lbl, False, False, 0)
        inner.pack_start(status_row, False, False, 0)

        # ── Step label ────────────────────────────────────────────────────────
        self._step_lbl = Gtk.Label(label=STEPS[0])
        self._step_lbl.set_name("step-label")
        self._step_lbl.set_halign(Gtk.Align.CENTER)
        inner.pack_start(self._step_lbl, False, False, 6)

        # ── Progress bar (custom drawn) ────────────────────────────────────────
        prog_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        prog_box.set_margin_top(18)

        # GTK progress bar styled
        self._pbar = Gtk.ProgressBar()
        self._pbar.set_fraction(0.0)
        self._pbar.set_size_request(-1, 7)
        self._pbar.set_name("vg-progress")
        prog_box.pack_start(self._pbar, False, False, 0)
        inner.pack_start(prog_box, False, False, 0)

        # Percentage label
        self._pct_lbl = Gtk.Label(label="0%")
        self._pct_lbl.set_name("step-label")
        self._pct_lbl.set_halign(Gtk.Align.END)
        self._pct_lbl.set_margin_top(4)
        inner.pack_start(self._pct_lbl, False, False, 0)

        # ── Bottom row: version + cancel ──────────────────────────────────────
        bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        bottom.set_margin_top(24)

        ver = Gtk.Label(label="v2.0.0 · RTX 4050 · Local Inference")
        ver.set_name("version-lbl")
        ver.set_halign(Gtk.Align.START)
        bottom.pack_start(ver, True, True, 0)

        self._cancel_btn = Gtk.Button(label="Cancel")
        self._cancel_btn.set_name("btn-cancel")
        self._cancel_btn.connect("clicked", self._on_cancel)
        bottom.pack_end(self._cancel_btn, False, False, 0)
        inner.pack_start(bottom, False, False, 0)

        card.pack_start(inner, True, True, 0)
        outer.pack_start(card, True, True, 0)
        self.add(outer)

    # ── Animation tick ────────────────────────────────────────────────────────
    def _animate_tick(self):
        if self._cancelled:
            return False
        # Pulse the dot
        self._pulse_angle = (self._pulse_angle + 0.08) % (2 * 3.14159)
        import math
        alpha = 0.5 + 0.5 * math.sin(self._pulse_angle)
        try:
            r, g, b = (0xef/255, 0x44/255, 0x44/255)  # red
            self._dot.override_background_color(
                Gtk.StateFlags.NORMAL, Gdk.RGBA(r, g, b, 0.4 + 0.6 * alpha))
        except Exception:
            pass
        return True

    # ── Launch sequence ───────────────────────────────────────────────────────
    def _set_status(self, text, step=None):
        def _update():
            self._status_lbl.set_text(text)
            if step is not None and step < len(STEPS):
                self._step_lbl.set_text(STEPS[step])
            return False
        GLib.idle_add(_update)

    def _set_progress(self, frac):
        def _update():
            self._pbar.set_fraction(min(frac, 1.0))
            self._pct_lbl.set_text(f"{int(min(frac,1.0)*100)}%")
            return False
        GLib.idle_add(_update)

    def _launch_server(self):
        # Step 0-1: environment init (visual only)
        self._set_status("Initializing environment…", 0)
        self._set_progress(0.05)
        time.sleep(0.8)
        self._set_status("Loading GPU libraries…", 1)
        self._set_progress(0.12)
        time.sleep(0.6)

        # Start the actual server process
        self._set_status("Starting server process…", 2)
        self._set_progress(0.18)

        try:
            python_bin = str(VENV / "bin" / "python")
            if not Path(python_bin).exists():
                python_bin = sys.executable

            self._server_proc = subprocess.Popen(
                [python_bin, "scripts/demo/demo_server.py"],
                cwd=str(ROOT),
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            GLib.idle_add(self._show_error, f"Failed to start server:\n{e}")
            return

        # Step 3-6: parse server stdout for progress signals
        step_map = {
            "Loading Vision Pipeline":          (3, 0.25),
            "Warming up pipeline":              (3, 0.35),
            "Pipeline warm-up complete":        (4, 0.50),
            "Initializing Gemini":              (5, 0.62),
            "Initializing Auditory":            (5, 0.70),
            "Pipeline Service Started":         (6, 0.80),
            "Running on http":                  (6, 0.88),
            "Application startup complete":     (7, 0.90),
            "Uvicorn running":                  (7, 0.90),
        }

        def _read_output():
            for line in self._server_proc.stdout:
                if self._cancelled:
                    break
                line = line.strip()
                if not line:
                    continue
                for keyword, (step, frac) in step_map.items():
                    if keyword.lower() in line.lower():
                        self._set_status(line[:80] if len(line) <= 80 else line[:77] + "…", step)
                        self._set_progress(frac)
                        break

        reader = threading.Thread(target=_read_output, daemon=True)
        reader.start()

        # Step 7: poll the server health endpoint
        self._set_status("Waiting for server to be ready…", 7)
        deadline = time.time() + 180  # 3-minute timeout
        last_frac = 0.88
        while time.time() < deadline and not self._cancelled:
            # Detect early server crash
            if self._server_proc.poll() is not None:
                GLib.idle_add(self._show_error,
                    "Server process exited unexpectedly.\n"
                    "Check that all Python dependencies are installed\n"
                    "and the venv is properly set up.")
                return
            try:
                urllib.request.urlopen(HEALTH_URL, timeout=2)
                break  # Server responded — we're ready!
            except Exception:
                last_frac = min(last_frac + 0.004, 0.98)
                self._set_progress(last_frac)
                time.sleep(1.5)
        else:
            if not self._cancelled:
                GLib.idle_add(self._show_error, "Server did not respond within 3 minutes.")
            return

        if self._cancelled:
            return

        # Done — signal ready
        GLib.idle_add(self._on_server_ready)

    def _on_server_ready(self):
        # Turn dot green
        self._dot.override_background_color(
            Gtk.StateFlags.NORMAL, Gdk.RGBA(0x34/255, 0xd3/255, 0x99/255, 1.0))
        self._status_lbl.set_text("✓ Vital Guardian is ready")
        self._step_lbl.set_text("Opening dashboard in your browser…")
        self._pbar.set_fraction(1.0)
        self._pct_lbl.set_text("100%")
        self._cancel_btn.set_label("Close Launcher")

        # Give the user a beat to see 100%, then open browser
        GLib.timeout_add(900, self._open_browser)

    def _open_browser(self):
        webbrowser.open(SERVER_URL)
        # After browser opens, transition the launcher to a minimal "running" state
        GLib.timeout_add(800, self._show_running_state)
        return False

    def _show_running_state(self):
        self._status_lbl.set_text("Server running — dashboard open in browser")
        self._step_lbl.set_text("You can close this window; the server will keep running.")
        return False

    def _on_cancel(self, btn):
        self._cancelled = True
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()
        Gtk.main_quit()

    def _on_destroy(self, *_):
        self._cancelled = True
        # Don't kill server — keep it alive so browser keeps working
        Gtk.main_quit()

    def _show_error(self, msg):
        dlg = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.CLOSE,
            text="Launch Failed",
        )
        dlg.format_secondary_text(msg)
        dlg.run()
        dlg.destroy()
        Gtk.main_quit()


# ── .desktop entry (auto-created) ─────────────────────────────────────────────
def install_desktop_entry():
    """Create ~/.local/share/applications/vital-guardian.desktop if not present."""
    desktop_dir  = Path.home() / ".local" / "share" / "applications"
    desktop_file = desktop_dir / "vital-guardian.desktop"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    icon_abs = str(ICON_PATH.resolve())
    launcher_abs = str(Path(__file__).resolve())
    content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Vital Guardian
GenericName=ICU Monitoring System
Comment=AI-Powered Patient Monitoring Dashboard
Exec=python3 {launcher_abs}
Icon={icon_abs}
Terminal=false
StartupNotify=true
StartupWMClass=Vital Guardian
Categories=Science;MedicalSoftware;
Keywords=ICU;Monitoring;AI;Medical;
"""
    desktop_file.write_text(content)
    desktop_file.chmod(0o755)
    # Also update GNOME desktop DB
    subprocess.Popen(["update-desktop-database", str(desktop_dir)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    install_desktop_entry()
    app = VitalGuardianLauncher()
    Gtk.main()
