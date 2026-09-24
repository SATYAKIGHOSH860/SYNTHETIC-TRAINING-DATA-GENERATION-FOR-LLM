"""Start / Stop control panel for Synthetic Training Data Studio (Windows).

Double-click the "Synthetic Data Studio" shortcut on the desktop, or
"Start Synthetic Data Studio.bat" in this folder. Press Start to open the
project in your browser (launching it first if needed); press Stop to shut it
down. No terminal or commands needed.

    venv\\Scripts\\pythonw.exe launcher.py                 # open the control panel
    venv\\Scripts\\python.exe launcher.py --make-shortcut  # (re)create the desktop shortcut
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import messagebox

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
PYTHONW = ROOT / "venv" / "Scripts" / "pythonw.exe"
PORT = 8501
URL = f"http://localhost:{PORT}"
PID_FILE = ROOT / "data" / "cache" / "dashboard.pid"
LOG_FILE = ROOT / "data" / "cache" / "dashboard.log"
BUSY_FILE = ROOT / "data" / "cache" / "dashboard.busy"  # present while a run or upload job is in progress (app.py)
ICON = ROOT / "assets" / "studio.ico"
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW: run helper processes without flashing a console
START_TIMEOUT = 180     # seconds; the first start loads PyTorch and the embedding model

BG, CARD, INK, INK_2, MUTED, BORDER = "#fcfcfb", "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, BLUE_DARK, GOOD, WARN, OFF = "#2a78d6", "#1c5cab", "#0ca30c", "#eda100", "#c3c2b7"
RED, RED_DARK = "#d03b3b", "#a82a2a"
FONT = "Segoe UI"


# ---------------------------------------------------------------------------
# Server control (no UI code here, so it can be used and tested on its own)
# ---------------------------------------------------------------------------
def is_running() -> bool:
    """True when the dashboard answers its health check."""
    try:
        with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=1.5) as response:
            return response.read().strip() == b"ok"
    except OSError:
        return False


def work_in_progress() -> bool:
    """True while the dashboard is generating a dataset (full run or upload job)."""
    return BUSY_FILE.is_file() and is_running()


def start_server() -> None:
    """Launch `streamlit run dashboard/app.py` in the background (no console window).

    It listens on localhost only: other devices on the network cannot reach it,
    and the dashboard keeps its full (non-public) controls.
    """
    if is_running():
        return
    if not PYTHON.is_file():
        raise FileNotFoundError(f"The project's Python environment is missing: {PYTHON}")
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), "-m", "streamlit", "run", "dashboard/app.py",
         "--server.headless", "true", "--server.address", "localhost", "--server.port", str(PORT)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
    )
    PID_FILE.write_text(str(process.pid), encoding="utf-8")


def _listening_pids(port: int) -> set[int]:
    """PIDs listening on the port (IPv4 and IPv6), so Stop also works for a server started elsewhere."""
    output = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                            creationflags=NO_WINDOW).stdout
    pids = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
            pids.add(int(parts[4]))
    return pids


def stop_server() -> None:
    """Stop the dashboard: the process we started plus anything still listening on its port."""
    pids = _listening_pids(PORT)
    if PID_FILE.is_file():
        try:
            pids.add(int(PID_FILE.read_text(encoding="utf-8").strip()))
        except ValueError:
            pass
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
    PID_FILE.unlink(missing_ok=True)
    BUSY_FILE.unlink(missing_ok=True)


def wait_until(condition, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Icon and desktop shortcut
# ---------------------------------------------------------------------------
def make_icon() -> Path:
    """Draw the app icon (blue tile with a Q&A mark) with Pillow, which Streamlit already installs."""
    from PIL import Image, ImageDraw, ImageFont

    ICON.parent.mkdir(parents=True, exist_ok=True)
    size = 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, size - 8, size - 8), radius=56, fill=BLUE)
    draw.rounded_rectangle((52, 60, 204, 164), radius=26, fill="#ffffff")
    draw.polygon([(84, 160), (84, 204), (124, 162)], fill="#ffffff")
    try:
        font = ImageFont.truetype("segoeuib.ttf", 76)
    except OSError:
        font = ImageFont.load_default()
    draw.text((128, 112), "Q&A", fill=BLUE_DARK, font=font, anchor="mm")
    image.save(ICON, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return ICON


def make_shortcut() -> Path:
    """Create 'Synthetic Data Studio.lnk' on the desktop, opening this control panel without a console."""
    icon = make_icon()
    desktop = subprocess.run(["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                             capture_output=True, text=True, creationflags=NO_WINDOW).stdout.strip()
    link = Path(desktop) / "Synthetic Data Studio.lnk"

    def ps(value: Path | str) -> str:  # quote for a PowerShell single-quoted string
        return "'" + str(value).replace("'", "''") + "'"

    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + ps(link) + "); "
        "$s.TargetPath = " + ps(PYTHONW) + "; "
        "$s.Arguments = " + ps('"' + str(ROOT / "launcher.py") + '"') + "; "
        "$s.WorkingDirectory = " + ps(ROOT) + "; "
        "$s.IconLocation = " + ps(icon) + "; "
        "$s.Description = 'Start or stop the Synthetic Training Data Studio dashboard'; "
        "$s.Save()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, creationflags=NO_WINDOW)
    return link


# ---------------------------------------------------------------------------
# Control panel window
# ---------------------------------------------------------------------------
class ControlPanel:
    """Two buttons and a live status light.

    Start opens the project in the browser, launching it first when it is not
    running. Stop shuts it down.

    Server checks and start/stop run in worker threads, which only set the
    attributes below; tkinter is not thread-safe, so the window is redrawn
    solely by poll() on Tk's own loop. That keeps it responsive.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.running = is_running()
        self.busy: str | None = None      # "starting" / "stopping" while an action is in progress
        self.message = ""
        self.checking = False             # a server check is in flight
        self.last_check = time.time()
        self.epoch = 0                    # bumped by Start/Stop so an older check cannot overwrite the result

        root.title("Synthetic Training Data Studio")
        root.configure(bg=BG)
        root.resizable(False, False)
        if ICON.is_file():
            try:
                root.iconbitmap(str(ICON))
            except tk.TclError:
                pass

        card = tk.Frame(root, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        card.pack(padx=18, pady=18, fill="both")
        tk.Label(card, text="Synthetic Training Data Studio", font=(FONT, 15, "bold"), fg=INK, bg=CARD
                 ).pack(anchor="w", padx=20, pady=(18, 0))
        tk.Label(card, text="WHO guideline PDFs → validated Q&A dataset", font=(FONT, 10), fg=INK_2, bg=CARD
                 ).pack(anchor="w", padx=20)

        status_row = tk.Frame(card, bg=CARD)
        status_row.pack(anchor="w", padx=20, pady=(16, 2))
        self.dot = tk.Canvas(status_row, width=14, height=14, bg=CARD, highlightthickness=0)
        self.dot_id = self.dot.create_oval(2, 2, 12, 12, fill=OFF, outline="")
        self.dot.pack(side="left")
        self.status = tk.Label(status_row, text="", font=(FONT, 11, "bold"), fg=INK, bg=CARD)
        self.status.pack(side="left", padx=(8, 0))
        # Fixed-size box for the status text, so the window keeps one size in every state.
        wrap = int(root.winfo_fpixels("4.2i"))
        holder = tk.Frame(card, bg=CARD, width=wrap, height=2 * tkfont.Font(family=FONT, size=9).metrics("linespace"))
        holder.pack(anchor="w", padx=20)
        holder.pack_propagate(False)
        self.detail = tk.Label(holder, text="", font=(FONT, 9), fg=MUTED, bg=CARD, justify="left", wraplength=wrap)
        self.detail.pack(anchor="nw")

        buttons = tk.Frame(card, bg=CARD)
        buttons.pack(fill="x", padx=20, pady=(16, 20))
        self.start_btn = self._button(buttons, "▶  Start", self.on_start, BLUE, BLUE_DARK)
        self.stop_btn = self._button(buttons, "■  Stop", self.on_stop, RED, RED_DARK)
        for i, button in enumerate((self.start_btn, self.stop_btn)):
            button.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 10, 0))
            buttons.columnconfigure(i, weight=1, uniform="buttons")

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.render()
        self.poll()

    @staticmethod
    def _button(parent, text, command, bg, active):
        return tk.Button(parent, text=text, command=command, font=(FONT, 12, "bold"), bg=bg, fg="#ffffff",
                         activebackground=active, activeforeground="#ffffff", disabledforeground="#b5b3ac",
                         relief="flat", bd=0, padx=18, pady=12, cursor="hand2")

    # -- state -----------------------------------------------------------------
    def poll(self) -> None:
        """Redraw twice a second; re-check the server every 2 seconds in a worker thread."""
        def check(epoch: int) -> None:
            running = is_running()
            if epoch == self.epoch:
                self.running = running
            self.checking = False
        if not self.busy and not self.checking and time.time() - self.last_check >= 2:
            self.checking, self.last_check = True, time.time()
            threading.Thread(target=check, args=(self.epoch,), daemon=True).start()
        self.render()
        self.root.after(500, self.poll)

    def render(self) -> None:
        if self.busy == "starting":
            colour, text, detail = WARN, "Starting…", self.message or (
                "Loading the dashboard. The first start takes up to a minute while PyTorch loads.")
        elif self.busy == "stopping":
            colour, text, detail = WARN, "Stopping…", ""
        elif self.running:
            colour, text, detail = GOOD, "Running", self.message or (
                f"Open at {URL}. Start opens it in your browser; Stop shuts it down.")
        else:
            colour, text, detail = OFF, "Stopped", self.message or "Press Start to open the project in your browser."
        self.dot.itemconfigure(self.dot_id, fill=colour)
        self.status.configure(text=text)
        self.detail.configure(text=detail)
        idle = self.busy is None
        # (button, enabled, background when enabled); disabled buttons turn flat grey so they read as inactive.
        for button, enabled, colour in ((self.start_btn, idle, BLUE),
                                        (self.stop_btn, idle and self.running, RED)):
            button.configure(state="normal" if enabled else "disabled", bg=colour if enabled else "#eeede8",
                             cursor="hand2" if enabled else "arrow")

    # -- actions -----------------------------------------------------------------
    def on_start(self) -> None:
        """Open the project in the browser, launching it first when it is not running."""
        if is_running():
            webbrowser.open(URL)
            return
        self.busy, self.message = "starting", ""
        self.epoch += 1
        self.render()

        def work() -> None:
            try:
                start_server()
                ok = wait_until(is_running, START_TIMEOUT)
                self.message = "" if ok else f"The dashboard did not start. See {LOG_FILE} for details."
                if ok:
                    webbrowser.open(URL)
            except Exception as exc:  # show the reason in the window instead of failing silently
                self.message = f"Could not start: {exc}"
            self.running = is_running()
            self.busy = None

        threading.Thread(target=work, daemon=True).start()

    def on_stop(self) -> None:
        """Shut the project down; asks first only when that would interrupt a dataset being generated."""
        if work_in_progress() and not messagebox.askyesno(
                "Stop while generating?",
                "A dataset is being generated right now.\n\nStop anyway? Work finished up to its last "
                "checkpoint is kept, and Generate dataset in the dashboard continues it.",
                parent=self.root):
            return
        self.busy, self.message = "stopping", ""
        self.epoch += 1
        self.render()

        def work() -> None:
            stop_server()
            stopped = wait_until(lambda: not is_running(), 20)
            self.message = "" if stopped else "The dashboard did not stop. Press Stop again."
            self.running = is_running()
            self.busy = None

        threading.Thread(target=work, daemon=True).start()

    def on_close(self) -> None:
        if self.running and not self.busy:
            answer = messagebox.askyesnocancel(
                "Close the control panel",
                "The dashboard is still running.\n\nYes: stop it and close.\nNo: keep it running and close.",
                parent=self.root)
            if answer is None:
                return
            if answer:
                stop_server()
        self.root.destroy()


def enable_sharp_rendering() -> None:
    """Declare DPI awareness so Windows does not stretch (and blur) the window on scaled displays."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def main() -> None:
    if "--make-shortcut" in sys.argv:
        print(f"Created {make_shortcut()}")
        return
    if not ICON.is_file():
        try:
            make_icon()
        except Exception:
            pass
    enable_sharp_rendering()
    root = tk.Tk()
    ControlPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
