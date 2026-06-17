"""
CAUI v0.4 — Correlation Analysis for TEM Images
=================================================
Workflow:
  Step 0 → File selection, calibration, preview
  Step 1 → Material ROI + optional Background ROI
  Step 2 → Material AC map + mask
  Step 3 → Background AC map + mask  (only when BG ROI selected)
  Step 4 → Normalization: a*C_bg(r) + constant → C_result(r)
  Step 5 → Final result display & save .cr

v0.4 changes:
  - Independent masking for material and background AC maps
  - Each mask step has its own auto-scaled colormap
  - Common r-range intersection for subtraction
  - Logarithmic constant slider for high-intensity TEM data
"""

import sys
import os
import numpy as np

# ── GUI ──────────────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFormLayout, QStackedWidget, QStatusBar, QMessageBox,
    QComboBox, QRadioButton, QButtonGroup, QSlider, QSpinBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSizePolicy

# ── Matplotlib ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import RectangleSelector


# ══════════════════════════════════════════════════════════════════════
# FILE LOADERS
# ══════════════════════════════════════════════════════════════════════

def load_dm_file(filepath: str) -> dict:
    import hyperspy.api as hs
    sig = hs.load(filepath)
    img = sig.data
    if img.ndim == 3:
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    px_size = 1.0
    ax_mgr = getattr(sig, "axes_manager", None)
    if ax_mgr is not None and len(ax_mgr.signal_axes) >= 2:
        scale_x = ax_mgr.signal_axes[0].scale
        unit_x = ax_mgr.signal_axes[0].units
        px_size = float(scale_x)
        if "nm" in str(unit_x).lower():
            pass
        elif "m" in str(unit_x).lower() or "um" in str(unit_x).lower():
            px_size *= 1000.0
        elif "m" in str(unit_x).lower():
            px_size *= 1e9

    return {"image": img.astype(np.float64), "px_size_nm": px_size,
            "metadata": {"filepath": filepath, "shape": img.shape,
                          "dtype": str(img.dtype), "px_size_nm": px_size,
                          "px_size_source": "metadata"}}


def load_tiff_file(filepath: str) -> dict:
    import tifffile
    img = tifffile.imread(filepath)
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= 4 else img[0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    return {"image": img.astype(np.float64), "px_size_nm": 0.0,
            "metadata": {"filepath": filepath, "shape": img.shape,
                          "dtype": str(img.dtype), "px_size_nm": 0.0,
                          "px_size_source": "user"}}


def load_image(filepath: str, user_px_size: float = 0.0) -> dict:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".dm3", ".dm4"):
        r = load_dm_file(filepath)
        if user_px_size > 0:
            r["px_size_nm"] = user_px_size
        return r
    elif ext in (".tif", ".tiff"):
        r = load_tiff_file(filepath)
        if user_px_size > 0:
            r["px_size_nm"] = user_px_size
        return r
    else:
        raise ValueError(f"Unsupported extension: {ext}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def compute_autocorr_2d(image: np.ndarray) -> np.ndarray:
    F = np.fft.fft2(image)
    ac = np.fft.ifft2(np.abs(F) ** 2).real
    return np.fft.fftshift(ac)


def radial_average(data: np.ndarray, center=None, mask_radius: float = 0.0,
                   method: str = "splitpixel") -> tuple:
    if center is None:
        center = (data.shape[1] / 2.0, data.shape[0] / 2.0)
    cx, cy = center
    y_idx, x_idx = np.indices(data.shape)
    r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    max_r = int(np.ceil(np.sqrt(cy**2 + cx**2)))

    valid = np.ones_like(data, dtype=bool)
    if mask_radius > 0:
        valid[r <= mask_radius] = False

    radial_sum = np.zeros(max_r + 1, dtype=np.float64)
    radial_count = np.zeros(max_r + 1, dtype=np.float64)

    if method == "splitpixel":
        r_low = np.floor(r).astype(int); r_high = r_low + 1; frac = r - r_low
        vl = (r_low <= max_r) & valid
        np.add.at(radial_sum, r_low[vl], (1.0 - frac[vl]) * data[vl])
        np.add.at(radial_count, r_low[vl], 1.0 - frac[vl])
        vh = (r_high <= max_r) & valid
        np.add.at(radial_sum, r_high[vh], frac[vh] * data[vh])
        np.add.at(radial_count, r_high[vh], frac[vh])
    else:
        r_int = np.round(r).astype(int)
        vm = (r_int <= max_r) & valid
        np.add.at(radial_sum, r_int[vm], data[vm])
        np.add.at(radial_count, r_int[vm], 1.0)

    radial_mean = np.divide(radial_sum, radial_count,
                            where=radial_count > 0,
                            out=np.full_like(radial_sum, np.nan))
    return np.arange(0, max_r + 1, dtype=float), radial_mean


# ══════════════════════════════════════════════════════════════════════
# .CR PARSER
# ══════════════════════════════════════════════════════════════════════

def _parse_rect_str(s: str) -> tuple:
    nums = {}
    for part in s.split():
        if "=" in part:
            k, _, v = part.partition("=")
            if k in ("x0", "x1", "y0", "y1"):
                nums[k] = int(float(v))
    return (nums.get("x0", 0), nums.get("x1", 0),
            nums.get("y0", 0), nums.get("y1", 0))


def _parse_mask_str(s: str) -> tuple:
    """Parse 'center=(cx, cy) px  radius=r px' → (cx, cy, r)."""
    try:
        s = s.replace("(", "").replace(")", "")
        c_str = s.split("radius=")[0].replace("center=", "").strip()
        r_str = s.split("radius=")[1].split("px")[0].strip()
        return (float(c_str.split(",")[0].strip()),
                float(c_str.split(",")[1].strip()),
                float(r_str))
    except Exception:
        return (0.0, 0.0, 0.0)


def parse_cr_header(filepath: str) -> dict:
    header = {}
    data_rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                inner = s.lstrip("#").strip()
                if ":" in inner:
                    k, _, v = inner.partition(":")
                    header[k.strip()] = v.strip()
            else:
                parts = s.split()
                if len(parts) >= 2:
                    data_rows.append([float(p) for p in parts])

    if not data_rows:
        raise ValueError(f"No data columns in {filepath}")

    raw = {"source_filepath": header.get("filepath", "")}
    try:
        raw["px_size_nm"] = float(header.get("pixel size", "1.0").split()[0])
    except ValueError:
        raw["px_size_nm"] = 1.0
    raw["dark_field"] = "dark" in header.get("contrast", "").lower()

    roi_str = header.get("material ROI", header.get("ROI", ""))
    raw["roi_rect"] = _parse_rect_str(roi_str) if roi_str else None

    bg_str = header.get("background ROI", "")
    raw["bg_rect"] = _parse_rect_str(bg_str) if bg_str else None
    raw["bg_enabled"] = raw["bg_rect"] is not None and sum(raw["bg_rect"]) > 0

    mat_mask_str = header.get("mask material", header.get("mask center", ""))
    cx_m, cy_m, r_m = _parse_mask_str(mat_mask_str)
    raw["mask_cx"], raw["mask_cy"], raw["mask_r"] = cx_m, cy_m, r_m

    bg_mask_str = header.get("mask background", "")
    cx_b, cy_b, r_b = _parse_mask_str(bg_mask_str)
    raw["mask_cx_bg"] = cx_b if bg_mask_str else cx_m
    raw["mask_cy_bg"] = cy_b if bg_mask_str else cy_m
    raw["mask_r_bg"] = r_b if bg_mask_str else r_m

    raw["method"] = header.get("radial integration method", "splitpixel")
    try:
        raw["norm_a"] = float(
            header.get("normalization a",
                       header.get("normalization", "a=1.0")).split("a=")[-1].split()[0])
    except (ValueError, IndexError):
        raw["norm_a"] = 1.0

    return {"header": header, "data": data_rows, "raw_meta": raw}


# ══════════════════════════════════════════════════════════════════════
# CANVAS + MASK EDITOR
# ══════════════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.tight_layout()


class CircleMaskEditor:
    def __init__(self, ax, center, radius, on_changed=None):
        self.ax = ax
        self.on_changed = on_changed
        self._dragging = False
        self._resizing = False
        self._press_xy = None
        self._press_center = None
        self._press_radius = None
        self.patch = Circle(center, radius=radius, fill=False,
                            edgecolor="red", linewidth=2, linestyle="--",
                            alpha=0.9, picker=5)
        self.ax.add_patch(self.patch)
        self._cid_press = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self._on_press)
        self._cid_motion = self.ax.figure.canvas.mpl_connect(
            "motion_notify_event", self._on_motion)
        self._cid_release = self.ax.figure.canvas.mpl_connect(
            "button_release_event", self._on_release)
        self._cid_scroll = self.ax.figure.canvas.mpl_connect(
            "scroll_event", self._on_scroll)

    @property
    def center(self):
        return self.patch.center
    @property
    def radius(self):
        return self.patch.radius

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        contains, _ = self.patch.contains(event)
        if not contains:
            return
        self._dragging = True
        self._press_xy = (event.xdata, event.ydata)
        self._press_center = self.patch.center
        self._press_radius = self.patch.radius
        cx, cy = self.patch.center
        self._resizing = np.hypot(event.xdata - cx, event.ydata - cy) > 0.6 * self.patch.radius

    def _on_motion(self, event):
        if not self._dragging or event.inaxes != self.ax:
            return
        dx = event.xdata - self._press_xy[0]
        dy = event.ydata - self._press_xy[1]
        if self._resizing:
            self.patch.set_radius(max(1.0, self._press_radius + 0.5 * (dx + dy)))
        else:
            self.patch.set_center((self._press_center[0] + dx,
                                    self._press_center[1] + dy))
        self.ax.figure.canvas.draw_idle()
        if self.on_changed:
            self.on_changed()

    def _on_release(self, event):
        self._dragging = False

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        contains, _ = self.patch.contains(event)
        if contains:
            d = 1.1 if event.button == "up" else 0.9
            self.patch.set_radius(max(1.0, self.patch.radius * d))
            self.ax.figure.canvas.draw_idle()
            if self.on_changed:
                self.on_changed()

    def disconnect(self):
        for cid in [self._cid_press, self._cid_motion, self._cid_release, self._cid_scroll]:
            self.ax.figure.canvas.mpl_disconnect(cid)


# ══════════════════════════════════════════════════════════════════════
# STEP 0 — FILE
# ══════════════════════════════════════════════════════════════════════

class Step0Widget(QWidget):
    file_loaded = Signal(dict)
    cr_loaded = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        fr = QHBoxLayout()
        self._btn_browse = QPushButton("Browse TEM File...")
        self._btn_load_cr = QPushButton("Load .cr File")
        self._lbl_file = QLabel("No file selected")
        self._lbl_file.setWordWrap(True)
        fr.addWidget(self._btn_browse); fr.addWidget(self._btn_load_cr)
        fr.addWidget(self._lbl_file, 1)
        layout.addLayout(fr)

        grp = QGroupBox("Calibration && Image Type")
        form = QFormLayout(grp)
        self._lbl_format = QLabel("—")
        form.addRow("Format:", self._lbl_format)
        self._spin_px = QDoubleSpinBox()
        self._spin_px.setRange(0, 1e6); self._spin_px.setDecimals(6)
        self._spin_px.setSuffix(" nm/px")
        form.addRow("Pixel size:", self._spin_px)
        self._chk_override = QCheckBox("Override metadata value")
        form.addRow("", self._chk_override)

        self._btn_group = QButtonGroup(self)
        self._rb_bright = QRadioButton("Bright field")
        self._rb_dark = QRadioButton("Dark field")
        self._btn_group.addButton(self._rb_bright, 0)
        self._btn_group.addButton(self._rb_dark, 1)
        self._rb_bright.setChecked(True)
        form.addRow(self._rb_bright); form.addRow(self._rb_dark)

        self._btn_load = QPushButton("Load && Preview")
        form.addRow("", self._btn_load)
        layout.addWidget(grp)

        self._canvas = MplCanvas(self, 5, 4)
        self._toolbar = NavToolbar(self._canvas, self)
        layout.addWidget(self._toolbar); layout.addWidget(self._canvas)

        self._btn_browse.clicked.connect(self._on_browse)
        self._btn_load_cr.clicked.connect(self._on_load_cr)
        self._btn_load.clicked.connect(self._on_load)

    def _on_browse(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select TEM file",
            os.path.join(os.path.dirname(__file__), "databank"),
            "TEM files (*.dm3 *.dm4 *.tif *.tiff);;All files (*.*)")
        if fp:
            self._lbl_file.setText(fp)
            ext = os.path.splitext(fp)[1].lower()
            self._lbl_format.setText(ext.upper())
            if ext in (".dm3", ".dm4"):
                self._spin_px.setValue(0); self._spin_px.setEnabled(False)
                self._chk_override.setChecked(False); self._chk_override.setEnabled(True)
            else:
                self._spin_px.setValue(0); self._spin_px.setEnabled(True)
                self._chk_override.setChecked(True); self._chk_override.setEnabled(False)

    def _on_load_cr(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open correlation result",
            os.path.join(os.path.dirname(__file__), "databank"),
            "CR files (*.cr);;All files (*.*)")
        if not fp:
            return
        try:
            parsed = parse_cr_header(fp)
            raw = parsed["raw_meta"]
            dr = parsed["data"]
            tem_data = None
            src = raw["source_filepath"]
            if not src or not os.path.isfile(src):
                alt, _ = QFileDialog.getOpenFileName(
                    self, f"Locate source TEM file for {os.path.basename(fp)}",
                    os.path.dirname(fp) if os.path.isdir(os.path.dirname(fp))
                    else os.path.dirname(__file__),
                    "TEM files (*.dm3 *.dm4 *.tif *.tiff);;All files (*.*)")
                if alt:
                    src = alt
            if src and os.path.isfile(src):
                tem_data = load_image(src, user_px_size=raw["px_size_nm"])
                tem_data["dark_field"] = raw["dark_field"]

            ra = np.array([r[0] for r in dr])
            ca = np.array([r[-1] for r in dr])
            self._canvas.ax.remove()
            self._canvas.ax = self._canvas.fig.add_subplot(111)
            self._canvas.ax.plot(ra, ca, "b-", linewidth=1.2)
            self._canvas.ax.set_xlabel("r (A)"); self._canvas.ax.set_ylabel("C(r)")
            self._canvas.ax.set_title(f"CR History: {os.path.basename(fp)}")
            self._canvas.ax.grid(True, alpha=0.3)
            self._canvas.fig.tight_layout(); self._canvas.draw()
            self._lbl_file.setText(fp); self._lbl_format.setText("CR (history)")

            self.cr_loaded.emit({
                "cr_filepath": fp, "tem_data": tem_data,
                "roi_rect": raw["roi_rect"], "bg_rect": raw["bg_rect"],
                "bg_enabled": raw.get("bg_enabled", False),
                "mask_cx": raw["mask_cx"], "mask_cy": raw["mask_cy"],
                "mask_r": raw["mask_r"],
                "mask_cx_bg": raw.get("mask_cx_bg", raw["mask_cx"]),
                "mask_cy_bg": raw.get("mask_cy_bg", raw["mask_cy"]),
                "mask_r_bg": raw.get("mask_r_bg", raw["mask_r"]),
                "method": raw["method"],
                "norm_a": raw.get("norm_a", 1.0),
                "header": parsed["header"],
            })
        except Exception as e:
            QMessageBox.critical(self, "Load .cr error", str(e))

    def _on_load(self):
        fp = self._lbl_file.text()
        if not fp or not os.path.isfile(fp):
            QMessageBox.warning(self, "Error", "Please select a valid file first.")
            return
        try:
            px_val = self._spin_px.value() if (
                self._chk_override.isChecked()
                or os.path.splitext(fp)[1].lower() in (".tif", ".tiff")) else 0.0
            if os.path.splitext(fp)[1].lower() in (".tif", ".tiff") and px_val <= 0:
                QMessageBox.warning(self, "Missing calibration",
                                    "TIFF files require pixel size (nm/px).")
                return
            self._data = load_image(fp, user_px_size=px_val)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        img = self._data["image"].copy()
        if self._rb_dark.isChecked():
            img = -img
        self._canvas.ax.clear()
        self._canvas.ax.imshow(img, cmap="gray", origin="upper")
        self._canvas.ax.set_title(
            f"Preview — {os.path.basename(fp)}  |  "
            f"{self._data['metadata']['shape']}  |  "
            f"{self._data['px_size_nm']:.4f} nm/px")
        self._canvas.fig.tight_layout(); self._canvas.draw()
        self._data["dark_field"] = self._rb_dark.isChecked()
        self.file_loaded.emit(self._data)

    @property
    def data(self):
        return self._data


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — ROI + BG
# ══════════════════════════════════════════════════════════════════════

class Step1Widget(QWidget):
    roi_confirmed = Signal(np.ndarray, np.ndarray, tuple, tuple, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None; self._roi = None; self._bg = None
        self._full_img = None; self._sel_roi = None; self._sel_bg = None
        self._mode = "roi"
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel(
            "1. (Optional) Switch to 'Background' mode, drag a rectangle "
            "on a featureless region.\n"
            "2. Switch to 'Material' mode, drag a rectangle on the ROI.\n"
            "3. Click 'Confirm' when ready.")
        lbl.setWordWrap(True); layout.addWidget(lbl)

        mr = QHBoxLayout()
        self._btn_mode_bg = QPushButton("Background"); self._btn_mode_bg.setCheckable(True)
        self._btn_mode_roi = QPushButton("Material"); self._btn_mode_roi.setCheckable(True)
        self._btn_mode_roi.setChecked(True)
        mr.addWidget(QLabel("Selection mode:")); mr.addWidget(self._btn_mode_roi)
        mr.addWidget(self._btn_mode_bg); mr.addStretch()
        self._lbl_bg_status = QLabel("Background: not set")
        mr.addWidget(self._lbl_bg_status)
        layout.addLayout(mr)

        self._canvas = MplCanvas(self, 6, 5)
        self._toolbar = NavToolbar(self._canvas, self)
        layout.addWidget(self._toolbar); layout.addWidget(self._canvas)

        br = QHBoxLayout()
        self._btn_confirm = QPushButton("Confirm"); self._btn_confirm.setEnabled(False)
        br.addStretch(); br.addWidget(self._btn_confirm); br.addStretch()
        layout.addLayout(br)

        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_mode_bg.clicked.connect(lambda: self._switch_mode("bg"))
        self._btn_mode_roi.clicked.connect(lambda: self._switch_mode("roi"))

    def _switch_mode(self, m):
        self._mode = m
        self._btn_mode_bg.setChecked(m == "bg")
        self._btn_mode_roi.setChecked(m == "roi")
        # entering a mode → its static patch has served its purpose.
        # remove it now so it doesn't linger when the user draws a new rect.
        if m == "roi" and getattr(self, "_static_roi_patch", None) is not None:
            self._static_roi_patch.remove()
            self._static_roi_patch = None
        elif m == "bg" and getattr(self, "_static_bg_patch", None) is not None:
            self._static_bg_patch.remove()
            self._static_bg_patch = None
        self._activate_selectors()

    def _activate_selectors(self):
        if self._sel_roi: self._sel_roi.set_active(self._mode == "roi")
        if self._sel_bg: self._sel_bg.set_active(self._mode == "bg")
        self._canvas.draw()

    def set_image(self, data):
        self._data = data
        img = data["image"].copy()
        if data.get("dark_field", False): img = -img
        self._full_img = img; self._roi = None; self._bg = None
        self._static_roi_patch = None; self._static_bg_patch = None
        self._btn_confirm.setEnabled(False)
        self._lbl_bg_status.setText("Background: not set")
        self._canvas.ax.clear()
        self._canvas.ax.imshow(img, cmap="gray", origin="upper")
        self._canvas.ax.set_title(f"Select ROI — {os.path.basename(data['metadata']['filepath'])}")
        self._canvas.fig.tight_layout(); self._canvas.draw()
        self._recreate_selectors(); self._activate_selectors()

    def _recreate_selectors(self):
        for attr in ("_sel_roi", "_sel_bg"):
            old = getattr(self, attr, None)
            if old:
                try: old.disconnect_events()
                except Exception: pass
        self._sel_roi = RectangleSelector(
            self._canvas.ax, self._on_roi_select, useblit=True,
            props=dict(facecolor=(0.4, 0.7, 1.0, 0.25),
                       edgecolor=(0.2, 0.5, 1.0, 0.9), linewidth=2, linestyle="-"),
            interactive=True)
        self._sel_bg = RectangleSelector(
            self._canvas.ax, self._on_bg_select, useblit=True,
            props=dict(facecolor=(1.0, 0.6, 0.2, 0.25),
                       edgecolor=(1.0, 0.4, 0.1, 0.9), linewidth=2, linestyle="--"),
            interactive=True)

    def _on_roi_select(self, eclick, erelease):
        try:
            x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
            x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        except (TypeError, ValueError): return
        xmin = max(0, min(x1, x2)); xmax = min(self._full_img.shape[1], max(x1, x2) + 1)
        ymin = max(0, min(y1, y2)); ymax = min(self._full_img.shape[0], max(y1, y2) + 1)
        if xmax - xmin < 5 or ymax - ymin < 5: return
        # remove static patch from .cr restore / back-nav if still present
        if getattr(self, "_static_roi_patch", None) is not None:
            self._static_roi_patch.remove()
            self._static_roi_patch = None
        self._roi = (xmin, xmax, ymin, ymax)
        self._btn_confirm.setEnabled(True)

    def _on_bg_select(self, eclick, erelease):
        try:
            x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
            x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        except (TypeError, ValueError): return
        xmin = max(0, min(x1, x2)); xmax = min(self._full_img.shape[1], max(x1, x2) + 1)
        ymin = max(0, min(y1, y2)); ymax = min(self._full_img.shape[0], max(y1, y2) + 1)
        if xmax - xmin < 5 or ymax - ymin < 5: return
        # remove static patch from .cr restore / back-nav if still present
        if getattr(self, "_static_bg_patch", None) is not None:
            self._static_bg_patch.remove()
            self._static_bg_patch = None
        self._bg = (xmin, xmax, ymin, ymax)
        bg_val = float(np.mean(self._full_img[ymin:ymax, xmin:xmax]))
        self._lbl_bg_status.setText(
            f"Background: {bg_val:.2f} ({xmax - xmin}x{ymax - ymin} px)")

    def _on_confirm(self):
        if self._roi is None: return
        # safety: remove any lingering static patches before moving on
        for attr in ("_static_roi_patch", "_static_bg_patch"):
            p = getattr(self, attr, None)
            if p is not None:
                p.remove()
                setattr(self, attr, None)
        xmin, xmax, ymin, ymax = self._roi
        mat_roi = self._full_img[ymin:ymax, xmin:xmax].copy()
        bg_enabled = self._bg is not None and sum(self._bg) > 0
        bg_roi = None; bg_rect = (0, 0, 0, 0)
        if bg_enabled:
            bx0, bx1, by0, by1 = self._bg
            bg_roi = self._full_img[by0:by1, bx0:bx1].copy()
            bg_rect = self._bg
        self.roi_confirmed.emit(mat_roi, bg_roi, self._roi, bg_rect, bg_enabled)

    @property
    def roi(self):
        return self._roi

    @property
    def roi_arrays(self):
        if self._roi is None: return None, None, False
        xmin, xmax, ymin, ymax = self._roi
        mat_roi = self._full_img[ymin:ymax, xmin:xmax].copy()
        bg_enabled = self._bg is not None and sum(self._bg) > 0
        bg_roi = None
        if bg_enabled:
            bx0, bx1, by0, by1 = self._bg
            bg_roi = self._full_img[by0:by1, bx0:bx1].copy()
        return mat_roi, bg_roi, bg_enabled

    def reload_with_rects(self, data, roi_rect, bg_rect, bg_enabled):
        self._data = data
        img = data["image"].copy()
        if data.get("dark_field", False): img = -img
        self._full_img = img
        self._roi = roi_rect
        self._bg = bg_rect if (bg_rect and sum(bg_rect) > 0) else None
        self._btn_confirm.setEnabled(True)
        self._canvas.ax.clear()
        self._canvas.ax.imshow(img, cmap="gray", origin="upper")
        self._canvas.ax.set_title(
            f"Select ROI — {os.path.basename(data['metadata']['filepath'])}")
        self._canvas.fig.tight_layout(); self._canvas.draw()
        self._recreate_selectors()
        if self._bg:
            bx0, bx1, by0, by1 = self._bg
            self._lbl_bg_status.setText(
                f"Background: {float(np.mean(img[by0:by1, bx0:bx1])):.2f} "
                f"({bx1 - bx0}x{by1 - by0} px)")
        self._activate_selectors()
        if self._roi:
            rx0, rx1, ry0, ry1 = self._roi
            self._static_roi_patch = Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                facecolor=(0.4, 0.7, 1.0, 0.25), edgecolor=(0.2, 0.5, 1.0, 0.9),
                linewidth=2, linestyle="-")
            self._canvas.ax.add_patch(self._static_roi_patch)
        else:
            self._static_roi_patch = None
        if self._bg:
            bx0, bx1, by0, by1 = self._bg
            self._static_bg_patch = Rectangle(
                (bx0, by0), bx1 - bx0, by1 - by0,
                facecolor=(1.0, 0.6, 0.2, 0.25), edgecolor=(1.0, 0.4, 0.1, 0.9),
                linewidth=2, linestyle="--")
            self._canvas.ax.add_patch(self._static_bg_patch)
        else:
            self._static_bg_patch = None
        self._canvas.draw()


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — MATERIAL AC MAP + MASK
# ══════════════════════════════════════════════════════════════════════

class Step2Widget(QWidget):
    mat_mask_confirmed = Signal(np.ndarray, float, float, float, str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ac_map = None; self._roi_img = None
        self._mask_editor = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("Material AC Map — drag the red circle to mask the central spot.")
        lbl.setWordWrap(True); layout.addWidget(lbl)

        cr = QHBoxLayout()
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["viridis", "plasma", "inferno", "gray", "hot", "jet"])
        self._cmap_combo.setCurrentText("viridis")
        cr.addWidget(QLabel("Colormap:")); cr.addWidget(self._cmap_combo)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["splitpixel", "simple"])
        self._method_combo.setCurrentText("splitpixel")
        cr.addWidget(QLabel("Method:")); cr.addWidget(self._method_combo)

        self._chk_log = QCheckBox("Log scale"); cr.addWidget(self._chk_log)
        cr.addWidget(QLabel("vmin:"))
        self._spin_vmin = QDoubleSpinBox(); self._spin_vmin.setRange(0, 1e20)
        self._spin_vmin.setDecimals(2); cr.addWidget(self._spin_vmin)
        cr.addWidget(QLabel("vmax:"))
        self._spin_vmax = QDoubleSpinBox(); self._spin_vmax.setRange(0, 1e20)
        self._spin_vmax.setDecimals(2); cr.addWidget(self._spin_vmax)
        self._btn_refresh = QPushButton("Refresh"); cr.addWidget(self._btn_refresh)
        cr.addStretch(); layout.addLayout(cr)

        # canvas row: ROI preview | AC map with mask
        sr = QHBoxLayout()
        rc = QVBoxLayout()
        self._canvas_roi = MplCanvas(self, 4, 4)
        self._toolbar_roi = NavToolbar(self._canvas_roi, self)
        rc.addWidget(self._toolbar_roi); rc.addWidget(self._canvas_roi); sr.addLayout(rc)

        ac2 = QVBoxLayout()
        self._canvas_ac = MplCanvas(self, 4, 4)
        self._toolbar_ac = NavToolbar(self._canvas_ac, self)
        ac2.addWidget(self._toolbar_ac); ac2.addWidget(self._canvas_ac); sr.addLayout(ac2)
        layout.addLayout(sr)

        br = QHBoxLayout()
        self._btn_apply = QPushButton("Apply Material Mask"); self._btn_apply.setEnabled(False)
        br.addStretch(); br.addWidget(self._btn_apply); br.addStretch()
        layout.addLayout(br)

        self._btn_apply.clicked.connect(self._on_apply)
        self._cmap_combo.currentTextChanged.connect(self._redraw_ac)
        self._chk_log.toggled.connect(self._redraw_ac)
        self._btn_refresh.clicked.connect(self._redraw_ac)
        self._spin_vmin.editingFinished.connect(self._redraw_ac)
        self._spin_vmax.editingFinished.connect(self._redraw_ac)

    def set_roi(self, roi_img):
        self._roi_img = roi_img
        self._ac_map = compute_autocorr_2d(roi_img - np.mean(roi_img))
        cy, cx = self._ac_map.shape[0] / 2.0, self._ac_map.shape[1] / 2.0
        r0 = min(self._ac_map.shape) * 0.05
        vals = np.sort(self._ac_map.ravel())
        self._spin_vmin.setValue(float(self._ac_map.min()))
        self._spin_vmax.setValue(float(vals[int(len(vals) * 0.99)]))
        self._redraw_roi(); self._redraw_ac()
        self._setup_mask(cx, cy, r0)
        self._btn_apply.setEnabled(True)

    def _redraw_roi(self):
        if self._roi_img is None: return
        self._canvas_roi.ax.clear()
        self._canvas_roi.ax.imshow(self._roi_img, cmap="gray", origin="upper")
        self._canvas_roi.ax.set_title("Material ROI"); self._canvas_roi.fig.tight_layout()
        self._canvas_roi.draw()

    def _redraw_ac(self):
        if self._ac_map is None: return
        self._canvas_ac.ax.clear()
        d = self._ac_map.copy()
        vmin = self._spin_vmin.value(); vmax = self._spin_vmax.value()
        if vmax <= vmin: vmax, vmin = d.max(), d.min()
        if self._chk_log.isChecked():
            d = np.log1p(d - d.min())
            if vmax > 0: vmax = np.log1p(vmax - self._ac_map.min())
            if vmin >= 0: vmin = np.log1p(vmin - self._ac_map.min())
        self._canvas_ac.ax.imshow(d, cmap=self._cmap_combo.currentText(),
                                   origin="upper", vmin=vmin, vmax=vmax)
        self._canvas_ac.ax.set_title("Material AC Map"); self._canvas_ac.fig.tight_layout()
        self._canvas_ac.draw()

    def _setup_mask(self, cx, cy, r0):
        if self._mask_editor: self._mask_editor.disconnect()
        self._mask_editor = CircleMaskEditor(self._canvas_ac.ax, (cx, cy), r0)
        self._canvas_ac.draw()

    def _on_apply(self):
        if self._mask_editor is None: return
        cx, cy = self._mask_editor.center; r = self._mask_editor.radius
        self.mat_mask_confirmed.emit(self._ac_map, float(cx), float(cy),
                                      float(r), self._method_combo.currentText())

    @property
    def ac_map(self):
        return self._ac_map

    def reload_with_mask(self, roi_img, ac_map, cx, cy, r):
        self._roi_img = roi_img; self._ac_map = ac_map
        vals = np.sort(ac_map.ravel())
        self._spin_vmin.setValue(float(ac_map.min()))
        self._spin_vmax.setValue(float(vals[int(len(vals) * 0.99)]))
        self._redraw_roi(); self._redraw_ac(); self._setup_mask(cx, cy, r)
        self._btn_apply.setEnabled(True)


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — BACKGROUND AC MAP + MASK
# ══════════════════════════════════════════════════════════════════════

class Step3Widget(QWidget):
    bg_mask_confirmed = Signal(np.ndarray, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ac_map = None; self._roi_img = None
        self._mask_editor = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("Background AC Map — drag the red circle to mask the central spot.")
        lbl.setWordWrap(True); layout.addWidget(lbl)

        cr = QHBoxLayout()
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["viridis", "plasma", "inferno", "gray", "hot", "jet"])
        cr.addWidget(QLabel("Colormap:")); cr.addWidget(self._cmap_combo)
        self._chk_log = QCheckBox("Log scale"); cr.addWidget(self._chk_log)
        cr.addWidget(QLabel("vmin:"))
        self._spin_vmin = QDoubleSpinBox(); self._spin_vmin.setRange(0, 1e20)
        self._spin_vmin.setDecimals(2); cr.addWidget(self._spin_vmin)
        cr.addWidget(QLabel("vmax:"))
        self._spin_vmax = QDoubleSpinBox(); self._spin_vmax.setRange(0, 1e20)
        self._spin_vmax.setDecimals(2); cr.addWidget(self._spin_vmax)
        self._btn_refresh = QPushButton("Refresh"); cr.addWidget(self._btn_refresh)
        cr.addStretch(); layout.addLayout(cr)

        sr = QHBoxLayout()
        rc = QVBoxLayout()
        self._canvas_roi = MplCanvas(self, 4, 4)
        self._toolbar_roi = NavToolbar(self._canvas_roi, self)
        rc.addWidget(self._toolbar_roi); rc.addWidget(self._canvas_roi); sr.addLayout(rc)

        ac2 = QVBoxLayout()
        self._canvas_ac = MplCanvas(self, 4, 4)
        self._toolbar_ac = NavToolbar(self._canvas_ac, self)
        ac2.addWidget(self._toolbar_ac); ac2.addWidget(self._canvas_ac); sr.addLayout(ac2)
        layout.addLayout(sr)

        br = QHBoxLayout()
        self._btn_apply = QPushButton("Apply Background Mask"); self._btn_apply.setEnabled(False)
        br.addStretch(); br.addWidget(self._btn_apply); br.addStretch()
        layout.addLayout(br)

        self._btn_apply.clicked.connect(self._on_apply)
        self._cmap_combo.currentTextChanged.connect(self._redraw_ac)
        self._chk_log.toggled.connect(self._redraw_ac)
        self._btn_refresh.clicked.connect(self._redraw_ac)
        self._spin_vmin.editingFinished.connect(self._redraw_ac)
        self._spin_vmax.editingFinished.connect(self._redraw_ac)

    def set_roi(self, roi_img):
        self._roi_img = roi_img
        self._ac_map = compute_autocorr_2d(roi_img - np.mean(roi_img))
        cy, cx = self._ac_map.shape[0] / 2.0, self._ac_map.shape[1] / 2.0
        r0 = min(self._ac_map.shape) * 0.05
        vals = np.sort(self._ac_map.ravel())
        self._spin_vmin.setValue(float(self._ac_map.min()))
        self._spin_vmax.setValue(float(vals[int(len(vals) * 0.99)]))
        self._redraw_roi(); self._redraw_ac()
        self._setup_mask(cx, cy, r0)
        self._btn_apply.setEnabled(True)

    def _redraw_roi(self):
        if self._roi_img is None: return
        self._canvas_roi.ax.clear()
        self._canvas_roi.ax.imshow(self._roi_img, cmap="gray", origin="upper")
        self._canvas_roi.ax.set_title("Background ROI"); self._canvas_roi.fig.tight_layout()
        self._canvas_roi.draw()

    def _redraw_ac(self):
        if self._ac_map is None: return
        self._canvas_ac.ax.clear()
        d = self._ac_map.copy()
        vmin = self._spin_vmin.value(); vmax = self._spin_vmax.value()
        if vmax <= vmin: vmax, vmin = d.max(), d.min()
        if self._chk_log.isChecked():
            d = np.log1p(d - d.min())
            if vmax > 0: vmax = np.log1p(vmax - self._ac_map.min())
            if vmin >= 0: vmin = np.log1p(vmin - self._ac_map.min())
        self._canvas_ac.ax.imshow(d, cmap=self._cmap_combo.currentText(),
                                   origin="upper", vmin=vmin, vmax=vmax)
        self._canvas_ac.ax.set_title("Background AC Map"); self._canvas_ac.fig.tight_layout()
        self._canvas_ac.draw()

    def _setup_mask(self, cx, cy, r0):
        if self._mask_editor: self._mask_editor.disconnect()
        self._mask_editor = CircleMaskEditor(self._canvas_ac.ax, (cx, cy), r0)
        self._canvas_ac.draw()

    def _on_apply(self):
        if self._mask_editor is None: return
        cx, cy = self._mask_editor.center; r = self._mask_editor.radius
        self.bg_mask_confirmed.emit(self._ac_map, float(cx), float(cy), float(r))

    def reload_with_mask(self, roi_img, ac_map, cx, cy, r):
        self._roi_img = roi_img; self._ac_map = ac_map
        vals = np.sort(ac_map.ravel())
        self._spin_vmin.setValue(float(ac_map.min()))
        self._spin_vmax.setValue(float(vals[int(len(vals) * 0.99)]))
        self._redraw_roi(); self._redraw_ac(); self._setup_mask(cx, cy, r)
        self._btn_apply.setEnabled(True)


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — NORMALIZATION
# ══════════════════════════════════════════════════════════════════════

class Step4Widget(QWidget):
    norm_confirmed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._r_px = None; self._c_mat = None; self._c_bg = None
        self._px_size = 1.0; self._a = 1.0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("C_result(r) = C_mat(r) - a * C_bg(r)\n"
                     "(Both C_mat and C_bg are mean-subtracted, centred around 0.)")
        lbl.setWordWrap(True); layout.addWidget(lbl)

        sr = QHBoxLayout()
        sr.addWidget(QLabel("a (scale):"))
        self._slider_a = QSlider(Qt.Horizontal)
        self._slider_a.setRange(-100, 500); self._slider_a.setValue(100); sr.addWidget(self._slider_a)
        self._spin_a = QDoubleSpinBox()
        self._spin_a.setRange(-5.0, 5.0); self._spin_a.setDecimals(3)
        self._spin_a.setSingleStep(0.001); self._spin_a.setValue(1.0)
        sr.addWidget(self._spin_a)
        sr.addStretch(); layout.addLayout(sr)

        self._canvas_ov = MplCanvas(self, 8, 3); layout.addWidget(self._canvas_ov)
        self._canvas_res = MplCanvas(self, 8, 3); layout.addWidget(self._canvas_res)

        br = QHBoxLayout()
        self._btn_confirm = QPushButton("Confirm Normalization")
        br.addStretch(); br.addWidget(self._btn_confirm); br.addStretch()
        layout.addLayout(br)

        self._slider_a.valueChanged.connect(self._on_slider_a)
        self._spin_a.valueChanged.connect(self._on_spin_a)
        self._btn_confirm.clicked.connect(self._on_confirm)

    def _on_slider_a(self):
        self._a = self._slider_a.value() / 100.0
        self._spin_a.blockSignals(True)
        self._spin_a.setValue(self._a)
        self._spin_a.blockSignals(False)
        self._plot()

    def _on_spin_a(self, v):
        self._a = v
        self._slider_a.blockSignals(True)
        self._slider_a.setValue(int(v * 100))
        self._slider_a.blockSignals(False)
        self._plot()

    def compute(self, r_px_mat, c_mat, r_px_bg, c_bg, mask_r_m, mask_r_b, px_nm=1.0):
        self._px_size = px_nm
        r_min = max(mask_r_m, mask_r_b)
        r_max = min(r_px_mat[-1], r_px_bg[-1])

        m = (r_px_mat >= r_min) & (r_px_mat <= r_max)
        self._r_px = r_px_mat[m]
        self._c_mat = c_mat[m]
        self._c_bg = np.interp(self._r_px, r_px_bg, c_bg)

        self._a = 1.0
        self._slider_a.blockSignals(True); self._slider_a.setValue(100); self._slider_a.blockSignals(False)
        self._spin_a.blockSignals(True); self._spin_a.setValue(1.0); self._spin_a.blockSignals(False)
        self._plot()

    def _plot(self):
        if self._r_px is None or len(self._r_px) == 0: return
        r = self._r_px * self._px_size * 10.0 if self._px_size > 0 else self._r_px
        xl = "r (A)" if self._px_size > 0 else "r (pixels)"
        cr = self._c_mat - self._a * self._c_bg

        ax = self._canvas_ov.ax; ax.clear()
        ax.plot(r, self._c_mat, "b-", lw=1.2, label="C_mat(r)")
        ax.plot(r, self._a * self._c_bg, "--", color="orange",
                lw=1.2, label=f"a * C_bg(r)")
        ax.set_xlabel(xl); ax.set_ylabel("C(r)")
        ax.set_title("Overlay: Material vs Scaled Background (mean-subtracted)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        self._canvas_ov.fig.tight_layout(); self._canvas_ov.draw()

        ax2 = self._canvas_res.ax; ax2.clear()
        ax2.plot(r, cr, "r-", lw=1.5)
        ax2.set_xlabel(xl); ax2.set_ylabel("C_result(r)")
        ax2.set_title(f"C_result(r) = C_mat(r) - {self._a:.3f} * C_bg(r)")
        ax2.grid(True, alpha=0.3)
        self._canvas_res.fig.tight_layout(); self._canvas_res.draw()

    def _on_confirm(self):
        self.norm_confirmed.emit(self._a)

    @property
    def r_px(self): return self._r_px
    @property
    def c_mat(self): return self._c_mat
    @property
    def c_bg(self): return self._c_bg
    @property
    def c_result(self): return self._c_mat - self._a * self._c_bg


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — RESULT + SAVE
# ══════════════════════════════════════════════════════════════════════

class Step5Widget(QWidget):
    save_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._r_vals = None; self._c_mat = None; self._c_bg = None
        self._c_result = None; self._px_size = 1.0
        self._method = "splitpixel"; self._bg_enabled = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        lbl = QLabel("Final correlation function.")
        lbl.setWordWrap(True); layout.addWidget(lbl)
        self._canvas = MplCanvas(self, 8, 4)
        self._toolbar = NavToolbar(self._canvas, self)
        layout.addWidget(self._toolbar); layout.addWidget(self._canvas)
        br = QHBoxLayout()
        self._btn_save = QPushButton("Save .cr")
        br.addStretch(); br.addWidget(self._btn_save); br.addStretch()
        layout.addLayout(br)
        self._btn_save.clicked.connect(lambda: self.save_requested.emit())

    def show_result(self, r_vals, c_mat, c_bg, c_result,
                    px_nm=1.0, method="splitpixel", bg_enabled=False):
        self._r_vals = r_vals; self._c_mat = c_mat
        self._c_bg = c_bg; self._c_result = c_result
        self._px_size = px_nm; self._method = method
        self._bg_enabled = bg_enabled
        self._plot()

    def _plot(self):
        self._canvas.ax.clear()
        if self._r_vals is None: return
        r = self._r_vals * self._px_size * 10.0 if self._px_size > 0 else self._r_vals
        xl = "r (A)" if self._px_size > 0 else "r (pixels)"
        if self._bg_enabled and self._c_bg is not None:
            self._canvas.ax.plot(r, self._c_mat, "b-", alpha=0.4, lw=1, label="C_mat(r)")
            self._canvas.ax.plot(r, self._c_bg, "orange", alpha=0.4, lw=1, label="C_bg(r)")
        self._canvas.ax.plot(r, self._c_result, "r-", lw=1.5, label="C_result(r)")
        self._canvas.ax.set_xlabel(xl); self._canvas.ax.set_ylabel("C(r)")
        self._canvas.ax.set_title(f"Radial Correlation Function  ({self._method})")
        self._canvas.ax.legend(fontsize=8); self._canvas.ax.grid(True, alpha=0.3)
        self._canvas.fig.tight_layout(); self._canvas.draw()

    @property
    def r_px(self): return self._r_vals
    @property
    def c_mat_arr(self): return self._c_mat
    @property
    def c_bg_arr(self): return self._c_bg
    @property
    def c_result_arr(self): return self._c_result


# ══════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAUI v0.4 — Correlation Analysis for TEM Images")
        self.resize(1100, 800)

        # state
        self._loaded_data = None
        self._mat_roi = None; self._bg_roi = None; self._full_img = None
        self._mat_ac = None; self._bg_ac = None
        self._roi_rect = None; self._bg_rect = None
        self._bg_enabled = False
        self._mask_cx = self._mask_cy = self._mask_r = 0.0
        self._mask_cx_bg = self._mask_cy_bg = self._mask_r_bg = 0.0
        self._integration_method = "splitpixel"
        self._norm_a = 1.0

        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central)

        self._stack = QStackedWidget(); ml.addWidget(self._stack, 1)
        self._step0 = Step0Widget(); self._step1 = Step1Widget()
        self._step2 = Step2Widget(); self._step3 = Step3Widget()
        self._step4 = Step4Widget(); self._step5 = Step5Widget()
        self._stack.addWidget(self._step0); self._stack.addWidget(self._step1)
        self._stack.addWidget(self._step2); self._stack.addWidget(self._step3)
        self._stack.addWidget(self._step4); self._stack.addWidget(self._step5)
        self._stack.setCurrentIndex(0)

        nr = QHBoxLayout()
        self._btn_back = QPushButton("<- Back"); self._btn_back.setEnabled(False)
        self._btn_next = QPushButton("Next ->"); self._btn_next.setEnabled(False)
        self._btn_save = QPushButton("Save .cr"); self._btn_save.setEnabled(False)
        self._btn_save.setVisible(False)
        self._lbl_step = QLabel("Step 0")
        nr.addWidget(self._btn_back); nr.addStretch()
        nr.addWidget(self._lbl_step); nr.addStretch()
        nr.addWidget(self._btn_save); nr.addWidget(self._btn_next)
        ml.addLayout(nr)

        self._status = QStatusBar(); self.setStatusBar(self._status)

        self._btn_back.clicked.connect(self._go_back)
        self._btn_next.clicked.connect(self._go_next)
        self._btn_save.clicked.connect(self._save_cr)
        self._step0.file_loaded.connect(self._on_file_loaded)
        self._step0.cr_loaded.connect(self._on_cr_loaded)
        self._step1.roi_confirmed.connect(self._on_roi_confirmed)
        self._step2.mat_mask_confirmed.connect(self._on_mat_mask)
        self._step3.bg_mask_confirmed.connect(self._on_bg_mask)
        self._step4.norm_confirmed.connect(self._on_norm_confirmed)
        self._step5.save_requested.connect(self._save_cr)

    @property
    def _total_steps(self):
        return 5 if self._bg_enabled else 3  # 0,1,2,5 vs 0,1,2,3,4,5

    @property
    def _step_indices(self):
        """Map logical step number → stack index, respecting bg_enabled."""
        if self._bg_enabled:
            return [0, 1, 2, 3, 4, 5]  # normal flow
        else:
            return [0, 1, 2, None, None, 5]  # skip bg mask and norm

    def _idx_to_logical(self, stack_idx):
        if self._bg_enabled:
            return stack_idx
        # without bg: stack 0,1,2,5 map to logical 0,1,2,3
        m = {0: 0, 1: 1, 2: 2, 5: 3}
        return m.get(stack_idx, stack_idx)

    def _logical_to_stack(self, logical):
        if self._bg_enabled:
            return min(logical, 5)
        m = {0: 0, 1: 1, 2: 2, 3: 5}
        return m.get(logical, 5)

    def _go_back(self):
        idx = self._stack.currentIndex()
        if idx <= 0: return
        # pre-load state for the step we're going to
        if idx == 5 and self._mat_ac is not None:
            if self._bg_enabled:
                # going back to normalization step 4: no preload needed
                pass
            else:
                # going back to mask step 2 from result: reload mask
                self._step2.reload_with_mask(
                    self._mat_roi, self._mat_ac,
                    self._mask_cx, self._mask_cy, self._mask_r)
                self._restore_method()
        elif idx == 4 and self._mat_ac is not None:
            # going from norm to bg mask
            self._step3.reload_with_mask(
                self._bg_roi, self._bg_ac,
                self._mask_cx_bg, self._mask_cy_bg, self._mask_r_bg)
        elif idx == 3 and self._mat_ac is not None:
            # going from bg mask to mat mask
            self._step2.reload_with_mask(
                self._mat_roi, self._mat_ac,
                self._mask_cx, self._mask_cy, self._mask_r)
            self._restore_method()
        elif idx == 2 and self._loaded_data is not None:
            # snapshot current rects BEFORE going back (so _go_next can
            # detect changes even if the confirm signal overwrites them)
            self._pre_back_roi = self._roi_rect
            self._pre_back_bg = self._bg_rect
            self._step1.reload_with_rects(
                self._loaded_data, self._roi_rect,
                self._bg_rect if self._bg_rect else (0, 0, 0, 0),
                self._bg_enabled)
        elif idx == 1:
            pass

        prev_stack = self._stack.currentIndex() - 1
        if prev_stack == 3 and not self._bg_enabled:
            prev_stack = 2  # skip step 3 when no bg
        self._stack.setCurrentIndex(max(0, prev_stack))
        self._update_nav()

    def _go_next(self):
        idx = self._stack.currentIndex()
        if idx == 0 and self._loaded_data is not None:
            self._stack.setCurrentIndex(1)
            if self._roi_rect is not None:
                self._step1.reload_with_rects(
                    self._loaded_data, self._roi_rect,
                    self._bg_rect if self._bg_rect else (0, 0, 0, 0),
                    self._bg_enabled)
            else:
                self._step1.set_image(self._loaded_data)
        elif idx == 1 and self._step1.roi is not None:
            mat_roi, bg_roi, bg_enabled = self._step1.roi_arrays

            # detect whether ROI / BG changed since last time.
            # Compare against the pre-back snapshot (if we came here
            # via Back), otherwise against the stored rect.
            ref_roi = getattr(self, "_pre_back_roi", None) or self._roi_rect
            roi_changed = (ref_roi is not None
                           and ref_roi != self._step1.roi)
            bg_rect_new = self._step1._bg
            ref_bg = getattr(self, "_pre_back_bg", None) or self._bg_rect
            bg_changed = (ref_bg is not None and ref_bg != bg_rect_new)
            self._pre_back_roi = None; self._pre_back_bg = None

            self._mat_roi = mat_roi; self._bg_roi = bg_roi
            self._bg_enabled = bg_enabled
            self._roi_rect = self._step1.roi
            self._bg_rect = bg_rect_new if (bg_rect_new and sum(bg_rect_new) > 0) else None
            self._full_img = self._step1._full_img
            self._mat_ac = compute_autocorr_2d(mat_roi - np.mean(mat_roi))
            if bg_enabled and bg_roi is not None:
                self._bg_ac = compute_autocorr_2d(bg_roi - np.mean(bg_roi))
            else:
                self._bg_ac = None
            self._stack.setCurrentIndex(2)

            # material mask: only reuse if ROI hasn't changed
            if not roi_changed and self._mask_r > 0:
                self._step2.reload_with_mask(mat_roi, self._mat_ac,
                                              self._mask_cx, self._mask_cy, self._mask_r)
                self._restore_method()
            else:
                self._step2.set_roi(mat_roi)

            # if ROI changed, invalidate the material mask state
            if roi_changed:
                self._mask_cx = self._mask_cy = self._mask_r = 0.0

            # auto-sync default mask centre + radius from the freshly
            # created mask editor (so radial_average works even if the
            # user hasn't clicked "Apply Mask" yet)
            if self._mask_r == 0.0 and self._mat_ac is not None:
                self._mask_cx = self._mat_ac.shape[1] / 2.0
                self._mask_cy = self._mat_ac.shape[0] / 2.0
                self._mask_r = min(self._mat_ac.shape) * 0.05

            # if BG changed or BG was removed, invalidate the BG mask
            if bg_changed or not bg_enabled:
                self._mask_cx_bg = self._mask_cy_bg = self._mask_r_bg = 0.0

            # auto-sync default BG mask centre + radius
            if self._mask_r_bg == 0.0 and self._bg_ac is not None:
                self._mask_cx_bg = self._bg_ac.shape[1] / 2.0
                self._mask_cy_bg = self._bg_ac.shape[0] / 2.0
                self._mask_r_bg = min(self._bg_ac.shape) * 0.05
        elif idx == 2 and self._bg_enabled:
            if self._bg_roi is not None:
                self._stack.setCurrentIndex(3)
                if self._mask_r_bg > 0:
                    self._step3.reload_with_mask(
                        self._bg_roi, self._bg_ac,
                        self._mask_cx_bg, self._mask_cy_bg, self._mask_r_bg)
                else:
                    self._step3.set_roi(self._bg_roi)
                    # auto-sync default BG mask params (belt-and-suspenders
                    # in case the idx==1 block didn't catch it)
                    if self._mask_r_bg == 0.0 and self._bg_ac is not None:
                        self._mask_cx_bg = self._bg_ac.shape[1] / 2.0
                        self._mask_cy_bg = self._bg_ac.shape[0] / 2.0
                        self._mask_r_bg = min(self._bg_ac.shape) * 0.05
            else:
                # bg enabled but no bg roi — skip to norm with zeros
                self._stack.setCurrentIndex(4)
                self._goto_norm()
        elif idx == 2 and not self._bg_enabled:
            # no bg → skip to result
            self._goto_result()
        elif idx == 3:
            # after bg mask → go to norm
            self._stack.setCurrentIndex(4)
            self._goto_norm()
        elif idx == 4:
            pass  # norm_confirmed → _on_norm_confirmed handles it
        self._update_nav()

    def _goto_norm(self):
        px = self._loaded_data.get("px_size_nm", 1.0) if self._loaded_data else 1.0
        rp_m, cm = radial_average(self._mat_ac,
                                   center=(self._mask_cx, self._mask_cy),
                                   mask_radius=self._mask_r,
                                   method=self._integration_method)
        self._r_px_mat = rp_m; self._c_mat_raw = cm

        if self._bg_enabled and self._bg_ac is not None:
            rp_b, cb = radial_average(self._bg_ac,
                                       center=(self._mask_cx_bg, self._mask_cy_bg),
                                       mask_radius=self._mask_r_bg,
                                       method=self._integration_method)
        else:
            rp_b = rp_m; cb = np.zeros_like(cm)
        self._r_px_bg = rp_b; self._c_bg_raw = cb

        self._step4.compute(rp_m, cm, rp_b, cb,
                            self._mask_r, self._mask_r_bg, px_nm=px)
        self._stack.setCurrentIndex(4)

    def _goto_result(self):
        px = self._loaded_data.get("px_size_nm", 1.0) if self._loaded_data else 1.0
        rp_m, cm = radial_average(self._mat_ac,
                                   center=(self._mask_cx, self._mask_cy),
                                   mask_radius=self._mask_r,
                                   method=self._integration_method)
        if self._bg_enabled and self._bg_ac is not None:
            rp_b, cb = radial_average(self._bg_ac,
                                       center=(self._mask_cx_bg, self._mask_cy_bg),
                                       mask_radius=self._mask_r_bg,
                                       method=self._integration_method)
            r_min = max(self._mask_r, self._mask_r_bg)
            r_max = min(rp_m[-1], rp_b[-1])
            m = (rp_m >= r_min) & (rp_m <= r_max)
            cr = cm[m] - self._norm_a * np.interp(rp_m[m], rp_b, cb)
            self._step5.show_result(
                rp_m[m], cm[m], np.interp(rp_m[m], rp_b, cb), cr,
                px_nm=px, method=self._integration_method, bg_enabled=True)
        else:
            # no bg: C_result = C_mat
            self._step5.show_result(
                rp_m, cm, None, cm,
                px_nm=px, method=self._integration_method, bg_enabled=False)

        self._stack.setCurrentIndex(5)
        self._btn_save.setEnabled(True)

    def _restore_method(self):
        idx_c = self._step2._method_combo.findText(self._integration_method)
        if idx_c >= 0: self._step2._method_combo.setCurrentIndex(idx_c)

    def _update_nav(self):
        idx = self._stack.currentIndex()
        logical = self._idx_to_logical(idx)
        total = 3 if not self._bg_enabled else 5
        self._lbl_step.setText(f"Step {logical} / {total}")
        self._btn_back.setEnabled(logical > 0)
        self._btn_save.setVisible(idx == 5); self._btn_save.setEnabled(idx == 5)
        self._btn_next.setVisible(idx != 5)
        if idx == 0:
            self._btn_next.setEnabled(self._loaded_data is not None)
        elif idx == 1:
            self._btn_next.setEnabled(self._step1.roi is not None)
        elif idx == 2:
            self._btn_next.setEnabled(True)
        elif idx == 3:
            self._btn_next.setEnabled(self._bg_ac is not None)
        elif idx == 4:
            self._btn_next.setEnabled(False)

    # ── Signal handlers ───────────────────────────────────────────

    def _on_file_loaded(self, data):
        self._loaded_data = data; self._btn_next.setEnabled(True)
        self._status.showMessage(
            f"Loaded: {os.path.basename(data['metadata']['filepath'])}  |  "
            f"{data['metadata']['shape']}  |  "
            f"{data['px_size_nm']:.4f} nm/px  |  "
            f"{'Dark field' if data.get('dark_field') else 'Bright field'}")
        self._update_nav()

    def _on_cr_loaded(self, cr_data):
        tem_data = cr_data["tem_data"]
        if tem_data is None:
            QMessageBox.warning(self, "Source file not found",
                                "Could not locate the original TEM file.")
            self._status.showMessage(f"CR preview: {os.path.basename(cr_data['cr_filepath'])}")
            self._update_nav(); return

        self._loaded_data = tem_data
        self._roi_rect = cr_data["roi_rect"]; self._bg_rect = cr_data["bg_rect"]
        self._bg_enabled = cr_data.get("bg_enabled", False)
        self._mask_cx = cr_data["mask_cx"]; self._mask_cy = cr_data["mask_cy"]
        self._mask_r = cr_data["mask_r"]
        self._mask_cx_bg = cr_data.get("mask_cx_bg", cr_data["mask_cx"])
        self._mask_cy_bg = cr_data.get("mask_cy_bg", cr_data["mask_cy"])
        self._mask_r_bg = cr_data.get("mask_r_bg", cr_data["mask_r"])
        self._integration_method = cr_data["method"]
        self._norm_a = cr_data.get("norm_a", 1.0)
        self._btn_next.setEnabled(True)
        self._status.showMessage(
            f"Session restored from .cr | "
            f"source: {os.path.basename(cr_data['header'].get('filepath', '?'))} | "
            f"Use Next to walk through steps")
        self._update_nav()

    def _on_roi_confirmed(self, mat_roi, bg_roi, roi_rect, bg_rect, bg_enabled):
        self._mat_roi = mat_roi; self._bg_roi = bg_roi
        self._full_img = self._step1._full_img
        self._roi_rect = roi_rect
        self._bg_rect = bg_rect if sum(bg_rect) > 0 else None
        self._bg_enabled = bg_enabled
        if bg_enabled:
            self._status.showMessage(
                f"ROI confirmed | BG: correlation subtraction mode")
        else:
            self._status.showMessage("ROI confirmed (no background)")
        self._btn_next.setEnabled(True)
        self._update_nav()

    def _on_mat_mask(self, mat_ac, cx_m, cy_m, r_m, method):
        self._mat_ac = mat_ac
        self._mask_cx = cx_m; self._mask_cy = cy_m; self._mask_r = r_m
        self._integration_method = method
        self._btn_next.setEnabled(True)
        self._status.showMessage(f"Material mask applied: r={r_m:.1f} px")
        self._update_nav()

    def _on_bg_mask(self, bg_ac, cx_b, cy_b, r_b):
        self._bg_ac = bg_ac
        self._mask_cx_bg = cx_b; self._mask_cy_bg = cy_b; self._mask_r_bg = r_b
        self._btn_next.setEnabled(True)
        self._status.showMessage(f"BG mask applied: r={r_b:.1f} px")
        self._update_nav()

    def _on_norm_confirmed(self, a):
        self._norm_a = a
        self._goto_result()
        self._update_nav()

    # ── Save ──────────────────────────────────────────────────────

    def _save_cr(self):
        data = self._loaded_data
        if data is None or self._step5.r_px is None:
            QMessageBox.warning(self, "No data", "Complete all steps first.")
            return

        dn = os.path.splitext(os.path.basename(data["metadata"]["filepath"]))[0] + ".cr"
        fp, _ = QFileDialog.getSaveFileName(self, "Save correlation result", dn,
                                             "CR files (*.cr);;All files (*.*)")
        if not fp: return

        try:
            md = data["metadata"]; px = data["px_size_nm"]
            df = data.get("dark_field", False)
            rp = self._step5.r_px
            ra = rp * px * 10.0 if px > 0 else rp
            cm = self._step5.c_mat_arr; cb = self._step5.c_bg_arr
            cr = self._step5.c_result_arr

            lines = ["# CAUI v0.4 correlation result",
                     f"# filepath: {md['filepath']}",
                     f"# format: {os.path.splitext(md['filepath'])[1]}",
                     f"# image shape: {md['shape'][0]} x {md['shape'][1]}",
                     f"# pixel size: {px:.6f} nm/px",
                     f"# contrast: {'dark field' if df else 'bright field'}"]

            if self._roi_rect:
                rx0, rx1, ry0, ry1 = self._roi_rect
                lines.append(f"# material ROI: x0={rx0} y0={ry0} x1={rx1} y1={ry1}")
                lines.append(f"# material ROI size: {rx1 - rx0} x {ry1 - ry0} px")

            if self._bg_enabled and self._bg_rect:
                bx0, bx1, by0, by1 = self._bg_rect
                lines.append(f"# background ROI: x0={bx0} y0={by0} x1={bx1} y1={by1}")
                lines.append(f"# background ROI size: {bx1 - bx0} x {by1 - by0} px")

            lines.append(
                f"# mask material: "
                f"center=({self._mask_cx:.2f}, {self._mask_cy:.2f}) px  "
                f"radius={self._mask_r:.2f} px")
            if self._bg_enabled:
                lines.append(
                    f"# mask background: "
                    f"center=({self._mask_cx_bg:.2f}, {self._mask_cy_bg:.2f}) px  "
                    f"radius={self._mask_r_bg:.2f} px")
            lines.append(f"# radial integration method: {self._integration_method}")

            if self._bg_enabled:
                lines.append(
                    f"# normalization a: {self._norm_a:.4f}")
                lines.append(
                    "# columns: r_(Angstrom)  C_mat(r)  C_bg(r)  C_result(r)")
            else:
                lines.append("# columns: r_(Angstrom)  C(r)")
            lines.append("#")

            for i, ri in enumerate(ra):
                if np.isnan(cr[i]): continue
                if self._bg_enabled:
                    lines.append(f"{ri:.6f}  {cm[i]:.8e}  {cb[i]:.8e}  {cr[i]:.8e}")
                else:
                    lines.append(f"{ri:.6f}  {cr[i]:.8e}")

            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._status.showMessage(f"Saved: {fp}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))


# ══════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
