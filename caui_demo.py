"""
CAUI — Correlation Analysis for TEM Images (Demo)
==================================================
Workflow:
  Step 0 → Select file, set calibration, toggle bright/dark field, preview
  Step 1 → ROI selection on the full image
  Step 2 → 2D autocorrelation map + draggable circular mask
  Step 3 → Radial average → 1D correlation function

Changelog
---------
V0.3 (2026-05-16)
  - Mask: NaN-based exclusion — mask circle position now physically excludes
    pixels from integration (pyFAI-style); drag anywhere on the AC map.
  - Integration center is always the geometric center of the AC map.
  - Step 2: "Confirm Mask" button stores params; Next button triggers
    integration (no longer auto-advances).
  - Step 2: "Reset Mask" button recenters the circle at map center.
  - Step 1: "Clear" button removes all ROI/background rectangles.
  - Fix: re-selecting ROI clears old blue squares.
  - Fix: re-selecting ROI resets mask state (no stale coordinates).
  - Fix: Log-scale toggle preserves the mask circle.
  - Fix: "Reset original view" toolbar button now uses current data bounds.
  - Fix: old mask circle properly removed when creating a new one.

V0.2 (2026-05-15)
  - Initial beta release with ROI selection, 2D autocorrelation,
    draggable circular mask, radial integration, .cr file save/load.
"""

import sys
import os
import numpy as np

# ── GUI ──────────────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFormLayout, QStackedWidget, QStatusBar, QMessageBox,
    QSplitter, QSizePolicy, QComboBox, QRadioButton, QButtonGroup,
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QAction

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
    """Load a DM3/DM4 file via hyperspy. Returns {image, px_size_nm, metadata}."""
    import hyperspy.api as hs
    sig = hs.load(filepath)
    img = sig.data
    if img.ndim == 3:
        # e.g. (1, H, W) → squeeze
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    px_size = 1.0
    ax_mgr = getattr(sig, "axes_manager", None)
    if ax_mgr is not None and len(ax_mgr.signal_axes) >= 2:
        scale_x = ax_mgr.signal_axes[0].scale
        scale_y = ax_mgr.signal_axes[1].scale
        unit_x = ax_mgr.signal_axes[0].units
        unit_y = ax_mgr.signal_axes[1].units
        if abs(scale_x - scale_y) > 1e-6 * max(scale_x, scale_y):
            print(f"  ⚠ x/y scales differ: {scale_x} vs {scale_y} – using x-scale")
        px_size = float(scale_x)
        if "nm" in str(unit_x).lower():
            pass  # already nm
        elif "µm" in str(unit_x).lower() or "um" in str(unit_x).lower():
            px_size *= 1000.0
        elif "m" in str(unit_x).lower():
            px_size *= 1e9

    metadata = {
        "filepath": filepath,
        "shape": img.shape,
        "dtype": str(img.dtype),
        "px_size_nm": px_size,
        "px_size_source": "metadata",
    }
    try:
        md = sig.metadata
        metadata["instrument"] = getattr(md, "Acquisition_instrument", None)
    except Exception:
        pass
    return {"image": img.astype(np.float64), "px_size_nm": px_size, "metadata": metadata}


def load_tiff_file(filepath: str) -> dict:
    """Load a TIFF file. Returns {image, px_size_nm, metadata} (px_size_nm=0 = unknown)."""
    import tifffile
    img = tifffile.imread(filepath)
    if img.ndim == 3:
        # e.g. RGB or stack; take first channel / page
        if img.shape[0] <= 4:
            img = img[0]  # first channel
        else:
            img = img[0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    metadata = {
        "filepath": filepath,
        "shape": img.shape,
        "dtype": str(img.dtype),
        "px_size_nm": 0.0,
        "px_size_source": "user",
    }
    return {"image": img.astype(np.float64), "px_size_nm": 0.0, "metadata": metadata}


def load_image(filepath: str, user_px_size: float = 0.0) -> dict:
    """Unified loader: dispatch by extension. Returns {image, px_size_nm, metadata}."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".dm3", ".dm4"):
        result = load_dm_file(filepath)
        if user_px_size and user_px_size > 0:
            result["px_size_nm"] = user_px_size
            result["metadata"]["px_size_source"] = "user"
        return result
    elif ext in (".tif", ".tiff"):
        result = load_tiff_file(filepath)
        if user_px_size and user_px_size > 0:
            result["px_size_nm"] = user_px_size
        return result
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def compute_autocorr_2d(image: np.ndarray) -> np.ndarray:
    """2D autocorrelation via Wiener-Khinchin: IFFT(|FFT(I)|²)."""
    F = np.fft.fft2(image)
    power = np.abs(F) ** 2
    ac = np.fft.ifft2(power).real
    return np.fft.fftshift(ac)


def radial_average(data: np.ndarray, center=None, mask_radius: float = 0.0,
                   method: str = "splitpixel") -> tuple:
    """
    Radially average a 2D array around *center*.
    Methods:
      - "splitpixel": split each pixel across floor(r) & ceil(r) by fractional distance
      - "simple":     round(r) binning (fast, coarser at small r)
    Returns (r_px, radial_mean) — r in pixel units.
    If mask_radius > 0, pixels with r <= mask_radius are excluded.
    """
    if center is None:
        center = (data.shape[0] / 2.0, data.shape[1] / 2.0)
    cy, cx = center
    y_idx, x_idx = np.indices(data.shape)
    r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    max_r = int(np.ceil(np.sqrt(cy**2 + cx**2)))  # distance to farthest corner

    valid = np.isfinite(data)
    if mask_radius > 0:
        valid[r <= mask_radius] = False

    radial_sum = np.zeros(max_r + 1, dtype=np.float64)
    radial_count = np.zeros(max_r + 1, dtype=np.float64)

    if method == "splitpixel":
        r_low = np.floor(r).astype(int)
        r_high = r_low + 1
        frac = r - r_low

        vl = (r_low <= max_r) & valid
        np.add.at(radial_sum, r_low[vl], (1.0 - frac[vl]) * data[vl])
        np.add.at(radial_count, r_low[vl], 1.0 - frac[vl])

        vh = (r_high <= max_r) & valid
        np.add.at(radial_sum, r_high[vh], frac[vh] * data[vh])
        np.add.at(radial_count, r_high[vh], frac[vh])
    else:  # "simple"
        r_int = np.round(r).astype(int)
        vm = (r_int <= max_r) & valid
        np.add.at(radial_sum, r_int[vm], data[vm])
        np.add.at(radial_count, r_int[vm], 1.0)

    radial_mean = np.divide(radial_sum, radial_count,
                            where=radial_count > 0,
                            out=np.full_like(radial_sum, np.nan))
    r_px = np.arange(0, max_r + 1, dtype=float)
    return r_px, radial_mean


# ══════════════════════════════════════════════════════════════════════
# .CR FILE PARSER
# ══════════════════════════════════════════════════════════════════════

def parse_cr_header(filepath: str) -> dict:
    """
    Parse a .cr file, returning:
      {header: {key: val, ...}, data: [(r, c), ...], raw_meta: {...}}
    where raw_meta contains parsed rects, mask params etc. as typed values.
    """
    header = {}
    data_rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                inner = stripped.lstrip("#").strip()
                if ":" in inner:
                    key, _, val = inner.partition(":")
                    header[key.strip()] = val.strip()
            else:
                parts = stripped.split()
                if len(parts) >= 2:
                    data_rows.append((float(parts[0]), float(parts[1])))

    if not data_rows:
        raise ValueError(f"No data columns found in {filepath}")

    raw = {"source_filepath": header.get("filepath", "")}

    # pixel size
    px_str = header.get("pixel size", "1.0 nm/px")
    try:
        raw["px_size_nm"] = float(px_str.split()[0])
    except (ValueError, IndexError):
        raw["px_size_nm"] = 1.0

    # contrast
    raw["dark_field"] = "dark" in header.get("contrast", "").lower()

    # background rect
    bg_str = header.get("background", "")
    if bg_str and bg_str != "none":
        raw["bg_rect"] = _parse_rect_str(bg_str)
        try:
            raw["bg_mean"] = float(bg_str.split("mean=")[1].split()[0])
        except (ValueError, IndexError):
            raw["bg_mean"] = 0.0
    else:
        raw["bg_rect"] = None
        raw["bg_mean"] = 0.0

    # ROI rect
    roi_str = header.get("ROI", "")
    if roi_str:
        raw["roi_rect"] = _parse_rect_str(roi_str)
    else:
        raw["roi_rect"] = None

    # mask
    mask_c_str = header.get("mask center", "(0, 0) px")
    try:
        nums = mask_c_str.replace("(", "").replace(")", "").replace("px", "").split(",")
        raw["mask_cx"] = float(nums[0].strip())
        raw["mask_cy"] = float(nums[1].strip())
    except (ValueError, IndexError):
        raw["mask_cx"] = raw["mask_cy"] = 0.0

    mask_r_str = header.get("mask radius", "0 px")
    try:
        raw["mask_r"] = float(mask_r_str.split()[0])
    except (ValueError, IndexError):
        raw["mask_r"] = 0.0

    # method
    raw["method"] = header.get("radial integration method", "splitpixel")

    return {
        "header": header,
        "data": data_rows,
        "raw_meta": raw,
    }


def _parse_rect_str(s: str) -> tuple:
    """Parse 'x0=123 y0=456 x1=789 y1=1011' → (x0, x1, y0, y1).
    Only recognised keys are processed; values are safely converted via float."""
    nums = {}
    for part in s.split():
        if "=" in part:
            k, _, v = part.partition("=")
            if k in ("x0", "x1", "y0", "y1"):
                nums[k] = int(float(v))
    return (nums.get("x0", 0), nums.get("x1", 0),
            nums.get("y0", 0), nums.get("y1", 0))


# ══════════════════════════════════════════════════════════════════════
# MATPLOTLIB CANVAS WITH TOOLBAR (reusable)
# ══════════════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvas):
    """Matplotlib canvas wrapped for PySide6."""
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.tight_layout()


# ══════════════════════════════════════════════════════════════════════
# CIRCULAR MASK EDITOR  (draggable center + resizable radius)
# ══════════════════════════════════════════════════════════════════════

class CircleMaskEditor:
    """
    Manages a draggable Circle patch on a matplotlib axes.
    - Left-drag inside the circle → move center
    - Left-drag on/near the edge → resize radius
    - Scroll wheel → fine radius adjust
    """
    def __init__(self, ax, center, radius, on_changed=None):
        self.ax = ax
        self.on_changed = on_changed
        self._dragging = False
        self._resizing = False
        self._press_xy = None
        self._press_center = None
        self._press_radius = None

        self.patch = Circle(
            center, radius=radius,
            fill=False, edgecolor="red", linewidth=2, linestyle="--",
            alpha=0.9, picker=5,
        )
        self.ax.add_patch(self.patch)

        # connections
        self._cid_press = self.ax.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_motion = self.ax.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_release = self.ax.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_scroll = self.ax.figure.canvas.mpl_connect("scroll_event", self._on_scroll)

    @property
    def center(self):
        return self.patch.center

    @property
    def radius(self):
        return self.patch.radius

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        contains, info = self.patch.contains(event)
        if not contains:
            return
        self._dragging = True
        self._press_xy = (event.xdata, event.ydata)
        self._press_center = self.patch.center
        self._press_radius = self.patch.radius

        # determine if resizing: distance from center > 0.7 * radius
        cx, cy = self.patch.center
        dist = np.hypot(event.xdata - cx, event.ydata - cy)
        self._resizing = dist > 0.6 * self.patch.radius

    def _on_motion(self, event):
        if not self._dragging:
            return
        if event.inaxes != self.ax:
            return
        dx = event.xdata - self._press_xy[0]
        dy = event.ydata - self._press_xy[1]
        if self._resizing:
            new_r = max(1.0, self._press_radius + 0.5 * (dx + dy))
            self.patch.set_radius(new_r)
        else:
            self.patch.set_center((self._press_center[0] + dx, self._press_center[1] + dy))
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
            delta = 1.1 if event.button == "up" else 0.9
            new_r = max(1.0, self.patch.radius * delta)
            self.patch.set_radius(new_r)
            self.ax.figure.canvas.draw_idle()
            if self.on_changed:
                self.on_changed()

    def disconnect(self):
        for cid in [self._cid_press, self._cid_motion, self._cid_release, self._cid_scroll]:
            self.ax.figure.canvas.mpl_disconnect(cid)
        if self.patch is not None:
            try:
                self.patch.remove()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════
# INDIVIDUAL STEP WIDGETS
# ══════════════════════════════════════════════════════════════════════

class Step0Widget(QWidget):
    """File selection + calibration + preview."""
    file_loaded = Signal(dict)   # emits result dict from load_image
    cr_loaded = Signal(dict)     # emits metadata dict from .cr file

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None  # loaded data dict
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── file row ──
        file_row = QHBoxLayout()
        self._btn_browse = QPushButton("Browse TEM File…")
        self._btn_load_cr = QPushButton("Load .cr File")
        self._lbl_file = QLabel("No file selected")
        self._lbl_file.setWordWrap(True)
        file_row.addWidget(self._btn_browse)
        file_row.addWidget(self._btn_load_cr)
        file_row.addWidget(self._lbl_file, 1)
        layout.addLayout(file_row)

        # ── format-specific options ──
        grp = QGroupBox("Calibration && Image Type")
        form = QFormLayout(grp)

        self._lbl_format = QLabel("—")
        form.addRow("Format:", self._lbl_format)

        self._spin_px = QDoubleSpinBox()
        self._spin_px.setRange(0.0, 1e6)
        self._spin_px.setDecimals(6)
        self._spin_px.setSuffix(" nm/px")
        self._spin_px.setToolTip("Pixel size in nm. For TIFF this is required; for DM files this overrides metadata.")
        form.addRow("Pixel size:", self._spin_px)

        self._chk_override = QCheckBox("Override metadata value")
        form.addRow("", self._chk_override)

        self._btn_group = QButtonGroup(self)
        self._rb_bright = QRadioButton("Bright field (bright = high electron density)")
        self._rb_dark = QRadioButton("Dark field (dark = high electron density)")
        self._btn_group.addButton(self._rb_bright, 0)
        self._btn_group.addButton(self._rb_dark, 1)
        self._rb_bright.setChecked(True)
        form.addRow(self._rb_bright)
        form.addRow(self._rb_dark)

        self._btn_load = QPushButton("Load && Preview")
        form.addRow("", self._btn_load)

        layout.addWidget(grp)

        # ── preview canvas ──
        self._canvas = MplCanvas(self, width=5, height=4)
        self._toolbar = NavToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        # connections
        self._btn_browse.clicked.connect(self._on_browse)
        self._btn_load_cr.clicked.connect(self._on_load_cr)
        self._btn_load.clicked.connect(self._on_load)

    def _on_browse(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select TEM file",
            os.path.join(os.path.dirname(__file__), "databank"),
            "TEM files (*.dm3 *.dm4 *.tif *.tiff);;All files (*.*)",
        )
        if fp:
            self._lbl_file.setText(fp)
            ext = os.path.splitext(fp)[1].lower()
            self._lbl_format.setText(ext.upper())
            if ext in (".dm3", ".dm4"):
                self._spin_px.setValue(0.0)
                self._spin_px.setEnabled(False)
                self._chk_override.setChecked(False)
                self._chk_override.setEnabled(True)
            else:  # TIFF
                self._spin_px.setValue(0.0)
                self._spin_px.setEnabled(True)
                self._chk_override.setChecked(True)
                self._chk_override.setEnabled(False)

    def _on_load_cr(self):
        """Load a .cr file, find the source TEM image, and restore the full session."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open correlation result",
            os.path.join(os.path.dirname(__file__), "databank"),
            "CR files (*.cr);;All files (*.*)",
        )
        if not fp:
            return

        try:
            parsed = parse_cr_header(fp)
            raw = parsed["raw_meta"]
            data_rows = parsed["data"]

            # Try to load the source TEM file
            tem_data = None
            src_path = raw["source_filepath"]
            if not src_path or not os.path.isfile(src_path):
                # Prompt user to locate the source file
                fname = os.path.basename(src_path) if src_path else "*.dm4"
                alt, _ = QFileDialog.getOpenFileName(
                    self, f"Locate source TEM file for {os.path.basename(fp)}",
                    os.path.dirname(fp) if os.path.isdir(os.path.dirname(fp)) else os.path.dirname(__file__),
                    "TEM files (*.dm3 *.dm4 *.tif *.tiff);;All files (*.*)",
                )
                if alt:
                    src_path = alt

            if src_path and os.path.isfile(src_path):
                tem_data = load_image(src_path, user_px_size=raw["px_size_nm"])
                tem_data["dark_field"] = raw["dark_field"]

            # Show preview of the CR curve in Step0 canvas
            r_vals = np.array([r for r, _ in data_rows])
            c_vals = np.array([c for _, c in data_rows])
            self._canvas.ax.remove()
            self._canvas.ax = self._canvas.fig.add_subplot(111)
            self._canvas.ax.plot(r_vals, c_vals, "b-", linewidth=1.2)
            self._canvas.ax.set_xlabel("r (Å)")
            self._canvas.ax.set_ylabel("C(r)")
            self._canvas.ax.set_title(f"CR History: {os.path.basename(fp)}")
            self._canvas.ax.grid(True, alpha=0.3)
            self._canvas.fig.tight_layout()
            self._canvas.draw()

            self._lbl_file.setText(fp)
            self._lbl_format.setText("CR (history)")

            self.cr_loaded.emit({
                "cr_filepath": fp,
                "tem_data": tem_data,
                "roi_rect": raw["roi_rect"],
                "bg_rect": raw["bg_rect"],
                "bg_mean": raw["bg_mean"],
                "mask_cx": raw["mask_cx"],
                "mask_cy": raw["mask_cy"],
                "mask_r": raw["mask_r"],
                "method": raw["method"],
                "r_vals": r_vals,
                "c_vals": c_vals,
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
                self._chk_override.isChecked() or os.path.splitext(fp)[1].lower() in (".tif", ".tiff")
            ) else 0.0
            if os.path.splitext(fp)[1].lower() in (".tif", ".tiff") and px_val <= 0:
                QMessageBox.warning(self, "Missing calibration",
                                    "TIFF files require pixel size. Please enter a value in nm/px.")
                return
            self._data = load_image(fp, user_px_size=px_val)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        # handle dark-field reversal
        img = self._data["image"].copy()
        if self._rb_dark.isChecked():
            img = -img  # invert contrast

        self._canvas.ax.clear()
        self._canvas.ax.imshow(img, cmap="gray", origin="upper")
        self._canvas.ax.set_title(
            f"Preview — {os.path.basename(fp)}  |  "
            f"{self._data['metadata']['shape']}  |  "
            f"{self._data['px_size_nm']:.4f} nm/px"
        )
        self._canvas.fig.tight_layout()
        self._canvas.draw()

        # expose dark-field flag in metadata
        self._data["dark_field"] = self._rb_dark.isChecked()
        self.file_loaded.emit(self._data)

    @property
    def data(self):
        return self._data

    def reload_with_data(self, data: dict):
        """Re-display from a previous step (when user hits Back)."""
        self._data = data
        img = data["image"].copy()
        if data.get("dark_field", False):
            img = -img
        self._canvas.ax.clear()
        self._canvas.ax.imshow(img, cmap="gray", origin="upper")
        self._canvas.ax.set_title(
            f"Preview — {os.path.basename(data['metadata']['filepath'])}  |  "
            f"{data['metadata']['shape']}"
        )
        self._canvas.fig.tight_layout()
        self._canvas.draw()


class Step1Widget(QWidget):
    """ROI selection + optional background subtraction on the full image."""
    roi_confirmed = Signal(np.ndarray, np.ndarray, float, tuple, tuple)  # (roi_bg_sub, full_img, bg_mean, roi_rect, bg_rect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._roi = None          # (xmin, xmax, ymin, ymax)
        self._bg = None           # (xmin, xmax, ymin, ymax) for background
        self._full_img = None
        self._sel_roi = None
        self._sel_bg = None
        self._mode = "roi"        # "roi" or "bg"
        self._static_roi_patch = None  # static Rectangle from reload_with_rects
        self._static_bg_patch = None   # static Rectangle from reload_with_rects
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        lbl = QLabel(
            "1. (Optional) Switch to 'Background' mode and drag a rectangle on a featureless region.\n"
            "2. Switch to 'ROI' mode and drag a rectangle on the region of interest.\n"
            "3. Click 'Confirm' when ready."
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # ── mode toggle ──
        mode_row = QHBoxLayout()
        self._btn_mode_bg = QPushButton("Background")
        self._btn_mode_bg.setCheckable(True)
        self._btn_mode_roi = QPushButton("ROI")
        self._btn_mode_roi.setCheckable(True)
        self._btn_mode_roi.setChecked(True)
        mode_row.addWidget(QLabel("Selection mode:"))
        mode_row.addWidget(self._btn_mode_roi)
        mode_row.addWidget(self._btn_mode_bg)
        mode_row.addStretch()

        self._lbl_bg_status = QLabel("Background: not set")
        mode_row.addWidget(self._lbl_bg_status)
        layout.addLayout(mode_row)

        self._canvas = MplCanvas(self, width=6, height=5)
        self._toolbar = NavToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        btn_row = QHBoxLayout()
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setToolTip("Remove all ROI and background rectangles")
        self._btn_confirm = QPushButton("Confirm")
        self._btn_confirm.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_clear)
        btn_row.addWidget(self._btn_confirm)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_mode_bg.clicked.connect(lambda: self._switch_mode("bg"))
        self._btn_mode_roi.clicked.connect(lambda: self._switch_mode("roi"))

    def _switch_mode(self, mode: str):
        self._mode = mode
        self._btn_mode_bg.setChecked(mode == "bg")
        self._btn_mode_roi.setChecked(mode == "roi")
        self._activate_selectors()

    def _activate_selectors(self):
        if self._sel_roi:
            self._sel_roi.set_active(self._mode == "roi")
        if self._sel_bg:
            self._sel_bg.set_active(self._mode == "bg")
        self._canvas.draw()

    def set_image(self, data: dict):
        self._data = data
        img = data["image"].copy()
        if data.get("dark_field", False):
            img = -img
        self._full_img = img
        self._roi = None
        self._bg = None
        self._btn_confirm.setEnabled(False)
        self._lbl_bg_status.setText("Background: not set")

        self._canvas.ax.clear()
        self._canvas.ax.imshow(img, cmap="gray", origin="upper")
        self._canvas.ax.set_title(f"Select ROI — {os.path.basename(data['metadata']['filepath'])}")
        self._canvas.fig.tight_layout()
        self._canvas.draw()

        # Create two persistent RectangleSelectors
        for attr in ("_sel_roi", "_sel_bg"):
            old = getattr(self, attr, None)
            if old:
                try:
                    old.disconnect_events()
                except Exception:
                    pass

        self._sel_roi = RectangleSelector(
            self._canvas.ax, self._on_roi_select,
            useblit=True,
            props=dict(facecolor=(0.4, 0.7, 1.0, 0.25), edgecolor=(0.2, 0.5, 1.0, 0.9),
                       linewidth=2, linestyle="-"),
            interactive=True,
        )
        self._sel_bg = RectangleSelector(
            self._canvas.ax, self._on_bg_select,
            useblit=True,
            props=dict(facecolor=(1.0, 0.6, 0.2, 0.25), edgecolor=(1.0, 0.4, 0.1, 0.9),
                       linewidth=2, linestyle="--"),
            interactive=True,
        )
        self._activate_selectors()

    def _on_roi_select(self, eclick, erelease):
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        xmin = max(0, min(x1, x2))
        xmax = min(self._full_img.shape[1], max(x1, x2) + 1)
        ymin = max(0, min(y1, y2))
        ymax = min(self._full_img.shape[0], max(y1, y2) + 1)
        if xmax - xmin < 5 or ymax - ymin < 5:
            return
        self._roi = (xmin, xmax, ymin, ymax)
        self._btn_confirm.setEnabled(True)
        self._remove_static_patches()

    def _on_bg_select(self, eclick, erelease):
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        xmin = max(0, min(x1, x2))
        xmax = min(self._full_img.shape[1], max(x1, x2) + 1)
        ymin = max(0, min(y1, y2))
        ymax = min(self._full_img.shape[0], max(y1, y2) + 1)
        if xmax - xmin < 5 or ymax - ymin < 5:
            return
        self._bg = (xmin, xmax, ymin, ymax)
        bg_val = float(np.mean(self._full_img[ymin:ymax, xmin:xmax]))
        self._lbl_bg_status.setText(f"Background: {bg_val:.2f} ({xmax - xmin}×{ymax - ymin} px)")
        self._remove_static_patches()

    def _remove_static_patches(self):
        """Remove static Rectangle patches added by reload_with_rects()."""
        for attr in ("_static_roi_patch", "_static_bg_patch"):
            patch = getattr(self, attr, None)
            if patch is not None and patch.axes is not None:
                patch.remove()
            setattr(self, attr, None)

    def _on_clear(self):
        """Clear all ROI and background selections."""
        self._roi = None
        self._bg = None
        self._remove_static_patches()
        # Wipe axes and re-plot the image so all old RectangleSelector artists are removed
        self._canvas.ax.clear()
        self._canvas.ax.imshow(self._full_img, cmap="gray", origin="upper")
        self._canvas.ax.set_title(
            f"Select ROI — {os.path.basename(self._data['metadata']['filepath'])}"
            if self._data else "Select ROI")
        self._canvas.fig.tight_layout()
        self._recreate_selectors()
        self._activate_selectors()
        self._lbl_bg_status.setText("Background: not set")
        self._btn_confirm.setEnabled(False)
        self._canvas.draw()

    def _on_confirm(self):
        if self._roi is None:
            return
        xmin, xmax, ymin, ymax = self._roi
        roi = self._full_img[ymin:ymax, xmin:xmax].copy()
        bg_mean = 0.0
        if self._bg is not None:
            bxmin, bxmax, bymin, bymax = self._bg
            bg_mean = float(np.mean(self._full_img[bymin:bymax, bxmin:bxmax]))
            roi = roi - bg_mean
        self.roi_confirmed.emit(roi, self._full_img, bg_mean,
                               self._roi, self._bg if self._bg else (0, 0, 0, 0))

    def reload_with_rects(self, data: dict, roi_rect: tuple, bg_rect: tuple, bg_mean: float):
        """Reload Step1 with pre-configured ROI and BG rectangles (from .cr restore)."""
        self._data = data
        img = data["image"].copy()
        if data.get("dark_field", False):
            img = -img
        self._full_img = img
        self._roi = roi_rect
        self._bg = bg_rect if (bg_rect and sum(bg_rect) > 0) else None
        self._btn_confirm.setEnabled(True)

        self._canvas.ax.clear()
        self._canvas.ax.imshow(img, cmap="gray", origin="upper")
        self._canvas.ax.set_title(f"Select ROI — {os.path.basename(data['metadata']['filepath'])}")
        self._canvas.fig.tight_layout()
        self._canvas.draw()

        self._recreate_selectors()
        if self._bg:
            bx0, bx1, by0, by1 = self._bg
            self._lbl_bg_status.setText(f"Background: {bg_mean:.2f} ({bx1 - bx0}×{by1 - by0} px)")
        self._activate_selectors()

        # Draw initial rectangles explicitly so they are visible on load
        if self._roi:
            rx0, rx1, ry0, ry1 = self._roi
            self._static_roi_patch = Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                facecolor=(0.4, 0.7, 1.0, 0.25), edgecolor=(0.2, 0.5, 1.0, 0.9),
                linewidth=2, linestyle="-")
            self._canvas.ax.add_patch(self._static_roi_patch)
        if self._bg:
            bx0, bx1, by0, by1 = self._bg
            self._static_bg_patch = Rectangle(
                (bx0, by0), bx1 - bx0, by1 - by0,
                facecolor=(1.0, 0.6, 0.2, 0.25), edgecolor=(1.0, 0.4, 0.1, 0.9),
                linewidth=2, linestyle="--")
            self._canvas.ax.add_patch(self._static_bg_patch)
        self._canvas.draw()

    def _recreate_selectors(self):
        """Re-create the two RectangleSelectors (used after .cr restore)."""
        for attr in ("_sel_roi", "_sel_bg"):
            old = getattr(self, attr, None)
            if old:
                try:
                    old.disconnect_events()
                except Exception:
                    pass
        self._sel_roi = RectangleSelector(
            self._canvas.ax, self._on_roi_select,
            useblit=True,
            props=dict(facecolor=(0.4, 0.7, 1.0, 0.25), edgecolor=(0.2, 0.5, 1.0, 0.9),
                       linewidth=2, linestyle="-"),
            interactive=True,
        )
        self._sel_bg = RectangleSelector(
            self._canvas.ax, self._on_bg_select,
            useblit=True,
            props=dict(facecolor=(1.0, 0.6, 0.2, 0.25), edgecolor=(1.0, 0.4, 0.1, 0.9),
                       linewidth=2, linestyle="--"),
            interactive=True,
        )

    @property
    def roi(self):
        return self._roi

    @property
    def roi_array(self):
        if self._roi is None:
            return None
        xmin, xmax, ymin, ymax = self._roi
        roi = self._full_img[ymin:ymax, xmin:xmax].copy()
        if self._bg is not None:
            bxmin, bxmax, bymin, bymax = self._bg
            bg_mean = float(np.mean(self._full_img[bymin:bymax, bxmin:bxmax]))
            roi = roi - bg_mean
        return roi


class Step2Widget(QWidget):
    """ROI image + 2D autocorrelation map side-by-side; mask operates on the AC map only."""
    mask_confirmed = Signal(np.ndarray, float, float, float, str)  # (ac_map, cx, cy, radius_px, method)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ac_map = None
        self._roi_img = None
        self._mask_editor = None
        self._im_ac = None
        self._im_roi = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        lbl = QLabel("Left: ROI preview.  Right: 2D autocorrelation map — drag the red circle to mask "
                     "the central self-correlation spot. "
                     "Left-drag center to move; left-drag edge to resize; scroll to fine-tune radius. "
                     "Click 'Apply Mask' when ready.")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # ── display controls ──
        ctrl_row = QHBoxLayout()

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["viridis", "plasma", "inferno", "gray", "hot", "jet"])
        self._cmap_combo.setCurrentText("viridis")
        ctrl_row.addWidget(QLabel("AC Colormap:"))
        ctrl_row.addWidget(self._cmap_combo)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["splitpixel", "simple"])
        self._method_combo.setCurrentText("splitpixel")
        self._method_combo.setToolTip("Radial integration method:\n"
                                      "  splitpixel — pixel-splitting, smoother at small r\n"
                                      "  simple     — round(r) binning, faster")
        ctrl_row.addWidget(QLabel("Method:"))
        ctrl_row.addWidget(self._method_combo)

        self._chk_log = QCheckBox("Log scale")
        self._chk_log.setChecked(False)
        ctrl_row.addWidget(self._chk_log)

        ctrl_row.addWidget(QLabel("vmin:"))
        self._spin_vmin = QDoubleSpinBox()
        self._spin_vmin.setRange(0, 1e20)
        self._spin_vmin.setDecimals(2)
        self._spin_vmin.setValue(0.0)
        ctrl_row.addWidget(self._spin_vmin)

        ctrl_row.addWidget(QLabel("vmax:"))
        self._spin_vmax = QDoubleSpinBox()
        self._spin_vmax.setRange(0, 1e20)
        self._spin_vmax.setDecimals(2)
        self._spin_vmax.setValue(0.0)
        ctrl_row.addWidget(self._spin_vmax)

        self._btn_refresh = QPushButton("Refresh")
        ctrl_row.addWidget(self._btn_refresh)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # ── side-by-side canvases ──
        canvases_row = QHBoxLayout()

        # left: ROI
        roi_container = QVBoxLayout()
        self._canvas_roi = MplCanvas(self, width=4, height=4)
        self._toolbar_roi = NavToolbar(self._canvas_roi, self)
        roi_container.addWidget(self._toolbar_roi)
        roi_container.addWidget(self._canvas_roi)
        canvases_row.addLayout(roi_container)

        # right: AC map
        ac_container = QVBoxLayout()
        self._canvas_ac = MplCanvas(self, width=4, height=4)
        self._toolbar_ac = NavToolbar(self._canvas_ac, self)
        ac_container.addWidget(self._toolbar_ac)
        ac_container.addWidget(self._canvas_ac)
        canvases_row.addLayout(ac_container)

        layout.addLayout(canvases_row)

        # ── apply row ──
        btn_row = QHBoxLayout()
        self._btn_reset_mask = QPushButton("Reset Mask")
        self._btn_reset_mask.setToolTip("Reset the mask circle to the image center")
        self._btn_reset_mask.setEnabled(False)
        self._btn_apply = QPushButton("Confirm Mask")
        self._btn_apply.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_reset_mask)
        btn_row.addWidget(self._btn_apply)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # connections
        self._btn_reset_mask.clicked.connect(self._on_reset_mask)
        self._btn_apply.clicked.connect(self._on_apply)
        self._cmap_combo.currentTextChanged.connect(self._redraw_ac)
        self._chk_log.toggled.connect(self._redraw_ac)
        self._btn_refresh.clicked.connect(self._redraw_ac)
        self._spin_vmin.editingFinished.connect(self._redraw_ac)
        self._spin_vmax.editingFinished.connect(self._redraw_ac)

    def set_roi(self, roi_img: np.ndarray, full_img: np.ndarray = None):
        self._roi_img = roi_img
        self._ac_map = compute_autocorr_2d(roi_img)
        cy, cx = self._ac_map.shape[0] / 2.0, self._ac_map.shape[1] / 2.0
        init_r = min(self._ac_map.shape) * 0.05

        # auto vmax
        vals = np.sort(self._ac_map.ravel())
        vmax = vals[int(len(vals) * 0.99)]
        vmin = self._ac_map.min()
        self._spin_vmin.setValue(float(vmin))
        self._spin_vmax.setValue(float(vmax))

        self._redraw_roi()
        self._redraw_ac()
        self._setup_mask(cx, cy, init_r)
        self._btn_apply.setEnabled(True)

    def _redraw_roi(self):
        if self._roi_img is None:
            return
        self._canvas_roi.ax.clear()
        self._canvas_roi.ax.imshow(self._roi_img, cmap="gray", origin="upper")
        self._canvas_roi.ax.set_title("ROI Preview")
        self._canvas_roi.fig.tight_layout()
        self._canvas_roi.draw()

    def _redraw_ac(self):
        if self._ac_map is None:
            return
        # Detach mask patch before clearing axes so it can be re-added after
        mask_patch = self._mask_editor.patch if self._mask_editor else None
        if mask_patch is not None:
            mask_patch.remove()

        self._canvas_ac.ax.clear()
        data = self._ac_map.copy()
        vmin = self._spin_vmin.value()
        vmax = self._spin_vmax.value()
        if vmax <= vmin:
            vmax = data.max()
            vmin = data.min()
        if self._chk_log.isChecked():
            data = np.log1p(data - data.min())
            if vmax > 0:
                vmax = np.log1p(vmax - self._ac_map.min())
            if vmin >= 0:
                vmin = np.log1p(vmin - self._ac_map.min())

        self._im_ac = self._canvas_ac.ax.imshow(
            data, cmap=self._cmap_combo.currentText(), origin="upper",
            vmin=vmin, vmax=vmax,
        )
        self._canvas_ac.ax.set_title("2D Autocorrelation Map")
        # Re-add mask patch after imshow (so it sits on top)
        if mask_patch is not None:
            self._canvas_ac.ax.add_patch(mask_patch)
        if self._canvas_ac.fig.axes and self._canvas_ac.fig.axes[-1] != self._canvas_ac.ax:
            try:
                self._canvas_ac.fig.colorbar(self._im_ac, ax=self._canvas_ac.ax, label="Intensity")
            except Exception:
                pass
        self._canvas_ac.fig.tight_layout()
        self._canvas_ac.draw()

        # Reset toolbar home view so "Reset original view" matches new data dimensions
        try:
            tb = self._toolbar_ac
            if hasattr(tb, '_views') and tb._views is not None:
                tb._views._home = None
                tb._views._history.clear()
                tb.push_current()
        except Exception:
            pass

    def _setup_mask(self, cx, cy, init_r):
        if self._mask_editor:
            self._mask_editor.disconnect()
        self._mask_editor = CircleMaskEditor(
            self._canvas_ac.ax, (cx, cy), init_r,
            on_changed=self._on_mask_changed,
        )
        self._btn_reset_mask.setEnabled(True)
        self._canvas_ac.draw()

    def _on_mask_changed(self):
        pass  # live preview could go here

    def _on_reset_mask(self):
        """Reset the mask circle to the center of the AC map."""
        if self._mask_editor is None or self._ac_map is None:
            return
        cy = self._ac_map.shape[0] / 2.0
        cx = self._ac_map.shape[1] / 2.0
        init_r = min(self._ac_map.shape) * 0.05
        self._mask_editor.patch.set_center((cx, cy))
        self._mask_editor.patch.set_radius(init_r)
        self._canvas_ac.draw()

    def _on_apply(self):
        if self._mask_editor is None or self._ac_map is None:
            return
        # Mask circle actual position on the AC map (may be off-center)
        mc_x, mc_y = self._mask_editor.center
        mc_r = self._mask_editor.radius
        method = self._method_combo.currentText()

        # NaN-mask: set pixels inside the mask circle to NaN (pyFAI-style)
        y_idx, x_idx = np.indices(self._ac_map.shape)
        dist = np.sqrt((x_idx - mc_x) ** 2 + (y_idx - mc_y) ** 2)
        masked_map = self._ac_map.copy()
        masked_map[dist <= mc_r] = np.nan

        # Emit: masked map + mask circle params (for reload) + method
        self.mask_confirmed.emit(masked_map, float(mc_x), float(mc_y), float(mc_r), method)

    @property
    def ac_map(self):
        return self._ac_map

    def reload_with_roi(self, roi_img: np.ndarray):
        self.set_roi(roi_img)

    def reload_with_mask(self, roi_img: np.ndarray, ac_map: np.ndarray,
                         mask_cx: float, mask_cy: float, mask_r: float):
        """Reload Step2 with pre-configured mask (from .cr restore)."""
        self._roi_img = roi_img
        self._ac_map = ac_map
        vals = np.sort(ac_map.ravel())
        vmax = vals[int(len(vals) * 0.99)]
        vmin = ac_map.min()
        self._spin_vmin.setValue(float(vmin))
        self._spin_vmax.setValue(float(vmax))
        self._method_combo.setCurrentText("splitpixel")  # will be overridden if needed
        self._redraw_roi()
        self._redraw_ac()
        self._setup_mask(mask_cx, mask_cy, mask_r)
        self._btn_apply.setEnabled(True)


class Step3Widget(QWidget):
    """1D radial correlation function."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._r_px = None
        self._c_r = None
        self._px_size = 1.0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        lbl = QLabel("Radial correlation function (masked central spot excluded).")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self._canvas = MplCanvas(self, width=6, height=4)
        self._toolbar = NavToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    def compute(self, ac_map: np.ndarray, cx: float, cy: float, mask_radius: float,
                px_size_nm: float = 1.0, method: str = "splitpixel"):
        self._px_size = px_size_nm
        self._method = method
        self._r_px, self._c_r = radial_average(ac_map, center=(cy, cx), mask_radius=mask_radius,
                                                method=method)
        self._plot()

    def _plot(self):
        self._canvas.ax.clear()
        if self._r_px is None:
            return
        if self._px_size and self._px_size > 0:
            r = self._r_px * self._px_size * 10.0  # nm → Å
            xlabel = "r (Å)"
        else:
            r = self._r_px
            xlabel = "r (pixels)"

        self._canvas.ax.plot(r, self._c_r, "b-", linewidth=1.5)
        self._canvas.ax.set_xlabel(xlabel)
        self._canvas.ax.set_ylabel("C(r)")
        method_label = getattr(self, "_method", "splitpixel")
        self._canvas.ax.set_title(f"Radial Correlation Function  ({method_label})")
        self._canvas.ax.grid(True, alpha=0.3)
        self._canvas.fig.tight_layout()
        self._canvas.draw()


# ══════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAUI beta V0.3 — Correlation Analysis for TEM Images")
        self.resize(1100, 750)

        # shared state
        self._loaded_data = None  # from step 0
        self._roi_img = None
        self._full_img = None
        self._ac_map = None
        # metadata for .cr export
        self._roi_rect = None
        self._bg_rect = None
        self._bg_mean = 0.0
        self._mask_cx = self._mask_cy = self._mask_r = 0.0
        self._integration_method = "splitpixel"

        # central widget with stacked steps
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self._stack = QStackedWidget()
        main_layout.addWidget(self._stack, 1)

        # steps
        self._step0 = Step0Widget()
        self._step1 = Step1Widget()
        self._step2 = Step2Widget()
        self._step3 = Step3Widget()
        self._stack.addWidget(self._step0)  # index 0
        self._stack.addWidget(self._step1)  # index 1
        self._stack.addWidget(self._step2)  # index 2
        self._stack.addWidget(self._step3)  # index 3
        self._stack.setCurrentIndex(0)

        # navigation bar
        nav_row = QHBoxLayout()
        self._btn_back = QPushButton("← Back")
        self._btn_next = QPushButton("Next →")
        self._btn_save = QPushButton("Save .cr")
        self._btn_save.setEnabled(False)
        self._btn_back.setEnabled(False)
        self._btn_next.setEnabled(False)

        self._lbl_step = QLabel("Step 0 / 3")
        nav_row.addWidget(self._btn_back)
        nav_row.addStretch()
        nav_row.addWidget(self._lbl_step)
        nav_row.addStretch()
        nav_row.addWidget(self._btn_save)
        nav_row.addWidget(self._btn_next)
        main_layout.addLayout(nav_row)

        # status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # connections
        self._btn_back.clicked.connect(self._go_back)
        self._btn_next.clicked.connect(self._go_next)
        self._btn_save.clicked.connect(self._save_cr)
        self._step0.file_loaded.connect(self._on_file_loaded)
        self._step0.cr_loaded.connect(self._on_cr_loaded)
        self._step1.roi_confirmed.connect(self._on_roi_confirmed)
        self._step2.mask_confirmed.connect(self._on_mask_confirmed)

    def _go_back(self):
        idx = self._stack.currentIndex()
        if idx <= 0:
            return
        new_idx = idx - 1
        # Restore state on the target step
        if idx == 3 and self._ac_map is not None:
            # going back to Step 2: restore AC with mask
            self._step2.reload_with_mask(
                self._roi_img, self._ac_map,
                self._mask_cx, self._mask_cy, self._mask_r,
            )
            self._method_combo_set_state(self._integration_method)
        elif idx == 2 and self._loaded_data is not None:
            # going back to Step 1: restore image with rects
            self._step1.reload_with_rects(
                self._loaded_data, self._roi_rect,
                self._bg_rect if self._bg_rect else (0, 0, 0, 0),
                self._bg_mean,
            )
        elif idx == 1:
            # going back to Step 0
            pass
        self._stack.setCurrentIndex(new_idx)
        self._update_nav()

    def _method_combo_set_state(self, method: str):
        """Set the method combo in Step2 to a given value."""
        idx = self._step2._method_combo.findText(method)
        if idx >= 0:
            self._step2._method_combo.setCurrentIndex(idx)

    def _go_next(self):
        idx = self._stack.currentIndex()
        if idx == 0 and self._loaded_data is not None:
            self._stack.setCurrentIndex(1)
            if self._roi_rect is not None:
                self._step1.reload_with_rects(
                    self._loaded_data, self._roi_rect,
                    self._bg_rect if self._bg_rect else (0, 0, 0, 0),
                    self._bg_mean)
            else:
                self._step1.set_image(self._loaded_data)
        elif idx == 1 and self._step1.roi is not None:
            roi = self._step1.roi_array
            self._roi_img = roi
            self._roi_rect = self._step1.roi
            self._ac_map = compute_autocorr_2d(roi)
            self._stack.setCurrentIndex(2)
            if self._mask_r > 0:
                self._step2.reload_with_mask(
                    roi, self._ac_map,
                    self._mask_cx, self._mask_cy, self._mask_r)
                self._method_combo_set_state(self._integration_method)
            else:
                self._step2.set_roi(roi)
        elif idx == 2 and self._ac_map is not None:
            self._stack.setCurrentIndex(3)
            # Apply NaN mask using the actual mask circle position
            masked = self._ac_map.copy()
            if self._mask_r > 0:
                y_idx, x_idx = np.indices(masked.shape)
                dist = np.sqrt((x_idx - self._mask_cx) ** 2 + (y_idx - self._mask_cy) ** 2)
                masked[dist <= self._mask_r] = np.nan
            # Integration center is always the map center
            map_cx = self._ac_map.shape[1] / 2.0
            map_cy = self._ac_map.shape[0] / 2.0
            px_size = self._loaded_data.get("px_size_nm", 1.0) if self._loaded_data else 1.0
            self._step3.compute(masked, map_cx, map_cy,
                               0.0, px_size_nm=px_size,
                               method=self._integration_method)
            self._btn_save.setEnabled(True)
        self._update_nav()

    def _on_file_loaded(self, data: dict):
        self._loaded_data = data
        self._btn_next.setEnabled(True)
        self._status.showMessage(
            f"Loaded: {os.path.basename(data['metadata']['filepath'])}  |  "
            f"{data['metadata']['shape']}  |  "
            f"{data['px_size_nm']:.4f} nm/px  |  "
            f"{'Dark field' if data.get('dark_field') else 'Bright field'}"
        )
        self._update_nav()

    def _on_cr_loaded(self, cr_data: dict):
        """Fully restore a session from a .cr history file."""
        tem_data = cr_data["tem_data"]
        if tem_data is None:
            QMessageBox.warning(
                self, "Source file not found",
                "Could not locate the original TEM file.\n\n"
                "The .cr curve is shown in the preview. "
                "To restore the full session, please re-open the .cr and select the source file."
            )
            self._status.showMessage(
                f"CR history (preview only): {os.path.basename(cr_data['cr_filepath'])}"
            )
            self._update_nav()
            return

        self._loaded_data = tem_data
        self._roi_rect = cr_data["roi_rect"]
        self._bg_rect = cr_data["bg_rect"] if cr_data["bg_rect"] else None
        self._bg_mean = cr_data["bg_mean"]
        self._mask_cx = cr_data["mask_cx"]
        self._mask_cy = cr_data["mask_cy"]
        self._mask_r = cr_data["mask_r"]
        self._integration_method = cr_data["method"]

        # State only — user clicks Next to walk through each step
        self._btn_next.setEnabled(True)
        hdr = cr_data["header"]
        self._status.showMessage(
            f"Session restored from .cr | "
            f"source: {os.path.basename(hdr.get('filepath', '?'))} | "
            f"Use Next to walk through steps"
        )
        self._update_nav()

    def _on_roi_confirmed(self, roi_img, full_img, bg_mean, roi_rect, bg_rect):
        self._roi_img = roi_img
        self._full_img = full_img
        self._roi_rect = roi_rect
        self._bg_rect = bg_rect if sum(bg_rect) > 0 else None
        self._bg_mean = bg_mean
        # Reset mask state — a new ROI invalidates the old mask position
        self._mask_cx = 0.0
        self._mask_cy = 0.0
        self._mask_r = 0.0
        if bg_mean:
            self._status.showMessage(
                f"ROI confirmed | Background subtracted: {bg_mean:.2f}"
            )
        else:
            self._status.showMessage("ROI confirmed (no background subtraction)")
        self._btn_next.setEnabled(True)
        self._update_nav()

    def _on_mask_confirmed(self, _masked_ac, cx, cy, radius, method):
        # Keep self._ac_map as the original (non-masked) for reload display.
        # Store mask circle params so the circle can be restored on Back.
        self._mask_cx = cx
        self._mask_cy = cy
        self._mask_r = radius
        self._integration_method = method
        self._btn_next.setEnabled(True)
        self._status.showMessage("Mask confirmed — press Next to integrate")
        self._update_nav()

    def _save_cr(self):
        """Export the 1D correlation function with header metadata to a .cr file."""
        data = self._loaded_data
        if data is None or self._step3._r_px is None:
            QMessageBox.warning(self, "No data", "Complete all steps before saving.")
            return

        default_name = os.path.splitext(os.path.basename(data["metadata"]["filepath"]))[0] + ".cr"
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save correlation result", default_name,
            "CR files (*.cr);;All files (*.*)",
        )
        if not fp:
            return

        md = data["metadata"]
        px = data["px_size_nm"]
        df = data.get("dark_field", False)

        lines = []
        lines.append(f"# CAUI correlation result — {os.path.basename(fp)}")
        lines.append(f"# filepath: {md['filepath']}")
        lines.append(f"# format: {os.path.splitext(md['filepath'])[1]}")
        lines.append(f"# image shape: {md['shape'][0]} × {md['shape'][1]}")
        lines.append(f"# pixel size: {px:.6f} nm/px")
        lines.append(f"# contrast: {'dark field' if df else 'bright field'}")

        if self._bg_rect:
            bx0, bx1, by0, by1 = self._bg_rect
            lines.append(f"# background: x0={bx0} y0={by0} x1={bx1} y1={by1} mean={self._bg_mean:.4f}")
        else:
            lines.append(f"# background: none")

        if self._roi_rect:
            rx0, rx1, ry0, ry1 = self._roi_rect
            lines.append(f"# ROI: x0={rx0} y0={ry0} x1={rx1} y1={ry1}")
            lines.append(f"# ROI size: {rx1 - rx0} × {ry1 - ry0} px")

        lines.append(f"# mask center: ({self._mask_cx:.2f}, {self._mask_cy:.2f}) px")
        lines.append(f"# mask radius: {self._mask_r:.2f} px")

        lines.append(f"# radial integration method: {self._integration_method}")
        lines.append(f"# columns: r_(Angstrom)  C(r)")
        lines.append("#")

        # data columns
        r_px = self._step3._r_px
        c_r = self._step3._c_r
        r_angstrom = r_px * px * 10.0 if px > 0 else r_px

        for ri, ci in zip(r_angstrom, c_r):
            if np.isnan(ci):
                continue
            lines.append(f"{ri:.6f}  {ci:.8e}")

        try:
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._status.showMessage(f"Saved: {fp}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _update_nav(self):
        idx = self._stack.currentIndex()
        self._lbl_step.setText(f"Step {idx} / 3")
        self._btn_back.setEnabled(idx > 0)
        self._btn_save.setEnabled(idx == 3)
        self._btn_save.setVisible(idx == 3)
        self._btn_next.setVisible(idx != 3)
        if idx == 0:
            self._btn_next.setEnabled(self._loaded_data is not None)
            self._btn_next.setText("Next →")
        elif idx == 1:
            self._btn_next.setEnabled(self._step1.roi is not None)
            self._btn_next.setText("Next →")
        elif idx == 2:
            self._btn_next.setEnabled(self._mask_r > 0)
            self._btn_next.setText("Next →")
        elif idx == 3:
            self._btn_next.setEnabled(False)
            self._btn_next.setVisible(False)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
