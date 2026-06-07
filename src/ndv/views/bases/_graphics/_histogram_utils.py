"""Shared utilities for histogram canvas implementations."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy.typing as npt

LUT_LINE_ALPHA = 0.6


class Grabbable(Enum):
    NONE = auto()
    LEFT_CLIM = auto()
    RIGHT_CLIM = auto()
    GAMMA = auto()


# ------------ Histogram data helpers ------------ #


def downsample_histogram(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    max_display_bins: int = 800,
    visible_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample histogram for display, optionally cropping to visible range.

    Parameters
    ----------
    counts : np.ndarray
        Raw histogram counts.
    bin_edges : np.ndarray
        Raw bin edges (len = len(counts) + 1).
    max_display_bins : int
        Target number of bins after downsampling.
    visible_range : tuple[float, float] | None
        If provided, crop to this (x_lo, x_hi) range before downsampling.
        A small margin of extra bins is kept so panning feels seamless.
    """
    if visible_range is not None:
        counts, bin_edges = _crop_histogram(counts, bin_edges, *visible_range)

    n = len(counts)
    if n > max_display_bins:
        factor = n // max_display_bins
        trim = n - (n % factor)
        counts = counts[:trim].reshape(-1, factor).mean(axis=1)
        bin_edges = np.concatenate(
            [bin_edges[:trim:factor], bin_edges[trim : trim + 1]]
        )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, counts


def _crop_histogram(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    x_lo: float,
    x_hi: float,
    margin_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop histogram to the visible range plus a margin for panning."""
    span = x_hi - x_lo
    margin = span * margin_frac
    lo = x_lo - margin
    hi = x_hi + margin

    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    mask = (centers >= lo) & (centers <= hi)
    if not np.any(mask):
        return counts, bin_edges

    idx = np.nonzero(mask)[0]
    i0, i1 = idx[0], idx[-1] + 1
    return counts[i0:i1], bin_edges[i0 : i1 + 1]


def area_to_mesh(
    centers: np.ndarray,
    counts: np.ndarray,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    """Convert area plot data to mesh vertices and faces (triangle strip)."""
    n = len(centers)
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)

    # 2 vertices per point: one on the curve, one on the baseline
    vertices = np.zeros((2 * n, 3), np.float32)
    vertices[0::2, 0] = centers
    vertices[0::2, 1] = counts
    vertices[1::2, 0] = centers
    # vertices[1::2, 1] = 0  (baseline, already 0)

    # Vectorized face generation
    idx = np.arange(n - 1, dtype=np.uint32)
    faces = np.zeros((2 * (n - 1), 3), np.uint32)
    faces[0::2, 0] = 2 * idx  # top_left
    faces[0::2, 1] = 2 * idx + 1  # bot_left
    faces[0::2, 2] = 2 * idx + 2  # top_right
    faces[1::2, 0] = 2 * idx + 1  # bot_left
    faces[1::2, 1] = 2 * idx + 3  # bot_right
    faces[1::2, 2] = 2 * idx + 2  # top_right

    return vertices, faces


# ------------ Log scale ------------ #


def apply_log_counts(counts: np.ndarray, log_base: float | None) -> np.ndarray:
    """Apply log transform to counts if log_base is set."""
    if log_base:
        return np.log(counts + 1) / np.log(log_base)  # type: ignore[no-any-return]
    return counts


# ------------ Range computation ------------ #


def channel_y_max(display_counts: np.ndarray | None, log_base: float | None) -> float:
    """Get the max displayed count for channel data."""
    if display_counts is None:
        return 1.0
    counts = apply_log_counts(display_counts, log_base)
    return float(np.max(counts)) if len(counts) > 0 else 1.0


def compute_x_range(
    channels: Mapping,
) -> tuple[float, float] | None:
    """Compute x range across visible channels with bin_edges.

    Channels must have `.visible` and `.bin_edges` attributes.
    """
    x_min, x_max = float("inf"), float("-inf")
    for ch in channels.values():
        if not ch.visible or ch.bin_edges is None:
            continue
        x_min = min(x_min, ch.bin_edges[0])
        x_max = max(x_max, ch.bin_edges[-1])
    if x_min == float("inf"):
        return None
    return (float(x_min), float(x_max))


def compute_y_range(
    channels: Mapping,
    log_base: float | None,
    y_headroom: float = 1.05,
) -> tuple[float, float] | None:
    """Compute y range across visible channels, with headroom.

    Channels must have `.visible` and `._display_counts` attributes.
    """
    y_max = 0.0
    for ch in channels.values():
        if not ch.visible:
            continue
        y_max = max(y_max, channel_y_max(ch._display_counts, log_base))
    if y_max == 0:
        return None
    return (0, y_max * y_headroom)


def y_top_from_range(
    y_range: tuple[float, float] | None, y_top_fraction: float = 0.98
) -> float:
    """Get the effective y_top for clim/gamma handle positioning.

    y_top_frac is Fraction of y_max used as upper bound for clim handles and gamma curve
    """
    y_max = y_range[1] if y_range else 1.0
    if y_max == 0:
        y_max = 1.0
    return y_max * y_top_fraction


# ------------ LUT geometry ------------ #


def gamma_handle_pos(
    clims: tuple[float, float], gamma: float, y_top: float
) -> tuple[float, float]:
    """Compute the (x, y) position of the gamma drag handle."""
    return float(np.mean(clims)), (2 ** (-gamma)) * y_top


# ------------ Mouse interaction helpers ------------ #


def clamp_clim_drag(
    grabbed: Grabbable,
    data_x: float,
    current_clims: tuple[float, float],
    bounds: tuple[float | None, float | None],
) -> tuple[float, float]:
    """Compute new clims from a clim-handle drag at data_x."""
    lo, hi = bounds
    c = data_x
    if lo is not None:
        c = max(c, lo)
    if hi is not None:
        c = min(c, hi)
    if grabbed is Grabbable.LEFT_CLIM:
        return (min(current_clims[1], c), current_clims[1])
    return (current_clims[0], max(current_clims[0], c))


def gamma_from_mouse_y(
    data_y: float,
    y_range: tuple[float, float] | None,
    min_gamma: float = np.float64(1e-6),
) -> float | None:
    """Compute gamma from a mouse y position. Returns None if out of bounds."""
    y_top = y_top_from_range(y_range)
    if data_y <= 0 or data_y > y_top:
        return None
    if y_top == 0:
        return 1.0
    return max(float(min_gamma), -np.log2(data_y / y_top))  # type: ignore[no-any-return]


_NO_KEY = object()  # sentinel for "no channel grabbed"


def find_nearest_grabbable(
    channels: Mapping,
    click_xy: tuple[float, float],
    world_to_canvas: Callable[[float, float], tuple[float, float]],
    y_range: tuple[float, float] | None,
    tolerance: int = 5,
) -> tuple[object, Grabbable]:
    """Find the nearest grabbable handle across all visible channels.

    Parameters
    ----------
    channels : Mapping
        Mapping of channel keys to objects with `.visible`, `.clims`, `.gamma`.
    click_xy : tuple
        Click position in canvas coordinates.
    world_to_canvas : callable
        Maps (data_x, data_y) to (canvas_x, canvas_y).
    y_range : tuple or None
        Current y range from compute_y_range.
    tolerance : int
        Max pixel distance to consider a handle "nearby".
    """
    click_x, click_y = click_xy
    best_dist = float("inf")
    best_key: object = _NO_KEY
    best_grab = Grabbable.NONE

    y_top = y_top_from_range(y_range)

    for key, ch in channels.items():
        if not ch.visible or ch.clims is None:
            continue

        left_cx = world_to_canvas(ch.clims[0], 0)[0]
        right_cx = world_to_canvas(ch.clims[1], 0)[0]

        d_right = abs(right_cx - click_x)
        if d_right < tolerance and d_right < best_dist:
            best_dist = d_right
            best_key = key
            best_grab = Grabbable.RIGHT_CLIM

        d_left = abs(left_cx - click_x)
        if d_left < tolerance and d_left < best_dist:
            best_dist = d_left
            best_key = key
            best_grab = Grabbable.LEFT_CLIM

        mid_x, mid_y = gamma_handle_pos(ch.clims, ch.gamma, y_top)
        gx, gy = world_to_canvas(mid_x, mid_y)
        d_gamma = ((gx - click_x) ** 2 + (gy - click_y) ** 2) ** 0.5
        if d_gamma < tolerance and d_gamma < best_dist:
            best_dist = d_gamma
            best_key = key
            best_grab = Grabbable.GAMMA

    return best_key, best_grab
