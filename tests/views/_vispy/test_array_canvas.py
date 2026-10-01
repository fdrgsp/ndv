from __future__ import annotations

import numpy as np
import pytest

from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._vispy._array_canvas import VispyArrayCanvas


@pytest.mark.usefixtures("any_app")
def test_zoom_center() -> None:
    """Zoom should keep the center point fixed in world space."""
    canvas = VispyArrayCanvas(ArrayViewerModel())
    canvas.set_ndim(2)
    canvas.add_image(np.random.rand(100, 100).astype(np.float32))
    canvas.set_range()

    cam = canvas._camera
    initial_rect = cam.rect

    # Zoom in at a specific world point
    center = (30.0, 70.0)
    canvas.zoom(factor=0.5, center=center)

    # Camera rect should have changed (smaller = zoomed in)
    new_rect = cam.rect
    assert new_rect.width < initial_rect.width

    # The center point should still be inside the rect
    assert new_rect.left <= center[0] <= new_rect.right
    assert new_rect.bottom <= center[1] <= new_rect.top

    # Zoom back out by the inverse factor
    canvas.zoom(factor=2.0, center=center)

    # Should return approximately to initial state
    restored_rect = cam.rect
    assert restored_rect.width == pytest.approx(initial_rect.width)
    assert restored_rect.height == pytest.approx(initial_rect.height)
    assert restored_rect.left == pytest.approx(initial_rect.left)
    assert restored_rect.bottom == pytest.approx(initial_rect.bottom)

    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_center_cross_is_above_composite_images() -> None:
    canvas = VispyArrayCanvas(ArrayViewerModel())
    canvas.set_ndim(2)
    first = canvas.add_image(np.zeros((100, 200), dtype=np.uint8))
    second = canvas.add_image(np.zeros((100, 200), dtype=np.uint8))

    canvas.set_center_cross(True)

    assert canvas._center_cross_lines is not None
    horizontal, vertical = canvas._center_cross_lines
    assert horizontal.parent is canvas._view.scene
    assert vertical.parent is canvas._view.scene
    assert horizontal.order > first._visual.order
    assert horizontal.order > second._visual.order
    assert horizontal.pos.tolist() == [[0, 50], [200, 50]]
    assert vertical.pos.tolist() == [[100, 0], [100, 100]]

    canvas.set_center_cross(False)
    assert canvas._center_cross_lines is None
    assert horizontal.parent is None
    assert vertical.parent is None
    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_center_cross_tracks_image_size_and_scale() -> None:
    canvas = VispyArrayCanvas(ArrayViewerModel())
    canvas.set_ndim(2)
    image = canvas.add_image(np.zeros((40, 80), dtype=np.uint8))
    canvas.set_center_cross(True)
    canvas.set_scales((2.0, 3.0))

    assert canvas._center_cross_lines is not None
    horizontal, vertical = canvas._center_cross_lines
    assert horizontal.transform is image._visual.transform
    assert vertical.transform is image._visual.transform
    assert horizontal.pos.tolist() == [[0, 20], [80, 20]]
    assert vertical.pos.tolist() == [[40, 0], [40, 40]]

    image.set_data(np.zeros((20, 60), dtype=np.uint8))
    canvas.refresh()
    assert horizontal.pos.tolist() == [[0, 10], [60, 10]]
    assert vertical.pos.tolist() == [[30, 0], [30, 20]]
    canvas.close()
