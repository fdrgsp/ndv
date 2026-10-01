from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from pytest import fixture
from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
from qtpy.QtGui import QKeyEvent, QWheelEvent
from qtpy.QtWidgets import QApplication, QWidget

from ndv._types import KeyCode, KeyMod, KeyPressEvent
from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._app import get_histogram_canvas_class
from ndv.views._qt._app import QtAppWrap
from ndv.views._qt._array_view import DimRow, PlayButton, QtArrayView

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@fixture
def viewer(qtbot: QtBot) -> QtArrayView:
    viewer = QtArrayView(QWidget(), ArrayViewerModel())
    viewer.add_lut_view(None)
    viewer.create_sliders({0: range(10), 1: range(64), 2: range(128)})
    qtbot.addWidget(viewer.frontend_widget())
    return viewer


def test_array_options(viewer: QtArrayView) -> None:
    qwdg = viewer._qwidget
    qwdg.show()
    qlut = viewer._luts[None]._qwidget
    dims_wdg = viewer._qwidget.dims_sliders
    assert dims_wdg._sliders
    play_btn = dims_wdg._layout.itemAtPosition(1, dims_wdg._rPLAY_BTN).widget()  # type: ignore[union-attr]

    assert qwdg.ndims_btn.isVisible()
    viewer._viewer_model.show_3d_button = False
    assert not qwdg.ndims_btn.isVisible()

    # Per-channel histogram buttons are hidden when use_shared_histogram=True
    assert not qlut.histogram_btn.isVisible()
    viewer._viewer_model.use_shared_histogram = False
    assert qlut.histogram_btn.isVisible()
    viewer._viewer_model.show_histogram_button = False
    assert not qlut.histogram_btn.isVisible()

    assert qwdg.set_range_btn.isVisible()
    viewer._viewer_model.show_reset_zoom_button = False
    assert not qwdg.set_range_btn.isVisible()

    assert qwdg.channel_mode_combo.isVisible()
    viewer._viewer_model.show_channel_mode_selector = False
    assert not qwdg.channel_mode_combo.isVisible()

    assert not qwdg.add_roi_btn.isVisible()
    viewer._viewer_model.show_roi_button = True
    assert qwdg.add_roi_btn.isVisible()

    assert not qwdg.center_cross_btn.isVisible()
    viewer._viewer_model.show_center_cross_button = True
    assert qwdg.center_cross_btn.isVisible()
    qwdg.center_cross_btn.setChecked(True)
    assert viewer._viewer_model.center_cross_visible
    viewer._viewer_model.center_cross_visible = False
    assert not qwdg.center_cross_btn.isChecked()

    assert isinstance(play_btn, PlayButton)
    assert play_btn.isVisible()
    viewer._viewer_model.show_play_button = False
    assert not play_btn.isVisible()


def test_histogram(viewer: QtArrayView) -> None:
    channel = None
    lut = viewer._luts[channel]

    # Ensure lut signal gets passed through the viewer with the channel as the arg
    histogram_mock = Mock()
    viewer.histogramRequested.connect(histogram_mock)
    lut._qwidget.histogram_btn.setChecked(True)
    histogram_mock.assert_called_once_with(channel)

    # Test adding the histogram widget puts it on the relevant lut
    assert lut.histogram is None
    histogram = get_histogram_canvas_class()()  # will raise if not supported
    viewer.add_histogram(channel, histogram)
    assert lut.histogram is not None


def test_shared_histogram_is_below_controls(viewer: QtArrayView) -> None:
    """The shared histogram follows the complete viewer control row."""
    histogram = Mock()
    frontend = QWidget()
    histogram.frontend_widget.return_value = frontend

    viewer.add_shared_histogram(histogram)

    layout = viewer._qwidget._left_layout
    assert layout.indexOf(frontend) > layout.indexOf(viewer._qwidget._btns)


def test_play_btn(viewer: QtArrayView, qtbot: QtBot) -> None:
    """Test the play button functionality on the array view."""
    dims_wdg = viewer._qwidget.dims_sliders
    assert dims_wdg._sliders
    play_btn = dims_wdg._layout.itemAtPosition(1, dims_wdg._rPLAY_BTN).widget()  # type: ignore[union-attr]
    assert isinstance(play_btn, PlayButton)
    play_btn._show_fps_dialog()
    play_btn._popup.accept()
    with qtbot.waitSignal(dims_wdg.currentIndexChanged, timeout=1000):
        play_btn.click()
    play_btn.click()  # stop it


def _send_wheel_event(widget: QWidget, angle_delta: int) -> None:
    event = QWheelEvent(
        QPointF(1, 1),
        QPointF(1, 1),
        QPoint(),
        QPoint(0, angle_delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def test_dimension_slider_wheel_uses_one_step_per_notch(viewer: QtArrayView) -> None:
    slider = viewer._qwidget.dims_sliders._sliders[0]
    slider.setValue(5)

    _send_wheel_event(slider._slider, 120)
    assert slider.value() == 6

    _send_wheel_event(slider._slider, -120)
    assert slider.value() == 5


def test_dimension_slider_accumulates_partial_wheel_notches(
    viewer: QtArrayView,
) -> None:
    slider = viewer._qwidget.dims_sliders._sliders[0]
    slider.setValue(5)

    _send_wheel_event(slider._slider, 60)
    assert slider.value() == 5

    _send_wheel_event(slider._slider, 60)
    assert slider.value() == 6


def test_dimension_slider_displays_one_based_position(viewer: QtArrayView) -> None:
    dims = viewer._qwidget.dims_sliders
    row = next(row for row in dims.findChildren(DimRow) if row.label.text() == "0")

    assert row.slider.value() == 0
    assert row.index_label.text() == "1"
    assert row.out_of.text() == "/ 10"

    row.slider.setValue(9)
    assert row.index_label.text() == "10"

    row.index_label.valueEdited.emit(3.0)
    assert row.slider.value() == 2


def test_dimension_slider_one_based_nonzero_range(qtbot: QtBot) -> None:
    view = QtArrayView(QWidget(), ArrayViewerModel())
    qtbot.addWidget(view.frontend_widget())
    view.create_sliders({"z": range(5, 15)})
    row = view._qwidget.dims_sliders.findChildren(DimRow)[0]

    assert row.slider.value() == 5
    assert row.index_label.text() == "1"
    assert row.out_of.text() == "/ 10"

    row.index_label.valueEdited.emit(3.0)
    assert row.slider.value() == 7


def test_key_event_filter(qtbot: QtBot) -> None:
    app = QtAppWrap()
    view = QtArrayView(QWidget(), ArrayViewerModel())
    qtbot.addWidget(view.frontend_widget())

    received: list[KeyPressEvent] = []
    view.keyPressed.connect(received.append)

    disconnect = app.filter_key_events(view.frontend_widget(), view)

    widget = view.frontend_widget()

    # Simulate a Right arrow key press on the widget
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier
    )
    QApplication.sendEvent(widget, event)
    assert len(received) == 1
    assert received[0].key == KeyCode.RIGHT
    assert received[0].mods == KeyMod.NONE

    # Simulate Shift+Left
    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Left,
        Qt.KeyboardModifier.ShiftModifier,
    )
    QApplication.sendEvent(widget, event)
    assert len(received) == 2
    assert received[1].key == KeyCode.LEFT
    assert received[1].mods == KeyMod.SHIFT

    # Cleanup
    disconnect()
