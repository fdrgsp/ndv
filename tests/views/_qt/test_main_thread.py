from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import Qt, QThread

if TYPE_CHECKING:
    from concurrent.futures import Future

    import pytest
    from pytestqt.qtbot import QtBot
    from qtpy.QtWidgets import QApplication


def test_invoker_is_destroyed_on_main_thread(
    qapp: QApplication, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ndv.views._qt import _main_thread as mt

    original_invoke = mt.MainThreadInvoker.invoke
    submitted = threading.Event()
    release_worker = threading.Event()
    called = threading.Event()
    destroyed_on_main: list[bool] = []

    def invoke(self: mt.MainThreadInvoker, *args: Any, **kwargs: Any) -> Future:
        self.destroyed.connect(
            lambda: destroyed_on_main.append(QThread.currentThread() is qapp.thread()),
            Qt.ConnectionType.DirectConnection,
        )
        future = original_invoke(self, *args, **kwargs)
        submitted.set()
        # Make the main-thread callback finish before the submitting worker
        # drops its local reference to the invoker.
        if not release_worker.wait(5):
            raise TimeoutError("Main thread did not release the worker")
        return future

    monkeypatch.setattr(mt.MainThreadInvoker, "invoke", invoke)
    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(mt.call_in_main_thread, called.set)
        try:
            qtbot.waitUntil(submitted.is_set)
            qtbot.waitUntil(called.is_set)
        finally:
            release_worker.set()
        task.result(timeout=5).result(timeout=5)

    qtbot.waitUntil(lambda: bool(destroyed_on_main))
    assert destroyed_on_main == [True]
