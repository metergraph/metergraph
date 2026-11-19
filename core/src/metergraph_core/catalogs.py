"""Prepared pricing catalogs with an optional background refresher."""

import logging
import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar, cast

from .loader import load_catalog

T = TypeVar("T")
log = logging.getLogger(__name__)


class Catalogs(Generic[T]):
    """Serve one prepared catalog value, retaining it across refresh failures.

    The default loader builds the bundled catalog for ``region``. A custom
    loader must return a complete value that readers treat as read-only.
    """

    def __init__(
        self,
        *,
        region: str = "global",
        loader: Callable[[], T] | None = None,
        initial: T | None = None,
        refresh_seconds: float = 300,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be positive")
        self._loader = loader or cast(Callable[[], T], lambda: load_catalog(region=region))
        self._interval = refresh_seconds
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._loading = threading.Lock()
        self._lifecycle = threading.RLock()
        self._value: T | None = initial
        self._next_refresh = 0.0
        self._prepared = False
        self._worker: threading.Thread | None = None
        self._stopped = threading.Event()

    def current(self) -> T:
        """Read prepared data, or load on demand when no lifecycle is active."""
        with self._lock:
            if self._value is not None and (
                self._prepared
                or self._monotonic() < self._next_refresh
                or self._loading.locked()
            ):
                return self._value
        return self._reload(force=False)

    def _reload(self, *, force: bool) -> T:
        with self._loading:
            with self._lock:
                previous = self._value
                if not force and previous is not None and (
                    self._prepared or self._monotonic() < self._next_refresh
                ):
                    return previous
            try:
                value = self._loader()
            except Exception:
                with self._lock:
                    self._next_refresh = self._monotonic() + self._interval
                if force or previous is None:
                    raise
                log.warning("catalog refresh failed; keeping prepared data", exc_info=True)
                return previous
            with self._lock:
                self._value = value
                self._next_refresh = self._monotonic() + self._interval
            return value

    def prepare(self) -> T:
        """Load if needed and enable prepared-only reads. Starts no thread."""
        with self._lifecycle:
            value = self._reload(force=False)
            with self._lock:
                self._prepared = True
            return value

    def refresh(self) -> T:
        """Synchronously publish a complete replacement; raise on failure."""
        return self._reload(force=True)

    def start(self) -> T:
        """Prepare before returning and start one owned refresh thread."""
        with self._lifecycle:
            value = self.prepare()
            if self._worker is None:
                self._stopped.clear()
                self._worker = threading.Thread(
                    target=self._run, name="pricing-catalog-refresh", daemon=True
                )
                self._worker.start()
            return value

    def _run(self) -> None:
        while not self._stopped.wait(self._interval):
            try:
                self.refresh()
            except Exception:
                log.warning("catalog refresh failed; keeping prepared data", exc_info=True)

    def stop(self) -> None:
        """Join owned background work, retain data, and restore on-demand reads.

        Caller-scheduled refreshes remain the caller's responsibility. A custom
        loader must bound its I/O so an owned refresh can finish during shutdown.
        """
        with self._lifecycle:
            if self._worker is not None:
                self._stopped.set()
                self._worker.join()
                self._worker = None
            with self._lock:
                self._prepared = False
