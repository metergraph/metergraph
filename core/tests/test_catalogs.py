"""Catalog lifecycle and publication behavior."""
from threading import Event, Thread

import pytest

from metergraph_core import Catalogs


def test_prepare_prevents_request_refresh_and_stop_restores_on_demand():
    clock = [0.0]
    calls = []
    def load():
        calls.append(object())
        return calls[-1]
    catalogs = Catalogs(loader=load, refresh_seconds=10, monotonic=lambda: clock[0])
    first = catalogs.current()
    assert catalogs.current() is first
    catalogs.prepare()
    clock[0] = 20
    assert catalogs.current() is first
    assert len(calls) == 1
    catalogs.stop()
    assert catalogs.current() is not first
    assert len(calls) == 2


def test_refresh_publishes_complete_value_without_blocking_readers():
    entered, release = Event(), Event()
    calls = []
    def load():
        if calls:
            entered.set()
            assert release.wait(2)
        value = object()
        calls.append(value)
        return value
    catalogs = Catalogs(loader=load)
    first = catalogs.prepare()
    thread = Thread(target=catalogs.refresh)
    thread.start()
    try:
        assert entered.wait(2)
        assert catalogs.current() is first
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert catalogs.current() is calls[-1]
    assert catalogs.current() is not first


def test_refresh_failure_preserves_prepared_value():
    calls = []
    def load():
        if calls:
            raise ValueError("unavailable")
        calls.append(object())
        return calls[-1]
    catalogs = Catalogs(loader=load)
    first = catalogs.prepare()
    with pytest.raises(ValueError, match="unavailable"):
        catalogs.refresh()
    assert catalogs.current() is first


def test_start_refreshes_in_background_and_repeated_lifecycle_is_safe():
    updated = Event()
    calls = []
    def load():
        value = object()
        calls.append(value)
        if len(calls) > 1:
            updated.set()
        return value
    catalogs = Catalogs(loader=load, refresh_seconds=0.02)
    try:
        first = catalogs.start()
        assert catalogs.start() is first
        assert updated.wait(2)
    finally:
        catalogs.stop()
        catalogs.stop()
    assert len(calls) >= 2
    assert catalogs.start() is catalogs.current()
    catalogs.stop()


def test_initial_value_is_available_during_first_source_refresh():
    entered, release = Event(), Event()
    baseline, replacement = object(), object()
    def load():
        entered.set()
        assert release.wait(2)
        return replacement
    catalogs = Catalogs(initial=baseline, loader=load)
    result = []
    thread = Thread(target=lambda: result.append(catalogs.current()))
    thread.start()
    try:
        assert entered.wait(2)
        assert catalogs.current() is baseline
    finally:
        release.set()
        thread.join(2)
    assert result == [replacement]
