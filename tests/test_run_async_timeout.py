"""Verify that LeanServer handler threads never wait forever on a stuck
event loop.

Before the fix, every ``_run_async(coro)`` call in server.py passed
``timeout=None`` and so the HTTP handler thread would block on
``future.result()`` indefinitely if any coroutine on the loop didn't
complete.  That was the architectural root cause of the "leanserver
looks alive but /status hangs" failure mode - one stuck coro froze every
handler behind it.

These tests exercise the fix by blocking the event loop *on purpose*
(via ``loop.call_soon_threadsafe`` + ``time.sleep`` on the loop thread,
which really parks the loop - not just an async sleep that yields) and
asserting that handler threads still return promptly with HTTP 503.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import http.client
import json
import threading
import time
import urllib.error

import pytest

from leantree.repl_adapter.server import (
    LeanServer,
    LeanClient,
    LOOP_WATCHDOG_THRESHOLD,
    RUN_ASYNC_HEADROOM,
    DEFAULT_IS_VALID_SOURCE_TIMEOUT,
)


class _FakePool:
    """Minimum shape of LeanProcessPool that LeanServer needs.

    Stubs out the new pool API (``acquire_async``, ``release``,
    ``return_entry_async``, ``get_entry``, ``stats``, ``start``,
    ``shutdown``) so we can exercise server behaviour without spinning up
    real Lean. ``acquire_async`` simulates a fully-occupied pool by sleeping
    until the caller's timeout fires - that's what lets these tests time
    out the get-process path on a stuck loop.
    """

    def __init__(self, max_processes: int = 2):
        self.max_processes = max_processes
        self.has_active_request = lambda entry_id: False
        # Backing state for stats(). Stays empty here - this fake never
        # actually hands out an entry.
        self._live: dict[int, object] = {}
        self._lock = threading.Lock()

    def start(self, loop):
        return None

    async def start_async(self):
        return None

    def shutdown(self):
        return None

    async def shutdown_async(self):
        return None

    async def acquire_async(self, timeout: float | None = None):
        # Simulates a fully-occupied pool: never returns an entry.
        if timeout is None:
            await asyncio.sleep(999)
        else:
            await asyncio.sleep(timeout)
        return None

    async def return_entry_async(self, entry):
        return None

    def release(self, entry, *, recycle: bool, reason: str = ""):
        return None

    def get_entry(self, entry_id: int):
        return None

    def touch(self, entry):
        return None

    def stats(self):
        from leantree.repl_adapter.process_pool import PoolStats
        return PoolStats(
            max_processes=self.max_processes,
            total_processes=0,
            available_processes=0,
            starting_processes=0,
            used_processes=0,
            stopping_processes=0,
        )


@pytest.fixture
def running_server():
    """Start a LeanServer on an ephemeral port, yield (server, port), tear
    down.  Uses a fake pool so no Lean subprocess is involved."""
    pool = _FakePool(max_processes=2)
    server = LeanServer(pool, address="127.0.0.1", port=0, log_level="ERROR")
    server.start()
    port = server.server.server_address[1]
    try:
        yield server, port
    finally:
        server.stop()


def _post_json(port: int, path: str, body: dict, timeout: float = 120.0) -> tuple[int, dict]:
    """POST JSON over the stdlib HTTP client (not urllib) so we can tune
    our own socket timeout independently from any urllib global."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read()
        try:
            return resp.status, json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            return resp.status, {"raw": payload}
    finally:
        conn.close()


def _block_loop(server: LeanServer, seconds: float) -> threading.Event:
    """Make the event loop unresponsive for ``seconds`` seconds.

    We schedule a callback that calls ``time.sleep`` directly on the loop
    thread - not ``asyncio.sleep``, which would yield.  Returns an event
    set once the block has actually started so the caller can synchronize.
    """
    started = threading.Event()

    def _park():
        started.set()
        time.sleep(seconds)

    server._event_loop.call_soon_threadsafe(_park)
    # Wait for the block to actually land on the loop thread.
    assert started.wait(timeout=5.0), "loop never scheduled the blocker"
    return started


def test_handler_returns_503_when_loop_is_blocked(running_server):
    """A fully-blocked event loop should surface to the client as HTTP 503
    within the configured server-side timeout, not as a hung connection."""
    server, port = running_server
    # Block the loop for 60 s.  The get_process_async handler uses
    # timeout + RUN_ASYNC_HEADROOM (=30 s) as its _run_async deadline, and
    # the inner coro's own timeout is what the caller passes as
    # ``timeout`` in the request body.  We pass timeout=1.0 so the total
    # server-side deadline is ~31 s - well under the 60 s loop block.
    _block_loop(server, seconds=60.0)

    # Allow a socket timeout generous enough to catch a real 503 but short
    # enough that we fail fast if the fix regresses.
    t0 = time.monotonic()
    status, body = _post_json(port, "/process/get", {"blocking": True, "timeout": 1.0}, timeout=90.0)
    elapsed = time.monotonic() - t0
    assert status == 503, f"expected 503 on stuck loop, got {status} body={body}"
    # Client-visible latency should match the server-side deadline
    # (1.0 + RUN_ASYNC_HEADROOM=30.0), not the full 60 s loop block.
    assert elapsed < 1.0 + RUN_ASYNC_HEADROOM + 5.0, (
        f"handler took {elapsed:.1f}s - server didn't honor the _run_async deadline"
    )


def test_status_still_responds_while_loop_slow(running_server):
    """/status does not cross into the event loop, so it must stay
    responsive even when the loop is blocked.  This is a regression guard
    for future refactors that might accidentally route /status through
    _run_async."""
    server, port = running_server
    _block_loop(server, seconds=10.0)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        conn.request("GET", "/status")
        resp = conn.getresponse()
        assert resp.status == 200
    finally:
        conn.close()


def test_loop_watchdog_logs_and_dumps_when_loop_stalls(running_server, caplog):
    """The watchdog must log a 'Loop watchdog' error (and trigger
    faulthandler.dump_traceback) when the loop is unresponsive for longer
    than LOOP_WATCHDOG_THRESHOLD. Without this, a real production stall
    leaves only the 40s aftermath log line - by which point SIGUSR1 sees an
    idle process and the actual culprit's stack is gone."""
    server, _port = running_server
    caplog.set_level("ERROR", logger="LeanServer")
    # Block well past the threshold so the watchdog has time to notice.
    _block_loop(server, seconds=LOOP_WATCHDOG_THRESHOLD + 3.0)
    # Watchdog probes every 1s; allow a generous margin for it to land.
    deadline = time.monotonic() + LOOP_WATCHDOG_THRESHOLD + 8.0
    while time.monotonic() < deadline:
        if any("Loop watchdog" in r.message for r in caplog.records):
            return
        time.sleep(0.2)
    pytest.fail(
        f"watchdog did not log within {LOOP_WATCHDOG_THRESHOLD + 8.0:.0f}s; "
        f"records={[r.message for r in caplog.records]}"
    )


def test_loop_watchdog_quiet_when_loop_healthy(running_server, caplog):
    """Regression guard: a responsive loop must NOT trip the watchdog."""
    server, _port = running_server
    caplog.set_level("ERROR", logger="LeanServer")
    # Sleep longer than the threshold so we'd catch a misfire, but don't
    # block the loop - it should keep servicing watchdog probes.
    time.sleep(LOOP_WATCHDOG_THRESHOLD + 2.0)
    misfires = [r.message for r in caplog.records if "Loop watchdog" in r.message]
    assert not misfires, f"watchdog misfired on healthy loop: {misfires}"


def test_run_async_cancels_coroutine_on_timeout(running_server):
    """When ``_run_async``'s outer wait expires, the underlying coroutine
    must be cancelled - not left running on the loop. Without this, a
    slow ``_spawn_async`` (parallel Mathlib import under recycle pressure)
    silently completes after the handler returned 503, publishing a
    checked_out entry that nobody owns. That orphan eats a pool slot
    until the lease reaper reclaims it ten minutes later - the cascade
    that motivated this fix during long RL training runs.
    """
    server, port = running_server
    cancelled = threading.Event()
    started = threading.Event()

    async def parking_acquire(timeout=None):
        started.set()
        try:
            await asyncio.Event().wait()  # park forever
        except asyncio.CancelledError:
            cancelled.set()
            raise

    server.pool.acquire_async = parking_acquire

    # timeout=1.0 -> server-side _run_async deadline = 1.0 + 30 = 31s.
    status, _ = _post_json(
        port, "/process/get", {"blocking": True, "timeout": 1.0}, timeout=90.0
    )
    assert status == 503, f"expected 503 on stuck acquire, got {status}"
    assert started.is_set(), "parking_acquire never started on the loop"
    # Cancellation propagates asynchronously after future.cancel(); give
    # the loop a moment to deliver CancelledError.
    assert cancelled.wait(timeout=5.0), (
        "coroutine was not cancelled after _run_async timeout - "
        "_spawn_async would leak orphan placeholders"
    )


if __name__ == "__main__":
    pool = _FakePool()
    server = LeanServer(pool, address="127.0.0.1", port=0, log_level="ERROR")
    server.start()
    port = server.server.server_address[1]
    try:
        test_handler_returns_503_when_loop_is_blocked((server, port))
        test_status_still_responds_while_loop_slow((server, port))
        print("OK: stuck loop -> HTTP 503, /status stays up")
    finally:
        server.stop()
