import asyncio
import concurrent.futures
import contextlib
import faulthandler
import json
import logging
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Self
import urllib.request
import urllib.error

import psutil

from leantree.repl_adapter.interaction import LeanProcessException, LeanProofBranch, LeanInteractionException
from leantree.repl_adapter.process_pool import LeanProcessPool, PoolEntry
from leantree.core.lean import LeanProofState, LeanTactic, LeanGoal
from leantree.utils import serialize_exception, deserialize_exception, ValueOrError


# Hard deadlines for every cross-thread await-on-the-event-loop call.
# Without these, one stuck coroutine can block an HTTP handler thread forever,
# queueing every subsequent request behind it until the server appears dead
# (observed in production, see plan `the-fix-2-is-streamed-valiant.md`).
# These wrap the *inner* async operation's own timeout with a small headroom,
# so the server-side `_run_async` never waits longer than the coroutine itself
# promised to take.
RUN_ASYNC_HEADROOM = 30.0  # seconds - wraps caller-supplied operation timeouts
POOL_SHUTDOWN_TIMEOUT = 30.0

# Default per-operation deadlines used when the caller didn't specify one.
# Each of these is an inner (coro-level) deadline; add RUN_ASYNC_HEADROOM
# when passing to _run_async.
DEFAULT_COMMAND_TIMEOUT = 300.0  # matches _send_to_repl_async default
DEFAULT_PROOF_FROM_SORRY_TIMEOUT = 300.0
DEFAULT_IS_VALID_SOURCE_TIMEOUT = 60.0
DEFAULT_TRY_APPLY_TACTIC_MS = 1000  # matches existing handler default
DEFAULT_RETURN_TIMEOUT = 30.0  # PSS read + drain + rollback; bounded internally

# Loop watchdog. A background thread schedules a no-op callback on the event
# loop every INTERVAL seconds; if the loop fails to run it within THRESHOLD,
# we dump every thread's traceback to stderr and log an error. COOLDOWN keeps
# a sustained stall from spamming dumps. Threshold is set well below the
# RUN_ASYNC_HEADROOM-driven 40s `loop is unresponsive` line so the dump
# captures the loop *while* it's stuck, not after the request handlers have
# already given up and exited.
LOOP_WATCHDOG_INTERVAL = 1.0
LOOP_WATCHDOG_THRESHOLD = 5.0
LOOP_WATCHDOG_COOLDOWN = 60.0

# How often to drop branch tracking for entries that have been recycled out
# of the pool. Branches outlive a single tactic but their associated process
# may be recycled; without periodic cleanup, _branches grows unbounded.
BRANCH_RECONCILER_INTERVAL = 60.0


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a separate thread."""
    daemon_threads = True
    # Increase listen backlog for high-load scenarios (default is usually 5)
    request_queue_size = 128


class LeanServer:
    """Manages a LeanProcessPool and exposes it over a HTTP port.

    The server keeps no process registry of its own: HTTP-facing process IDs
    are pool entry IDs, and ``LeanProcessPool`` is the single source of truth
    for which subprocesses exist and what state they're in. The classes that
    used to coordinate four pieces of bookkeeping (``_ProcessRegistry``,
    ``_destroy_process``, ``_destroy_untracked_process``,
    ``_reap_leaked_processes``) have all been removed - the pool's idempotent
    ``release`` and its built-in idle/liveness workers absorb their roles.
    """

    def __init__(self, pool: LeanProcessPool, address: str = "localhost", port: int = 8000, log_level: str = "INFO"):
        self.pool = pool
        self.address = address
        self.port = port
        self.log_level = log_level
        self.server = None
        self.server_thread = None

        # Root handler + formatter come from leantree.utils.setup_default_logging;
        # just lower this named logger's threshold if requested.
        self.logger = logging.getLogger("LeanServer")
        self.logger.setLevel(log_level)

        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

        # Warmup gating. When a warmup_coro is passed to start(), this flips
        # to True only after it completes; /process/get rejects with 503 until
        # then. Default True so servers started without warmup immediately
        # accept requests.
        self._warmup_complete: bool = True

        # Branch tracking. Keys are (entry_id, branch_id). Outlives a single
        # tactic application; reconciled against pool._live every
        # BRANCH_RECONCILER_INTERVAL seconds so recycled processes don't leak
        # branch entries.
        self._branch_id_counter = 0
        self._branches: dict[tuple[int, int], LeanProofBranch] = {}
        self._branch_last_used: dict[tuple[int, int], float] = {}
        self._branches_lock = threading.Lock()
        self._branch_reconciler_thread: threading.Thread | None = None
        self._branch_reconciler_stop = threading.Event()

        # Request tracking for monitoring.
        self._active_requests: dict[int, dict] = {}
        self._request_id_counter = 0
        self._requests_lock = threading.Lock()

        # Event-loop watchdog: see _run_loop_watchdog and LOOP_WATCHDOG_*.
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

    # ------------------------------------------------------------------
    # Branch tracking
    # ------------------------------------------------------------------

    def _register_branch(self, entry_id: int, branch: LeanProofBranch) -> int:
        with self._branches_lock:
            self._branch_id_counter += 1
            branch_id = self._branch_id_counter
            key = (entry_id, branch_id)
            self._branches[key] = branch
            self._branch_last_used[key] = time.time()
            return branch_id

    def _get_branch(self, entry_id: int, branch_id: int) -> LeanProofBranch:
        with self._branches_lock:
            key = (entry_id, branch_id)
            if key not in self._branches:
                raise ValueError(f"Branch {branch_id} not found for process {entry_id}")
            self._branch_last_used[key] = time.time()
            branch = self._branches[key]
        # Refresh the entry's last_used too - otherwise a worker that only
        # hits try_apply_tactic / branch/state (never /process/get) could see
        # its entry reaped out from under it by the pool's idle reaper.
        entry = self.pool.get_entry(entry_id)
        if entry is not None:
            self.pool.touch(entry)
        return branch

    def _remove_branches_for_entry(self, entry_id: int) -> None:
        with self._branches_lock:
            keys_to_remove = [k for k in self._branches if k[0] == entry_id]
            for key in keys_to_remove:
                del self._branches[key]
                self._branch_last_used.pop(key, None)

    def _branch_reconciler_loop(self) -> None:
        """Drop branch tracking for entries that have been recycled out of
        the pool. Replaces the implicit cleanup that used to happen via
        ``_remove_process``: the pool now owns process lifecycle, and the
        server's branch table is downstream state that has to be reconciled.
        """
        while not self._branch_reconciler_stop.wait(BRANCH_RECONCILER_INTERVAL):
            with self._branches_lock:
                stale_entry_ids = {
                    eid for (eid, _) in self._branches
                    if self.pool.get_entry(eid) is None
                }
            if not stale_entry_ids:
                continue
            self.logger.info(
                f"Branch reconciler: dropping branches for "
                f"{len(stale_entry_ids)} recycled entries"
            )
            for eid in stale_entry_ids:
                self._remove_branches_for_entry(eid)

    # ------------------------------------------------------------------
    # Async dispatch helpers
    # ------------------------------------------------------------------

    def _run_async(self, coro, timeout: float | None = None):
        """Run an async coroutine on the server's event loop.

        Args:
            coro: The coroutine to run.
            timeout: Raw wait-for-future timeout in seconds. Use this when
                the coroutine does NOT have its own inner deadline. For
                coroutines that DO have one, prefer ``_run_async_op`` which
                adds the ``RUN_ASYNC_HEADROOM`` for you.
        """
        if self._event_loop is None:
            raise RuntimeError("Server not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # Cancel the in-flight coroutine so it doesn't silently complete
            # after the handler has already given up - otherwise a slow
            # _spawn_async (parallel Mathlib import under recycle pressure)
            # publishes a checked_out entry that nobody owns, eating a pool
            # slot until the lease reaper reclaims it ten minutes later. The
            # cancellation propagates through acquire_async / _spawn_async's
            # BaseException handlers, which already recycle the placeholder.
            future.cancel()
            raise

    def _run_async_op(self, coro, op_timeout: float):
        """Run a coroutine that has its own inner deadline (``op_timeout``).

        Adds ``RUN_ASYNC_HEADROOM`` so the coroutine has a chance to enforce
        its own deadline (and emit a meaningful ``LeanProcessException``)
        before the outer threadsafe wait gives up with a generic
        TimeoutError. Forgetting this headroom used to be a latent-deadlock
        footgun - the outer wait would fire first, cancel observation of
        the coroutine, and the inner operation would never surface its real
        error.
        """
        return self._run_async(coro, timeout=op_timeout + RUN_ASYNC_HEADROOM)

    # ------------------------------------------------------------------
    # Request tracking
    # ------------------------------------------------------------------

    def _start_request(self, path: str) -> int:
        with self._requests_lock:
            self._request_id_counter += 1
            request_id = self._request_id_counter
            self._active_requests[request_id] = {
                "path": path,
                "start_time": time.time(),
                "thread_name": threading.current_thread().name,
            }
            return request_id

    def _end_request(self, request_id: int):
        with self._requests_lock:
            self._active_requests.pop(request_id, None)

    def get_active_requests(self) -> list[dict]:
        """Get list of currently active requests with their duration."""
        now = time.time()
        with self._requests_lock:
            return [
                {
                    "request_id": rid,
                    "path": info["path"],
                    "duration_seconds": round(now - info["start_time"], 2),
                    "thread": info["thread_name"],
                }
                for rid, info in self._active_requests.items()
            ]

    def _has_active_request_for_entry(self, entry_id: int) -> bool:
        """Hook installed on ``pool.has_active_request`` so the pool's idle
        reaper doesn't reclaim entries with in-flight HTTP requests."""
        prefix = f"/process/{entry_id}/"
        with self._requests_lock:
            return any(prefix in req["path"] for req in self._active_requests.values())

    # ------------------------------------------------------------------
    # Loop watchdog
    # ------------------------------------------------------------------

    def _run_loop_watchdog(self):
        """Detect event-loop stalls and dump all-thread tracebacks while
        stuck. Catches the case where ``_run_async_op`` callers eventually
        log ``loop is unresponsive`` after their 40s wait expires - by then
        the stall is usually over and SIGUSR1 sees an idle process. This
        watchdog captures the stack *during* the stall.
        """
        last_dump = 0.0
        while not self._watchdog_stop.is_set():
            loop = self._event_loop
            if loop is None or loop.is_closed():
                return
            scheduled_at = time.monotonic()
            ran = threading.Event()
            try:
                loop.call_soon_threadsafe(ran.set)
            except RuntimeError:
                return  # loop already closed
            if not ran.wait(LOOP_WATCHDOG_THRESHOLD):
                stall_so_far = time.monotonic() - scheduled_at
                now = time.monotonic()
                if now - last_dump >= LOOP_WATCHDOG_COOLDOWN:
                    self.logger.error(
                        f"Loop watchdog: probe scheduled but not run within "
                        f"{stall_so_far:.1f}s (threshold {LOOP_WATCHDOG_THRESHOLD}s) - "
                        f"dumping all-thread tracebacks to stderr"
                    )
                    try:
                        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    except Exception as e:
                        self.logger.error(f"Loop watchdog: faulthandler.dump_traceback failed: {e}")
                    last_dump = now
                if not ran.wait(LOOP_WATCHDOG_COOLDOWN):
                    self.logger.error(
                        f"Loop watchdog: probe still not run after "
                        f"{LOOP_WATCHDOG_THRESHOLD + LOOP_WATCHDOG_COOLDOWN:.0f}s; "
                        f"continuing to poll"
                    )
            if self._watchdog_stop.wait(LOOP_WATCHDOG_INTERVAL):
                return

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, warmup_coro=None):
        """Start the HTTP server.

        If ``warmup_coro`` is given, the HTTP listener comes up immediately
        (so ``/status`` is queryable during warmup) but ``/process/get``
        returns 503 until the coroutine completes. This prevents clients
        from racing warmup and triggering extra spawns on top of the warmup
        N, while keeping the server introspectable.
        """
        if self.server is not None:
            raise RuntimeError("Server already started")

        # Start event loop in a separate thread.
        def run_event_loop():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()

        self._loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self._loop_thread.start()

        # Wait for event loop to be ready.
        while self._event_loop is None:
            time.sleep(0.01)

        # Start the loop watchdog as soon as the loop is up - it's most
        # useful during warmup/heavy startup, when stalls are most likely.
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._run_loop_watchdog, daemon=True, name="LeanServer-loop-watchdog"
        )
        self._watchdog_thread.start()

        # Bind the pool to our event loop and install the active-request
        # hook so its idle reaper can defer to in-flight HTTP requests.
        self.pool.has_active_request = self._has_active_request_for_entry
        self.pool.start(self._event_loop)

        if warmup_coro is not None:
            self._warmup_complete = False

        handler = self._create_handler()
        self.server = ThreadingHTTPServer((self.address, self.port), handler)

        def run_server():
            self.server.serve_forever()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Run warmup after the HTTP listener is up so /status stays
        # reachable while it runs. Block start() until it completes so
        # callers can log "ready" in the obvious order.
        if warmup_coro is not None:
            try:
                self._run_async(warmup_coro)
            finally:
                self._warmup_complete = True

        # Branch reconciler watches for branches whose pool entry has been
        # recycled and drops them.
        self._branch_reconciler_stop.clear()
        self._branch_reconciler_thread = threading.Thread(
            target=self._branch_reconciler_loop,
            daemon=True,
            name="LeanServer-branch-reconciler",
        )
        self._branch_reconciler_thread.start()

    def stop(self):
        """Stop the HTTP server and all Lean processes."""
        # Stop the watchdog before tearing the loop down, so a probe
        # scheduled mid-shutdown doesn't trip a false "loop stalled" dump.
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=LOOP_WATCHDOG_INTERVAL + 1.0)
            self._watchdog_thread = None

        # Stop the branch reconciler.
        self._branch_reconciler_stop.set()
        if self._branch_reconciler_thread is not None:
            self._branch_reconciler_thread.join(timeout=BRANCH_RECONCILER_INTERVAL + 1.0)
            self._branch_reconciler_thread = None

        # Pool shutdown: tears down all subprocesses (idle, checked-out,
        # in-flight stopping). Uses its own internal deadline.
        try:
            self.pool.shutdown()
        except Exception as e:
            self.logger.warning(f"Error shutting down pool: {e}")

        if self.server is not None:
            self.server.shutdown()
            self.server = None
        if self._event_loop is not None:
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            self._event_loop = None

    # ------------------------------------------------------------------
    # Handler factory
    # ------------------------------------------------------------------

    def _create_handler(self):
        server = self

        class LeanServerHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.logger.debug(f"GET request to {self.path} from {self.client_address}")
                request_id = server._start_request(self.path)
                try:
                    if self.path == "/status":
                        self._handle_status()
                    elif self.path == "/heap":
                        self._handle_heap()
                    else:
                        self._send_error(404, "Not Found")
                except BrokenPipeError:
                    server.logger.debug(f"Client {self.client_address} disconnected during GET {self.path}")
                finally:
                    server._end_request(request_id)

            def do_POST(self):
                server.logger.debug(f"POST request to {self.path} from {self.client_address}")
                request_id = server._start_request(self.path)
                try:
                    self._do_POST_inner()
                except BrokenPipeError:
                    server.logger.debug(f"Client {self.client_address} disconnected during POST {self.path}")
                finally:
                    server._end_request(request_id)

            def _do_POST_inner(self):
                if self.path == "/process/get":
                    self._handle_get_process()
                elif self.path.startswith("/process/"):
                    parts = self.path.split("/")
                    if len(parts) >= 4:
                        entry_id = int(parts[2])
                        action = parts[3]
                        if action == "command":
                            self._handle_command(entry_id)
                        elif action == "return":
                            self._handle_return_process(entry_id)
                        elif action == "is_valid_source":
                            self._handle_is_valid_source(entry_id)
                        elif action == "proof_from_sorry":
                            self._handle_proof_from_sorry(entry_id)
                        elif action == "branch" and len(parts) >= 6:
                            branch_id = int(parts[4])
                            if parts[5] == "try_apply_tactic":
                                self._handle_try_apply_tactic(entry_id, branch_id)
                            elif parts[5] == "state":
                                self._handle_branch_state(entry_id, branch_id)
                            else:
                                self._send_error(404, "Not Found")
                        else:
                            self._send_error(404, "Not Found")
                    else:
                        self._send_error(404, "Not Found")
                else:
                    self._send_error(404, "Not Found")

            def _resolve_entry(self, entry_id: int) -> PoolEntry | None:
                """Look up an entry; on miss, send 404 and return None."""
                entry = server.pool.get_entry(entry_id)
                if entry is None:
                    self._send_error(404, f"Process {entry_id} not found")
                    return None
                return entry

            @contextlib.contextmanager
            def _process_op_errors(self, entry: PoolEntry, op_name: str, op_timeout: float):
                """Uniform error handling for handlers that run a per-process
                coroutine. On ``concurrent.futures.TimeoutError`` or
                ``LeanProcessException`` the entry is recycled (one call to
                the pool's idempotent ``release(recycle=True)``) and the
                appropriate HTTP error is sent. On any other ``Exception``
                the entry is left alone (failure may be unrelated to the
                Lean subprocess - e.g. malformed request JSON) and a generic
                500 is sent.
                """
                total_timeout = op_timeout + RUN_ASYNC_HEADROOM
                try:
                    yield
                except concurrent.futures.TimeoutError:
                    server.logger.error(
                        f"{op_name} on entry {entry.id} did not complete in "
                        f"{total_timeout}s - recycling entry"
                    )
                    server.pool.release(
                        entry,
                        recycle=True,
                        reason=f"{op_name} exceeded {total_timeout}s deadline",
                    )
                    server._remove_branches_for_entry(entry.id)
                    self._send_error(503, "leanserver event loop did not respond in time")
                except LeanProcessException as e:
                    server.pool.release(entry, recycle=True, reason=f"{op_name}: {e}")
                    server._remove_branches_for_entry(entry.id)
                    self._send_error(500, str(e), exception=e)
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            # ----- handlers -----

            def _handle_heap(self):
                """Live Python-object census for diagnosing memory leaks."""
                import gc
                import sys
                from collections import Counter

                gc.collect()
                objs = gc.get_objects()
                type_counts: Counter[str] = Counter()
                type_sizes: Counter[str] = Counter()
                for o in objs:
                    tn = type(o).__name__
                    type_counts[tn] += 1
                    try:
                        type_sizes[tn] += sys.getsizeof(o)
                    except (TypeError, AttributeError):
                        pass
                try:
                    rss_bytes = psutil.Process().memory_info().rss
                except Exception:
                    rss_bytes = None
                stats = server.pool.stats()
                self._send_json(200, {
                    "total_objects": len(objs),
                    "rss_bytes": rss_bytes,
                    "top_by_count": type_counts.most_common(30),
                    "top_by_size_bytes": type_sizes.most_common(30),
                    "server_branches": len(server._branches),
                    "pool_total_processes": stats.total_processes,
                    "pool_used_processes": stats.used_processes,
                    "pool_available_processes": stats.available_processes,
                    "pool_stopping_processes": stats.stopping_processes,
                })

            def _handle_status(self):
                pool = server.pool

                cpu_percent_per_core = psutil.cpu_percent(percpu=True)
                memory = psutil.virtual_memory()
                try:
                    self_rss_bytes = psutil.Process().memory_info().rss
                except Exception:
                    self_rss_bytes = None

                stats = pool.stats()

                active_requests = [
                    req for req in server.get_active_requests()
                    if req["path"] != "/status"
                ]

                # Idle-too-long counter: pool entries whose `last_used` is
                # more than 60s ago.  Useful as an early-warning signal for
                # client leaks (checked_out + idle_too_long_60s climbing
                # together = actors holding processes without working
                # them).  The hard idle reaper kicks in at ``lease_timeout``
                # (default 600s), so this 60s view is intentionally noisier
                # than the reclaim threshold.
                now_mono = time.monotonic()
                inactive_threshold = 60.0
                idle_too_long = 0
                with pool._lock:
                    for e in pool._live.values():
                        if now_mono - e.last_used > inactive_threshold:
                            idle_too_long += 1

                now = time.time()
                with server._branches_lock:
                    inactive_branches_60s = sum(
                        1 for last_used in server._branch_last_used.values()
                        if now - last_used > inactive_threshold
                    )
                    total_branches = len(server._branch_last_used)

                status = {
                    "status": "ready" if server._warmup_complete else "warming_up",

                    # Pool state.  Invariant:
                    #   total_processes ==
                    #       used + available + starting + stopping
                    # ``used_processes`` keeps its pre-redesign meaning
                    # (process held by an actor) so existing dashboards
                    # need no rewrite.  ``stopping_processes`` is the
                    # janitor backlog (>0 transient is normal after a
                    # recycle; persistently >0 means subprocess teardown
                    # is wedged - the most useful new operational signal).
                    # ``total_processes`` is the new alive-in-pool sum.
                    "max_processes": stats.max_processes,
                    "total_processes": stats.total_processes,
                    "used_processes": stats.used_processes,
                    "available_processes": stats.available_processes,
                    "starting_processes": stats.starting_processes,
                    "stopping_processes": stats.stopping_processes,
                    # Pool entries (any state) untouched for >60s.  Client-
                    # leak early warning.  Replaces the old, less
                    # specifically-named ``inactive_processes``.
                    "idle_too_long_60s": idle_too_long,

                    # Server-level branch tracking.
                    "total_branches": total_branches,
                    "inactive_branches_60s": inactive_branches_60s,

                    # Request and host telemetry.
                    "active_requests": active_requests,
                    "cpu_percent_per_core": cpu_percent_per_core,
                    "ram": {
                        "total_bytes": memory.total,
                        "available_bytes": memory.available,
                        "used_bytes": memory.used,
                        "percent": memory.percent,
                    },
                    "leanserver_rss_bytes": self_rss_bytes,
                }
                self._send_json(200, status)

            def _handle_get_process(self):
                # Reject while warmup is running. Otherwise these requests
                # would race warmup and pile spawns on top of the warmup N.
                if not server._warmup_complete:
                    self._send_error(503, "server is warming up")
                    return

                data = self._read_json()
                blocking = data.get("blocking", True)
                # Default timeout of 1 minute to prevent indefinite blocking.
                timeout = data.get("timeout", 60.0)
                # Non-blocking semantics: timeout=0 means "return immediately
                # if no capacity". The pool's acquire treats timeout=None as
                # unbounded wait, so map blocking=False -> timeout=0.
                if not blocking:
                    timeout = 0.0

                entry = None
                try:
                    entry = server._run_async_op(
                        server.pool.acquire_async(timeout=timeout),
                        op_timeout=timeout,
                    )
                    if entry is None:
                        self._send_json(200, {"process_id": None})
                    else:
                        self._send_json(200, {"process_id": entry.id})
                except concurrent.futures.TimeoutError:
                    server.logger.error(
                        f"acquire_async exceeded {timeout + RUN_ASYNC_HEADROOM}s on the event loop - "
                        f"loop is unresponsive (send SIGUSR1 for a thread dump)"
                    )
                    self._send_error(503, "leanserver event loop is unresponsive")
                except BrokenPipeError:
                    # Client disconnected. Recycle the entry if we got one;
                    # the pool's release is idempotent and synchronous so
                    # this can't deadlock or leak.
                    if entry is not None:
                        server.logger.warning(
                            f"Client disconnected during /process/get, recycling entry {entry.id}"
                        )
                        server.pool.release(entry, recycle=True, reason="client disconnect during get")
                    raise  # let the server handle the broken connection
                except Exception as e:
                    if entry is not None:
                        server.pool.release(
                            entry,
                            recycle=True,
                            reason=f"get_process handler raised {type(e).__name__}: {e}",
                        )
                    self._send_error(500, str(e), exception=e)

            def _handle_return_process(self, entry_id: int):
                """Client-driven return path. Idempotent at the entry level
                (the pool's ``release`` and ``return_entry_async`` both no-op
                if the entry has already been removed from ``_live``)."""
                entry = server.pool.get_entry(entry_id)
                if entry is None:
                    # Already returned - this is idempotent on the wire.
                    self._send_json(200, {"status": "ok", "already_returned": True})
                    return
                # Snapshot the branch removal up-front - the entry's REPL is
                # about to be rolled back to its checkpoint, so any branch
                # state tied to it is invalid regardless of the return path.
                server._remove_branches_for_entry(entry_id)
                try:
                    server._run_async(
                        server.pool.return_entry_async(entry),
                        timeout=DEFAULT_RETURN_TIMEOUT + RUN_ASYNC_HEADROOM,
                    )
                except concurrent.futures.TimeoutError:
                    server.logger.error(
                        f"return_entry_async for entry {entry_id} exceeded "
                        f"{DEFAULT_RETURN_TIMEOUT + RUN_ASYNC_HEADROOM}s - recycling"
                    )
                    server.pool.release(
                        entry,
                        recycle=True,
                        reason=f"return_entry_async timed out for entry_id={entry_id}",
                    )
                    self._send_error(503, "leanserver event loop did not respond in time")
                    return
                except Exception as e:
                    server.pool.release(
                        entry,
                        recycle=True,
                        reason=f"return_entry_async raised {type(e).__name__}: {e}",
                    )
                    self._send_error(500, str(e), exception=e)
                    return
                self._send_json(200, {"status": "ok"})

            def _handle_command(self, entry_id: int):
                entry = self._resolve_entry(entry_id)
                if entry is None:
                    return
                with self._process_op_errors(entry, "send_command_async", DEFAULT_COMMAND_TIMEOUT):
                    command = self._read_json()["command"]
                    response = server._run_async_op(
                        entry.process.send_command_async(command),
                        op_timeout=DEFAULT_COMMAND_TIMEOUT,
                    )
                    self._send_json(200, response)

            def _handle_is_valid_source(self, entry_id: int):
                entry = self._resolve_entry(entry_id)
                if entry is None:
                    return
                with self._process_op_errors(entry, "is_valid_source_async", DEFAULT_IS_VALID_SOURCE_TIMEOUT):
                    source = self._read_json()["source"]
                    is_valid = server._run_async_op(
                        entry.process.is_valid_source_async(source),
                        op_timeout=DEFAULT_IS_VALID_SOURCE_TIMEOUT,
                    )
                    self._send_json(200, {"is_valid": is_valid})

            def _handle_proof_from_sorry(self, entry_id: int):
                entry = self._resolve_entry(entry_id)
                if entry is None:
                    return
                with self._process_op_errors(entry, "proof_from_sorry", DEFAULT_PROOF_FROM_SORRY_TIMEOUT):
                    theorem_with_sorry = self._read_json()["theorem_with_sorry"]

                    async def collect_proof_branches():
                        return [
                            branch
                            async for branch in entry.process.proofs_from_sorries_async(theorem_with_sorry)
                        ]

                    # LeanInteractionException is a business-level error
                    # (bad theorem input, not a process failure) - return
                    # 200 with an error payload rather than recycling.
                    # Intercept before it escapes to the context manager.
                    try:
                        proof_branches = server._run_async_op(
                            collect_proof_branches(),
                            op_timeout=DEFAULT_PROOF_FROM_SORRY_TIMEOUT,
                        )
                    except LeanInteractionException as e:
                        self._send_json(200, {"error": str(e)})
                        return

                    if len(proof_branches) == 0:
                        self._send_json(200, {"error": "No sorries found in theorem"})
                        return
                    if len(proof_branches) > 1:
                        self._send_json(200, {"error": f"Expected 1 sorry, found {len(proof_branches)}"})
                        return

                    proof_branch = proof_branches[0]
                    branch_id = server._register_branch(entry_id, proof_branch)
                    goals = [goal.serialize() for goal in proof_branch.state.goals]
                    self._send_json(200, {"value": {"branch_id": branch_id, "goals": goals}})

            def _handle_try_apply_tactic(self, entry_id: int, branch_id: int):
                entry = self._resolve_entry(entry_id)
                if entry is None:
                    return
                data = self._read_json()
                tactic = data["tactic"]
                tactic_timeout_ms = data.get("timeout", DEFAULT_TRY_APPLY_TACTIC_MS)
                op_timeout_s = tactic_timeout_ms / 1000.0
                op_name = f"try_apply_tactic (branch {branch_id}, tactic={tactic!r})"
                with self._process_op_errors(entry, op_name, op_timeout_s):
                    branch = server._get_branch(entry_id, branch_id)
                    result = server._run_async_op(
                        branch.try_apply_tactic_async(tactic, timeout=tactic_timeout_ms),
                        op_timeout=op_timeout_s,
                    )
                    if not result.is_success():
                        self._send_json(200, {"error": str(result.error)})
                        return
                    new_branches = result.value
                    branches_data = []
                    for new_branch in new_branches:
                        new_branch_id = server._register_branch(entry_id, new_branch)
                        branches_data.append({
                            "branch_id": new_branch_id,
                            "goals": [goal.serialize() for goal in new_branch.state.goals],
                        })
                    self._send_json(200, {"value": branches_data})

            def _handle_branch_state(self, entry_id: int, branch_id: int):
                try:
                    branch = server._get_branch(entry_id, branch_id)
                    self._send_json(200, {
                        "branch_id": branch_id,
                        "goals": [goal.serialize() for goal in branch.state.goals],
                        "is_solved": branch.is_solved,
                    })
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            # ----- I/O helpers -----

            def _read_json(self) -> dict:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                return json.loads(body.decode("utf-8"))

            def _send_json(self, status_code: int, data: dict):
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))

            def _send_error(self, status_code: int, message: str, exception: Exception = None):
                error_data = {"error": message}
                if exception is not None:
                    exception_data = serialize_exception(exception)
                    error_data.update(exception_data)
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(error_data).encode("utf-8"))

            def log_message(self, format, *args):
                # Suppress default access logging.
                pass

        return LeanServerHandler


# ----------------------------------------------------------------------
# Client side (talks HTTP; unaffected by pool internals)
# ----------------------------------------------------------------------


class LeanClient:
    """Connects to a LeanServer."""

    def __init__(self, address: str, port: int):
        self.address = address
        self.port = port
        self.base_url = f"http://{address}:{port}"

    def _request(self, method: str, path: str, data: dict = None, timeout: float | None = 360) -> dict:
        """Make an HTTP request to the server.

        Args:
            timeout: Socket timeout in seconds.  Defaults to 360s (slightly
                     above the server-side 300s REPL timeout so the server
                     has a chance to respond with an error first).  Pass
                     ``None`` to wait indefinitely.
        """
        url = f"{self.base_url}{path}"
        if data is not None:
            data_bytes = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=data_bytes, method=method)
            req.add_header("Content-Type", "application/json")
        else:
            req = urllib.request.Request(url, method=method)
        try:
            kwargs = {"timeout": timeout} if timeout is not None else {}
            with urllib.request.urlopen(req, **kwargs) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_data = json.loads(error_body)
                error_message = error_data.get("error", str(e))
                raise deserialize_exception(error_data, f"Error from LeanServer: {error_message}")
            except json.JSONDecodeError:
                raise RuntimeError(f"Error from LeanServer: {str(e)}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"LeanServer request to {path} failed: {e}") from e
        except TimeoutError as e:
            raise ConnectionError(f"LeanServer request to {path} timed out: {e}") from e

    def check_status(self) -> dict:
        """Check the status of the server."""
        return self._request("GET", "/status")

    def get_process(self, blocking: bool = True, timeout: float | None = 300.0) -> "LeanRemoteProcess | None":
        """Get a process from the server.

        Args:
            blocking: If True, wait until a process is available. If False,
                return None immediately if unavailable.
            timeout: Maximum time to wait for a process (in seconds). Only
                used if blocking=True. Default is 300 seconds (5 minutes).
                Set to None for no timeout.

        Returns:
            A LeanRemoteProcess if available, None if not available
            (non-blocking) or timeout expired.
        """
        data = {"blocking": blocking}
        if timeout is not None:
            data["timeout"] = timeout
        socket_timeout = timeout + 30.0 if timeout is not None else None
        response = self._request("POST", "/process/get", data, timeout=socket_timeout)
        process_id = response.get("process_id")
        if process_id is None:
            return None
        return LeanRemoteProcess(self, process_id)


class LeanRemoteProcess:
    """A remote Lean process managed by a LeanServer."""

    def __init__(self, client: LeanClient, process_id: int):
        self.client = client
        self.process_id = process_id
        self._returned = False
        self._lock = threading.Lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs):
        try:
            self.return_process()
        except Exception:
            logging.getLogger("LeanClient").warning(
                f"Failed to return process {self.process_id} in __exit__"
            )

    def __del__(self):
        if not self._returned:
            try:
                self.return_process()
            except Exception:
                pass

    def _check_not_returned(self):
        if self._returned:
            raise RuntimeError("Process has already been returned to the pool")

    def send_command(self, command: str) -> dict:
        with self._lock:
            self._check_not_returned()
            return self.client._request(
                "POST",
                f"/process/{self.process_id}/command",
                {"command": command},
                timeout=360,
            )

    def is_valid_source(self, source: str) -> bool:
        with self._lock:
            self._check_not_returned()
            response = self.client._request(
                "POST",
                f"/process/{self.process_id}/is_valid_source",
                {"source": source},
                timeout=60,
            )
            return response["is_valid"]

    def return_process(self):
        """Return the process to the pool. Safe to call multiple times.
        Retries up to 3 times to avoid leaking processes on the server when
        the network is transiently unreachable.
        """
        with self._lock:
            if self._returned:
                return
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                self.client._request("POST", f"/process/{self.process_id}/return", timeout=10)
                with self._lock:
                    self._returned = True
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        with self._lock:
            self._returned = True
        logging.getLogger("LeanClient").error(
            f"Failed to return process {self.process_id} after {max_retries} attempts: {last_error}"
        )

    def proof_from_sorry(self, theorem_with_sorry: str) -> ValueOrError["RemoteLeanProofBranch"]:
        with self._lock:
            self._check_not_returned()
            response = self.client._request(
                "POST",
                f"/process/{self.process_id}/proof_from_sorry",
                {"theorem_with_sorry": theorem_with_sorry},
                timeout=60,
            )

        if "error" in response:
            return ValueOrError.from_error(response["error"])

        value = response["value"]
        return ValueOrError.from_success(RemoteLeanProofBranch(
            self,
            value["branch_id"],
            value["goals"]
        ))


class RemoteLeanProofBranch:
    """A remote proof branch managed by a LeanServer.

    Thin client-side proxy that maps 1:1 to a LeanProofBranch on the server.
    All operations are delegated to the server-side branch.
    """

    def __init__(self, remote_process: LeanRemoteProcess, branch_id: int, goals: list[dict]):
        # Hold a reference to the LeanRemoteProcess to prevent it from being
        # garbage collected (and returning the process to the pool) while
        # this proof branch is still in use.
        self._remote_process = remote_process
        self._branch_id = branch_id
        self._goals = [LeanGoal.deserialize(g) for g in goals]

    @property
    def client(self) -> LeanClient:
        return self._remote_process.client

    @property
    def process_id(self) -> int:
        return self._remote_process.process_id

    @property
    def state(self) -> LeanProofState:
        return LeanProofState(self._goals)

    @property
    def is_solved(self) -> bool:
        return self.state.is_solved()

    def try_apply_tactic(
            self,
            tactic: LeanTactic | str,
            timeout: int | None = 1000,
    ) -> ValueOrError[list["RemoteLeanProofBranch"]]:
        """Apply a tactic to the proof branch.

        Delegates to LeanProofBranch.try_apply_tactic_async on the server.
        """
        if isinstance(tactic, LeanTactic):
            tactic = tactic.tactic

        data = {"tactic": tactic}
        if timeout is not None:
            data["timeout"] = timeout

        socket_timeout = (timeout / 1000 if timeout else 300) + 30
        response = self.client._request(
            "POST",
            f"/process/{self.process_id}/branch/{self._branch_id}/try_apply_tactic",
            data,
            timeout=socket_timeout,
        )

        if "error" in response:
            return ValueOrError.from_error(response["error"])

        branches_data = response["value"]
        branches = []
        for branch_data in branches_data:
            branches.append(RemoteLeanProofBranch(
                self._remote_process,
                branch_data["branch_id"],
                branch_data["goals"]
            ))

        return ValueOrError.from_success(branches)


def start_server(
        pool: LeanProcessPool,
        address: str = "localhost",
        port: int = 8000,
        log_level: str = "INFO",
        warmup: bool = False,
        warmup_batch_size: int | None = None,
) -> LeanServer:
    """Start a LeanServer with the given pool.

    If ``warmup`` is True, pre-start processes up to ``pool.max_processes``
    and block until they are ready *before* the HTTP server accepts requests.
    If ``warmup_batch_size`` is set, processes are started in batches of
    that size to limit I/O/memory contention during warmup.
    """
    server = LeanServer(pool, address, port, log_level)
    warmup_coro = pool.warmup_async(batch_size=warmup_batch_size) if warmup else None
    server.start(warmup_coro=warmup_coro)
    return server


if __name__ == "__main__":
    repl_exe = Path("../lean-repl/.lake/build/bin/repl")
    project_path = Path("../leantree_project")

    pool = LeanProcessPool(
        repl_exe=repl_exe,
        project_path=project_path,
        max_processes=2,
    )

    server = start_server(pool, address="localhost", port=8000)
    print(f"Server started on http://localhost:8000")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
