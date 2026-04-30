import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Literal, Self

from leantree.repl_adapter.interaction import LeanProcess, LeanEnvironmentCheckpoint
from leantree.utils import to_sync


# Pace at which the memory governor reads PSS for one process. The full
# sweep takes ``MEMORY_GOVERNOR_INTERVAL * len(_live)`` seconds; with the
# default and a 128-process pool that's a ~64 s round-robin. PSS reads are
# off the request hot path, so a slow individual read costs only the
# governor's own latency to reach the next entry, never a client request.
MEMORY_GOVERNOR_INTERVAL = 0.5

# Bound on subprocess teardown. After SIGKILL the kernel should reap within
# milliseconds, but NFS hangs / disk thrash have been seen to delay reaps.
# Cap so the janitor can't stall the rest of the queue on one wedged subprocess.
JANITOR_STOP_TIMEOUT = 5.0

# Max time shutdown() will wait for the janitor to drain `_live` before giving
# up and leaking remaining subprocesses (zombies). Long enough that a backlog
# of janitor work after a server stop usually completes; short enough that
# the host shutdown isn't blocked indefinitely.
SHUTDOWN_DRAIN_TIMEOUT = 30.0

# Cadence for the idle-lease reaper and liveness reconciler. 60 s matches the
# previous _reap_leaked_processes cadence in server.py, which has been
# production-tuned to balance reclamation latency against probe overhead.
#
# (Deadline-on-tactic kills live in interaction.py's module-level watchdog,
# not here - one watchdog covers bare LeanProcess and pool-managed alike.)
REAPER_INTERVAL = 60.0
LIVENESS_INTERVAL = 60.0


PoolEntryState = Literal["starting", "idle", "checked_out", "stopping"]


@dataclass
class PoolEntry:
    """A single managed Lean process and its lifecycle state.

    Carries everything that used to be split across the four pieces of
    bookkeeping (`_num_used_processes` int, `available_processes` deque,
    `checkpoints` dict, `_processes` registry). Membership in `pool._live`
    IS the source of truth; capacity used = `len(pool._live)`. An entry is
    removed from `_live` only after its subprocess is fully reaped, which
    is the strict-capacity guarantee.

    The dataclass is also an async context manager: an `async with` body
    exiting cleanly calls `pool.return_entry_async`, while an exception
    forces `pool.release(recycle=True)`. Direct use of `pool.acquire_async`
    + `pool.release` is also fine; release is idempotent.

    ``over_budget`` is set by the memory governor when this entry's PSS
    exceeds ``pool.pss_recycle_limit`` while the entry is checked_out. The
    return path honours the flag by recycling instead of returning to idle.
    Idle entries that cross the threshold are recycled inline by the
    governor and never have their flag inspected again.
    """
    id: int
    pool: "LeanProcessPool"
    process: LeanProcess | None
    pid: int | None
    state: PoolEntryState
    checkpoint: LeanEnvironmentCheckpoint | None = None
    last_used: float = 0.0
    over_budget: bool = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.id not in self.pool._live:
            return  # already released; idempotent
        if exc is not None:
            self.pool.release(
                self,
                recycle=True,
                reason=f"context exit raised: {exc_type.__name__}",
            )
        else:
            await self.pool.return_entry_async(self)


@dataclass
class PoolStats:
    """Snapshot of pool state for /status.  Wire DTO; field names use the
    operator-facing vocabulary (``used`` / ``available``) rather than the
    internal state-machine literals (``checked_out`` / ``idle``).  The
    ``used`` semantics match the pre-redesign field of the same name -
    "actor is currently holding this process" - so dashboards consuming
    /status need no rewrite.

    Invariant:
        total_processes ==
            available_processes + used_processes
            + starting_processes + stopping_processes

    Reading guide:
    - ``used_processes``: actors actively holding a process. Primary
      utilization signal.
    - ``available_processes``: idle, immediately acquirable.
    - ``starting_processes``: mid-spawn (warmup or on-demand).
    - ``stopping_processes``: janitor backlog. Transiently >0 after a
      recycle; persistently >0 means subprocess teardown is wedged.
    - ``total_processes``: alive in pool = capacity used. New summary
      field; sum of the four state counts.
    """
    max_processes: int
    total_processes: int
    available_processes: int
    starting_processes: int
    used_processes: int
    stopping_processes: int


class LeanProcessPool:
    """Process pool with single-source-of-truth accounting.

    The set of managed processes lives in `self._live: dict[int, PoolEntry]`.
    Capacity used is `len(self._live)`. There is no separate counter that
    can drift: every transition mutates `_live` under a `threading.RLock`.
    All accounting is synchronous and never crosses the asyncio event loop,
    so loop saturation cannot leave the pool in an inconsistent state - the
    failure mode that motivated this rewrite (production: 64 used / 49
    tracked, 15 unrecoverable slots after RL training).

    Subprocess I/O (spawn, stop, drain, PSS read) is async and runs on a
    caller-supplied event loop, but is decoupled from accounting: the
    janitor task drains `stopping` entries off the critical path and the
    deadline watchdog runs on its own thread.
    """

    def __init__(
        self,
        repl_exe: Path,
        project_path: Path,
        max_processes: int,
        env_setup_async: Callable[[LeanProcess], Coroutine] | None = None,
        logger: logging.Logger | None = None,
        rss_hard_limit: int | None = 32 * 1024 ** 3,
        pss_recycle_limit: int | None = 4 * 1024 ** 3,
        lease_timeout: float = 600.0,
        spawn_concurrency_limit: int | None = None,
    ):
        self.repl_exe = repl_exe
        self.project_path = project_path
        self.max_processes = max_processes
        self.env_setup_async = env_setup_async
        self.rss_hard_limit = rss_hard_limit
        self.pss_recycle_limit = pss_recycle_limit
        self.logger = logger if logger else logging.getLogger(__name__)

        # Cap concurrent _spawn_async calls. Without this, a recycle wave
        # (e.g. processes that all hit pss_recycle_limit on the same eval
        # boundary) triggers N simultaneous Mathlib imports which thrash
        # the disk; each spawn then runs longer than the 40s outer acquire
        # deadline, the HTTP handler returns 503, and the placeholder
        # silently completes spawning into an orphan checked_out entry
        # that only the 10-minute lease reaper can reclaim. The semaphore
        # bounds the parallelism the same way warmup batching does.
        # ``asyncio.Semaphore()`` in Python 3.10+ doesn't bind to a loop
        # at construction, so creating it in __init__ is safe.
        self._spawn_semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(spawn_concurrency_limit)
            if spawn_concurrency_limit and spawn_concurrency_limit > 0
            else None
        )

        # Single source of truth.
        self._live: dict[int, PoolEntry] = {}
        # threading primitives, NOT asyncio: callable from any thread,
        # immune to event-loop saturation. The whole point of the redesign.
        self._lock = threading.RLock()
        self._capacity_changed = threading.Condition(self._lock)
        self._id_counter = 0
        self._shutdown = False

        # Idle reaper threshold: how long a checked-out entry can sit
        # untouched before we assume the client is gone and reclaim it.
        self._lease_timeout = lease_timeout

        # Background workers. Set in start().
        self._loop: asyncio.AbstractEventLoop | None = None
        self._janitor_inbox: asyncio.Queue[PoolEntry] | None = None
        self._janitor_task: asyncio.Task | None = None
        self._workers_stop = threading.Event()
        self._idle_reaper_thread: threading.Thread | None = None
        self._liveness_thread: threading.Thread | None = None
        self._memory_governor_thread: threading.Thread | None = None

        # External hook the server installs so the idle reaper can avoid
        # reclaiming entries whose HTTP request is still in flight (e.g. a
        # long-running tactic on a checked-out process). Returns True iff
        # the entry currently has an active request. Default no-op for
        # tests / standalone usage.
        self.has_active_request: Callable[[int], bool] = lambda entry_id: False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the pool to a running asyncio loop in *another* thread
        (the leanserver pattern). For in-loop callers (pytest-asyncio,
        single-loop scripts) use ``await start_async()`` instead.
        Idempotent within a single pool lifetime.
        """
        if self._loop is not None:
            return
        # Sanity-check: callers using start() must NOT be on the target
        # loop themselves; otherwise run_coroutine_threadsafe would deadlock.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            raise RuntimeError(
                "pool.start(loop) called from within the loop itself; "
                "use `await pool.start_async()` instead"
            )

        self._loop = loop

        async def _create() -> tuple[asyncio.Queue, asyncio.Task]:
            q: asyncio.Queue = asyncio.Queue()
            task = asyncio.create_task(
                self._janitor_loop(q), name="LeanPool-janitor"
            )
            return q, task
        fut = asyncio.run_coroutine_threadsafe(_create(), loop)
        self._janitor_inbox, self._janitor_task = fut.result(timeout=5.0)

        self._start_worker_threads()

    async def start_async(self) -> None:
        """In-loop variant of ``start``. Pool binds to the currently-running
        loop and creates the janitor task directly. Idempotent.
        """
        if self._loop is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._janitor_inbox = asyncio.Queue()
        self._janitor_task = asyncio.create_task(
            self._janitor_loop(self._janitor_inbox), name="LeanPool-janitor"
        )
        self._start_worker_threads()

    def _start_worker_threads(self) -> None:
        self._idle_reaper_thread = threading.Thread(
            target=self._idle_reaper_loop,
            name="LeanPool-idle-reaper",
            daemon=True,
        )
        self._idle_reaper_thread.start()

        self._liveness_thread = threading.Thread(
            target=self._liveness_reconciler_loop,
            name="LeanPool-liveness-reconciler",
            daemon=True,
        )
        self._liveness_thread.start()

        # Memory governor: only meaningful when a recycle limit is set.
        # With pss_recycle_limit=None (tests, standalone scripts) the
        # governor would have nothing to do, so don't start the thread.
        if self.pss_recycle_limit:
            self._memory_governor_thread = threading.Thread(
                target=self._memory_governor_loop,
                name="LeanPool-memory-governor",
                daemon=True,
            )
            self._memory_governor_thread.start()

    def shutdown(self) -> None:
        """Stop all processes and join workers. Blocks until the janitor
        drains `_live` (or `SHUTDOWN_DRAIN_TIMEOUT` elapses, in which case
        remaining subprocesses leak as zombies - the kernel will reap them
        when our process exits).

        Cross-thread: this assumes the loop is running in another thread.
        For in-loop callers, use ``await shutdown_async()`` to avoid
        deadlocking the janitor task on its own loop.
        """
        entries_to_stop = self._mark_shutdown_and_collect()
        if entries_to_stop and self._janitor_inbox is not None and self._loop is not None:
            for entry in entries_to_stop:
                self._loop.call_soon_threadsafe(
                    self._janitor_inbox.put_nowait, entry
                )

        deadline = time.monotonic() + SHUTDOWN_DRAIN_TIMEOUT
        while True:
            with self._lock:
                if not self._live:
                    break
            if time.monotonic() > deadline:
                self.logger.warning(
                    f"Pool shutdown: janitor did not drain {len(self._live)} "
                    f"entries within {SHUTDOWN_DRAIN_TIMEOUT} s; leaking the rest"
                )
                break
            time.sleep(0.05)

        # Cancel the janitor task on its own loop and wait for it to
        # actually exit; otherwise the upcoming ``loop.stop()`` in the
        # server's ``stop()`` would leave the task parked on
        # ``inbox.get()`` and asyncio would log
        # "Task was destroyed but it is pending!".
        if (
            self._janitor_task is not None
            and self._loop is not None
            and not self._janitor_task.done()
        ):
            cancelled = threading.Event()
            def _cancel_and_wait():
                async def _cancel():
                    self._janitor_task.cancel()
                    try:
                        await self._janitor_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    cancelled.set()
                asyncio.ensure_future(_cancel(), loop=self._loop)
            try:
                self._loop.call_soon_threadsafe(_cancel_and_wait)
                cancelled.wait(timeout=5.0)
            except RuntimeError:
                # Loop closed already; nothing to do.
                pass

        self._join_worker_threads()

    async def shutdown_async(self) -> None:
        """In-loop variant of ``shutdown``. Awaits the janitor draining
        rather than blocking on a sync sleep, so the janitor task on the
        same loop can actually run."""
        entries_to_stop = self._mark_shutdown_and_collect()
        if entries_to_stop and self._janitor_inbox is not None:
            for entry in entries_to_stop:
                self._janitor_inbox.put_nowait(entry)

        deadline = time.monotonic() + SHUTDOWN_DRAIN_TIMEOUT
        while True:
            with self._lock:
                if not self._live:
                    break
            if time.monotonic() > deadline:
                self.logger.warning(
                    f"Pool shutdown: janitor did not drain {len(self._live)} "
                    f"entries within {SHUTDOWN_DRAIN_TIMEOUT} s; leaking the rest"
                )
                break
            await asyncio.sleep(0.05)

        # Cancel the janitor task explicitly; otherwise it would wait
        # forever for the next inbox item and leak as a pending coroutine.
        if self._janitor_task is not None and not self._janitor_task.done():
            self._janitor_task.cancel()
            try:
                await self._janitor_task
            except (asyncio.CancelledError, Exception):
                pass

        self._join_worker_threads()

    def _mark_shutdown_and_collect(self) -> list[PoolEntry]:
        """Mark the pool as shut down, wake all acquirers, and return the
        list of live entries that need teardown.  Shared by
        ``shutdown`` and ``shutdown_async``.
        """
        with self._capacity_changed:
            if self._shutdown:
                return []
            self._shutdown = True
            self._capacity_changed.notify_all()
            entries_to_stop = [
                e for e in self._live.values() if e.state != "stopping"
            ]
            for entry in entries_to_stop:
                entry.state = "stopping"
        self._workers_stop.set()
        return entries_to_stop

    def _join_worker_threads(self) -> None:
        for thread in (
            self._idle_reaper_thread,
            self._liveness_thread,
            self._memory_governor_thread,
        ):
            if thread is not None:
                thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Acquire / release (synchronous accounting; never touches the loop)
    # ------------------------------------------------------------------

    def acquire(self, timeout: float | None = None) -> PoolEntry | None:
        """Reserve capacity. Returns either an idle entry (state:
        checked_out) or a 'starting' placeholder whose subprocess the
        caller must spawn. Returns None iff timeout expired with the pool
        at capacity. Raises RuntimeError if the pool has been shut down.

        Synchronous; safe to call from any thread. Loop saturation cannot
        affect it - that's the whole reason this exists.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._capacity_changed:
            while True:
                if self._shutdown:
                    raise RuntimeError("Process pool has been shut down")
                # Reuse idle if any. Iteration order is insertion order in
                # Python 3.7+, so this is roughly LRU-by-creation.
                for entry in self._live.values():
                    if entry.state == "idle":
                        entry.state = "checked_out"
                        entry.last_used = time.monotonic()
                        return entry
                # Else: do we have capacity to start a new one?
                if len(self._live) < self.max_processes:
                    return self._new_placeholder_locked()
                # Capacity full; wait for janitor to free a slot or for
                # an in-flight return to publish an idle entry.
                if deadline is None:
                    self._capacity_changed.wait()
                else:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        return None
                    self._capacity_changed.wait(timeout=rem)

    async def acquire_async(self, timeout: float | None = None) -> PoolEntry | None:
        """Async wrapper around `acquire`. On placeholder return, spawns
        the subprocess outside any lock and transitions the entry from
        ``starting`` to ``checked_out``.  On spawn failure, recycles the
        placeholder (frees the capacity slot) and re-raises.
        """
        entry = await asyncio.to_thread(self.acquire, timeout)
        if entry is None:
            return None
        if entry.process is None:
            try:
                process = await self._spawn_async()
            except BaseException:
                self.release(entry, recycle=True, reason="spawn failed")
                raise
            with self._lock:
                entry.process = process
                entry.pid = process._proc.pid
                entry.checkpoint = process.checkpoint()
                # Promote out of "starting": this entry is now in the
                # caller's hands.  Without this, /status would report it
                # as a starting_processes count forever - including the
                # used_processes wire field, which would mis-report 0.
                entry.state = "checked_out"
                entry.last_used = time.monotonic()
        return entry

    def release(self, entry: PoolEntry, *, recycle: bool, reason: str = "") -> None:
        """Idempotent release. With ``recycle=True`` the entry is queued
        for teardown (capacity slot freed only after the subprocess is
        reaped - the strict-capacity choice). With ``recycle=False`` the
        entry returns to the idle pool for the next acquirer.

        Calling release on an already-released entry is a no-op. This is
        the property that eliminates the bookkeeping-drift class of bugs:
        every cleanup path can call ``release`` without coordinating with
        every other cleanup path.

        Synchronous; never touches the asyncio loop directly. The handoff
        to the janitor (when ``recycle=True``) is via a thread-safe
        ``call_soon_threadsafe`` onto the loop.
        """
        with self._capacity_changed:
            if entry.id not in self._live:
                return
            if recycle or self._shutdown:
                if entry.state == "stopping":
                    return  # already queued for teardown
                entry.state = "stopping"
                if reason:
                    self.logger.info(
                        f"Recycling Lean entry {entry.id} (pid={entry.pid}): {reason}"
                    )
                if self._janitor_inbox is None or self._loop is None:
                    # Pre-start path: pool was never started, so there's
                    # no janitor. Drop directly. Subprocess (if any) leaks
                    # to the OS - acceptable; this branch is for tests
                    # that use the pool without start().
                    self._live.pop(entry.id, None)
                    self._capacity_changed.notify()
                    return
                self._loop.call_soon_threadsafe(
                    self._janitor_inbox.put_nowait, entry
                )
            else:
                entry.state = "idle"
                entry.last_used = time.monotonic()
                if entry.process is not None:
                    entry.process.set_deadline(None)
                # Wake exactly one acquirer; mirroring asyncio.Semaphore's
                # FIFO release behaviour. notify_all would re-create the
                # thundering herd that the asyncio.Semaphore originally
                # fixed (see git history of process_pool.py).
                self._capacity_changed.notify()

    async def return_entry_async(self, entry: PoolEntry) -> None:
        """Client-driven return path. Drains REPL output, rolls back to
        checkpoint, and releases the entry to idle - OR recycles if the
        memory governor has flagged it as over budget.

        Memory-budget enforcement does NOT live here. The governor
        (`_memory_governor_loop`) measures PSS in the background and sets
        ``entry.over_budget`` for any process whose footprint crosses
        ``pss_recycle_limit``. Putting the read here used to form a
        thundering-herd hazard at job boundaries: 100+ simultaneous
        returns each issued an ``asyncio.to_thread`` PSS measurement,
        saturating the default 32-worker executor and timing out unrelated
        ``acquire_async`` calls (observed: second eval after a long
        collect cycle, all 88 acquires hit the 40 s outer ceiling).

        This is the only path that ever ends in ``release(recycle=False)``.
        Every brutal-recycle source (deadline watchdog, idle reaper,
        liveness reconciler, governor for over-budget idle entries,
        poisoned process from ``LeanProcessException``) goes through
        ``release(recycle=True)`` directly.
        """
        if entry.process is None:
            self.release(entry, recycle=True, reason="return of un-spawned placeholder")
            return

        if entry.over_budget:
            self.release(entry, recycle=True, reason="over PSS budget (governor flagged)")
            return

        try:
            await entry.process.drain_repl_output_async()
            if entry.checkpoint is not None:
                entry.process.rollback_to(entry.checkpoint)
        except Exception as e:
            self.release(
                entry,
                recycle=True,
                reason=f"drain/rollback raised {type(e).__name__}: {e}",
            )
            return
        self.release(entry, recycle=False)

    # ------------------------------------------------------------------
    # Per-entry attributes
    # ------------------------------------------------------------------

    def touch(self, entry: PoolEntry) -> None:
        """Refresh last_used so the idle reaper doesn't reclaim an entry
        that's actively serving requests on a long-lived branch."""
        with self._lock:
            entry.last_used = time.monotonic()

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: int) -> PoolEntry | None:
        with self._lock:
            return self._live.get(entry_id)

    def stats(self) -> PoolStats:
        with self._lock:
            counts = {"idle": 0, "starting": 0, "checked_out": 0, "stopping": 0}
            for e in self._live.values():
                counts[e.state] += 1
            # Translate state-machine names -> operator-facing names at
            # the wire boundary.  ``used`` == checked_out (matches the
            # pre-redesign meaning of /status's ``used_processes``).
            return PoolStats(
                max_processes=self.max_processes,
                total_processes=len(self._live),
                available_processes=counts["idle"],
                starting_processes=counts["starting"],
                used_processes=counts["checked_out"],
                stopping_processes=counts["stopping"],
            )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    async def warmup_async(self, batch_size: int | None = None) -> None:
        """Pre-spawn processes until ``len(_live) == max_processes``.
        Spawns in batches to limit resource contention - 128 concurrent
        Mathlib imports thrash I/O and blow past the per-import timeout.

        Allocates placeholders synchronously up-front so capacity is
        accurately reflected even mid-warmup; concurrent ``acquire``
        callers will either reuse a just-published idle entry or wait
        for capacity, never overcommit.
        """
        with self._lock:
            n = self.max_processes - len(self._live)
        if n <= 0:
            return
        effective_batch = batch_size if batch_size else n
        if effective_batch < n:
            self.logger.info(
                f"Starting {n} processes in batches of {effective_batch}"
            )
        else:
            self.logger.info(f"Starting {n} processes in parallel")

        done = 0
        while done < n:
            cur = min(effective_batch, n - done)
            if effective_batch < n:
                self.logger.info(
                    f"Starting batch of {cur} processes ({done}/{n} done)"
                )
            placeholders: list[PoolEntry] = []
            with self._capacity_changed:
                for _ in range(cur):
                    if len(self._live) >= self.max_processes:
                        break
                    placeholders.append(self._new_placeholder_locked())
            if not placeholders:
                break
            await asyncio.gather(
                *(self._fill_placeholder_async(p, into_idle=True) for p in placeholders)
            )
            done += len(placeholders)

        s = self.stats()
        self.logger.info(
            f"Started {done} processes. Live: {s.total_processes}, "
            f"available: {s.available_processes}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_placeholder_locked(self) -> PoolEntry:
        # Caller must hold self._lock (or self._capacity_changed, which
        # is the same RLock).
        self._id_counter += 1
        entry = PoolEntry(
            id=self._id_counter,
            pool=self,
            process=None,
            pid=None,
            state="starting",
            checkpoint=None,
            last_used=time.monotonic(),
        )
        self._live[entry.id] = entry
        return entry

    async def _spawn_async(self) -> LeanProcess:
        """Create + start + env-setup a Lean subprocess. Cleans up partial
        state on any failure between start and env-setup (otherwise the
        ~48 MiB-of-StreamReader-buffers leak documented in the old code's
        _create_process_async would resurface).

        Gated by ``self._spawn_semaphore`` when a concurrency limit was
        configured, so a runtime recycle wave can't fan out into N parallel
        Mathlib imports.
        """
        if self._spawn_semaphore is None:
            return await self._spawn_inner_async()
        async with self._spawn_semaphore:
            return await self._spawn_inner_async()

    async def _spawn_inner_async(self) -> LeanProcess:
        process = LeanProcess(
            self.repl_exe,
            self.project_path,
            self.logger,
            rss_hard_limit=self.rss_hard_limit,
        )
        started = False
        try:
            await process.start_async()
            started = True
            if self.env_setup_async:
                await self.env_setup_async(process)
            return process
        except BaseException:
            if started:
                try:
                    await process.stop_async_safe()
                except Exception as e:
                    self.logger.warning(
                        f"_spawn_async cleanup: stop_async_safe failed: {e}"
                    )
            raise

    async def _fill_placeholder_async(
        self, entry: PoolEntry, *, into_idle: bool
    ) -> None:
        """Spawn the subprocess for a placeholder. Used only by warmup
        right now (acquire_async inlines this for a simpler error path).
        """
        try:
            process = await self._spawn_async()
        except BaseException:
            self.release(entry, recycle=True, reason="warmup spawn failed")
            raise
        with self._capacity_changed:
            entry.process = process
            entry.pid = process._proc.pid
            entry.checkpoint = process.checkpoint()
            entry.state = "idle" if into_idle else "checked_out"
            entry.last_used = time.monotonic()
            if into_idle:
                self._capacity_changed.notify()

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    async def _janitor_loop(self, inbox: asyncio.Queue) -> None:
        """Drain entries marked 'stopping': stop the subprocess (5 s
        bounded), then remove from `_live` and notify capacity waiters.
        Lives on the server's event loop; never holds `_lock` across an
        `await`.

        Strict-capacity guarantee: capacity[i.e. `len(_live)`] decrements
        only here, after the subprocess is fully reaped. Combined with
        synchronous `release` adding entries to this queue, every
        `release(recycle=True)` eventually frees exactly one slot.
        """
        while True:
            entry = await inbox.get()
            if entry.process is not None:
                try:
                    await asyncio.wait_for(
                        entry.process.stop_async(), timeout=JANITOR_STOP_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Janitor: stop_async for entry {entry.id} "
                        f"(pid={entry.pid}) did not complete in "
                        f"{JANITOR_STOP_TIMEOUT} s; removing from pool anyway "
                        f"(subprocess leaked as zombie)"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Janitor: stop_async for entry {entry.id} "
                        f"(pid={entry.pid}) raised {type(e).__name__}: {e}"
                    )
            refill_placeholder: PoolEntry | None = None
            with self._capacity_changed:
                self._live.pop(entry.id, None)
                # notify (not notify_all): one freed slot wakes one waiter,
                # mirroring the FIFO semantics that asyncio.Semaphore gave us
                # before. notify_all here would re-create the thundering-herd
                # bug from the original pool design.
                self._capacity_changed.notify()
                # Eager refill: keep the pool at max_processes so the next
                # acquire wave doesn't pay Mathlib-import latency on the
                # first miss. Without this, recycled slots stay empty until
                # an acquire finds 0 idle entries and creates a placeholder
                # inline - observed in production as "62/64 alive" sitting
                # idle for hours after RL training stopped.
                if (
                    not self._shutdown
                    and len(self._live) < self.max_processes
                ):
                    refill_placeholder = self._new_placeholder_locked()
            if refill_placeholder is not None:
                # Schedule outside the lock; the spawn re-imports Mathlib
                # (~minutes) and must not block the janitor's next iteration.
                asyncio.create_task(
                    self._eager_refill_async(refill_placeholder),
                    name=f"LeanPool-eager-refill-{refill_placeholder.id}",
                )

    async def _eager_refill_async(self, entry: PoolEntry) -> None:
        """Spawn the subprocess for a janitor-created refill placeholder.

        Mirrors warmup's ``_fill_placeholder_async`` path but tolerates a
        shutdown racing the spawn: if ``_shutdown`` flips while we're
        waiting on Mathlib import, recycle the freshly-spawned process so
        the shutdown drain loop sees an empty pool. Spawn failures are
        already handled by ``_fill_placeholder_async`` (entry recycled);
        we just log so the failure isn't an unhandled task exception.
        """
        if self._shutdown:
            self.release(entry, recycle=True, reason="shutdown during eager refill")
            return
        try:
            await self._fill_placeholder_async(entry, into_idle=True)
        except Exception as e:
            self.logger.warning(
                f"Eager refill: spawn for entry {entry.id} failed: "
                f"{type(e).__name__}: {e}"
            )
            return
        if self._shutdown:
            self.release(entry, recycle=True, reason="shutdown during eager refill")

    def _idle_reaper_loop(self) -> None:
        """Reclaim entries whose client checked them out and went away.
        Skips entries with active HTTP requests (server installs
        ``has_active_request``). Replaces the old
        `LeanServer._reap_leaked_processes` thread.
        """
        while not self._workers_stop.wait(REAPER_INTERVAL):
            now = time.monotonic()
            stale: list[PoolEntry] = []
            with self._lock:
                for entry in self._live.values():
                    if entry.state != "checked_out":
                        continue
                    if now - entry.last_used <= self._lease_timeout:
                        continue
                    if self.has_active_request(entry.id):
                        continue
                    stale.append(entry)
            for entry in stale:
                self.logger.warning(
                    f"Idle reaper: reclaiming entry {entry.id} "
                    f"(pid={entry.pid}); unused for >{self._lease_timeout} s"
                )
                self.release(entry, recycle=True, reason="idle lease timeout")

    def _liveness_reconciler_loop(self) -> None:
        """Detect subprocesses whose kernel entry has vanished entirely
        (reaped without our knowledge) and recycle the pool entry. After
        commit e2a7390 made every kill use ``killpg``, the lake parent
        and repl grandchild always die together, so a missing PID is
        unambiguous: the whole REPL tree is gone.

        Note: ``os.kill(pid, 0)`` returns success for zombies, so this
        won't catch "subprocess died but kernel still has zombie entry";
        that case is already handled at the next REPL I/O attempt (EOF
        on stdout -> LeanProcessException -> recycle). This reconciler
        is the safety net for the rarer "fully reaped without us
        noticing" case.
        """
        while not self._workers_stop.wait(LIVENESS_INTERVAL):
            probes: list[tuple[PoolEntry, int]] = []
            with self._lock:
                for entry in self._live.values():
                    if entry.state not in ("idle", "checked_out"):
                        continue
                    if entry.pid is None:
                        continue
                    probes.append((entry, entry.pid))
            for entry, pid in probes:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    self.logger.warning(
                        f"Liveness reconciler: entry {entry.id} (pid={pid}) "
                        f"subprocess gone; recycling"
                    )
                    self.release(entry, recycle=True, reason="subprocess vanished")
                except PermissionError:
                    # PID recycled to a different process owned by someone
                    # else. Extremely unlikely (we're the parent, kernel
                    # reserves PIDs until wait()) but treat conservatively.
                    self.logger.warning(
                        f"Liveness reconciler: entry {entry.id} (pid={pid}) "
                        f"got EPERM; assuming PID was recycled, recycling entry"
                    )
                    self.release(
                        entry, recycle=True, reason="subprocess pid recycled"
                    )

    def _memory_governor_loop(self) -> None:
        """Background memory-budget enforcement.

        Round-robin walks ``_live`` one entry per
        ``MEMORY_GOVERNOR_INTERVAL``, reads the entry's PSS, and acts:

        - PSS at or below ``pss_recycle_limit``: do nothing.
        - PSS over the limit, entry idle: recycle inline.
        - PSS over the limit, entry checked_out: set ``over_budget=True``.
          ``return_entry_async`` honours the flag the next time the actor
          gives the process back.

        The PSS read happens *outside* the pool lock so a slow
        ``smaps_rollup`` walk can't stall acquire/release. The lock is
        only re-taken to read state and either flag or recycle, both of
        which are ~microseconds.

        This thread replaces the per-return PSS check that used to live
        in ``return_entry_async``. Putting the read on the request hot
        path turned every job-boundary return wave into an
        ``asyncio.to_thread`` storm; here it's a constant-rate background
        sweep that no client request waits on.
        """
        cursor = 0
        mib = 1024 * 1024
        while not self._workers_stop.wait(MEMORY_GOVERNOR_INTERVAL):
            if self.pss_recycle_limit is None:
                # Pool was reconfigured to disable PSS recycling; nothing
                # to do, but keep the thread alive (it's daemon and
                # joined at shutdown).
                continue
            with self._lock:
                entries = list(self._live.values())
            if not entries:
                cursor = 0
                continue
            cursor %= len(entries)
            entry = entries[cursor]
            cursor += 1

            with self._lock:
                if entry.id not in self._live:
                    continue
                if entry.state in ("starting", "stopping"):
                    continue
                state = entry.state
                flagged = entry.over_budget
                process = entry.process

            if flagged:
                # Already known over budget. If still checked_out, wait
                # for the actor's return; if idle, recycle now (rare
                # race: flag was set between drain and release(idle) in
                # the return path, so the return didn't honour it).
                if state == "idle":
                    self.release(
                        entry,
                        recycle=True,
                        reason="over PSS budget (idle, governor cleanup)",
                    )
                continue

            if process is None:
                continue

            try:
                pss = process.memory_usage_pss()
            except Exception as e:
                # Subprocess may have just died; let the liveness
                # reconciler handle it. Don't recycle here based on a
                # read failure - that would race with the normal
                # in-flight-tactic exception path.
                self.logger.debug(
                    f"memory governor: PSS read failed for entry {entry.id}: "
                    f"{type(e).__name__}: {e}"
                )
                continue

            if pss <= self.pss_recycle_limit:
                continue

            # Over budget. Decide based on current state under the lock so
            # we don't race acquire/return.
            with self._capacity_changed:
                if entry.id not in self._live:
                    continue
                state = entry.state
                if state == "checked_out":
                    entry.over_budget = True
                    self.logger.info(
                        f"memory governor: flagging entry {entry.id} "
                        f"(pid={entry.pid}) over budget "
                        f"(PSS={pss / mib:.1f} MiB > "
                        f"{self.pss_recycle_limit / mib:.1f} MiB); "
                        f"will recycle on return"
                    )
                    continue
                if state != "idle":
                    continue
            # Idle and over budget: recycle now. ``release`` re-takes the
            # lock and is idempotent; calling it outside the with-block
            # avoids holding the lock across the janitor handoff.
            self.release(
                entry,
                recycle=True,
                reason=(
                    f"PSS={pss / mib:.1f} MiB > "
                    f"recycle limit {self.pss_recycle_limit / mib:.1f} MiB "
                    f"(governor)"
                ),
            )

    # ------------------------------------------------------------------
    # Sync wrappers (for non-async callers, mainly tests)
    # ------------------------------------------------------------------

    return_entry = to_sync(return_entry_async)
    warmup = to_sync(warmup_async)
