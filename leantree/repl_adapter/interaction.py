import asyncio
import ctypes
import ctypes.util
import json
import logging
import os
import re
import resource
import signal
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self, AsyncIterator

import psutil

from leantree import utils
from leantree.core.abstraction import ProofBranch
from leantree.core.lean import LeanGoal, LeanTactic, LeanProofState
from leantree.core.lean_file import LeanTheorem, LeanFile, StoredError
from leantree.file_span import FilePosition, FileSpan
from leantree.metavar_graph import MetavarGraph
from leantree.repl_adapter.data import ReplGoalInfo, ReplCompilationUnit, FilePositionParser, ReplProofStepInfo
from leantree.utils import is_just_comments, ValueOrError, get_source_with_sorries, to_sync, to_sync_iterator


# Linux prctl numbers (from <sys/prctl.h>) - used to make lake/repl children
# die immediately when their parent (the leanserver Python process) dies,
# eliminating orphan processes after a crash or `kill -9`.
_PR_SET_PDEATHSIG = 1


def _make_subprocess_preexec_fn(max_memory_bytes: int | None):
    """Build a preexec_fn that hardens each lake/repl child subprocess.

    Runs in the *child* between fork() and execve() so the limits apply to the
    Lean subprocess (and propagate via execve to its lake/lean grandchildren).

    - ``RLIMIT_AS = max_memory_bytes`` (when set): a runaway tactic that blows past this gets SIGKILLed cleanly by the kernel; the parent sees EOF on the
      REPL pipes and surfaces a normal ``LeanProcessException``.  Without this, a single bad tactic can grow to 100+ GB and OOM-kill the whole leanserver
      Python process (observed in production).
    - ``prctl(PR_SET_PDEATHSIG, SIGKILL)``: when the parent process exits for any reason (including SIGKILL by the OOM killer), the kernel
      immediately sends SIGKILL to this child.  Eliminates orphaned lake/repl-processes.
    - ``setsid()``: put the child (and everything it forks: lake -> repl ->
      lean threads) into a fresh process group whose PGID equals this child's
      PID.  Without this, ``self._proc.kill()`` only SIGKILLs the immediate
      ``lake`` child; its ``repl`` grandchild survives, gets reparented to
      init, and keeps grinding on the last tactic forever.  Observed in
      production: a single orphaned repl with 101 threads burning ~5000% CPU
      and 60+ hours of cumulative CPU time after the lake parent died.  With
      setsid in place, ``os.killpg(proc.pid, SIGKILL)`` reaches every
      descendant in one syscall.
    """
    libc_path = ctypes.util.find_library("c") or "libc.so.6"
    try:
        _libc = ctypes.CDLL(libc_path, use_errno=True)
    except OSError:
        _libc = None

    # Runs post-fork, pre-exec: avoid Python's logging module (logger locks
    # held in the parent at fork time can deadlock here).  Write directly to
    # fd 2 instead - async-signal-safe and buffer-free.
    def _warn(msg: str) -> None:
        try:
            os.write(2, f"[preexec] {msg}\n".encode("utf-8", "replace"))
        except OSError:
            pass

    def _preexec():
        # Address-space ceiling. Set only the soft limit and preserve the
        # inherited hard limit. Wrap in try/except so a non-Linux platform
        # (where RLIMIT_AS is unavailable / works differently) doesn't take
        # down subprocess creation.
        if max_memory_bytes is not None:
            try:
                _, hard = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(
                    resource.RLIMIT_AS, (max_memory_bytes, hard)
                )
            except (ValueError, OSError) as e:
                _warn(f"setrlimit(RLIMIT_AS, {max_memory_bytes}) failed: {e!r}")
        # Parent-death signal: SIGKILL when leanserver Python process exits.
        # Linux-only; harmless no-op elsewhere because libc.prctl will just
        # error out and we swallow it.
        if _libc is not None:
            try:
                rc = _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
                if rc != 0:
                    errno = ctypes.get_errno()
                    _warn(
                        f"prctl(PR_SET_PDEATHSIG, SIGKILL) returned {rc} "
                        f"(errno={errno})"
                    )
            except (AttributeError, OSError) as e:
                _warn(f"prctl(PR_SET_PDEATHSIG, SIGKILL) failed: {e!r}")
        # New session + new process group with this child as leader; PGID
        # equals our PID after this call.  Forked descendants (repl, lean
        # threads) inherit the PGID, so a later os.killpg(pid, SIGKILL)
        # reaps the whole tree atomically.
        try:
            os.setsid()
        except OSError as e:
            _warn(f"setsid() failed: {e!r}")

    return _preexec


def _kill_process_group(proc, logger: logging.Logger) -> None:
    """SIGKILL every process in ``proc``'s process group.

    Relies on ``_preexec`` having called ``setsid()`` so that ``proc.pid`` is
    also the PGID.  ``proc.kill()`` alone would only signal the immediate
    child (``lake``) and leave its ``repl`` grandchild orphaned and spinning;
    ``killpg`` reaches every descendant in one syscall.

    Never raises: callers (e.g. ``stop_async``) need their cleanup to
    continue regardless of what the kill returned.  Outcomes are logged so
    nothing is silent:
    - ``ProcessLookupError`` -> warning.  Expected when the kernel
      SIGKILL'd the group via RLIMIT_AS before our kill ran, but worth
      surfacing because if it happens routinely our tactic timeouts are
      losing races to the memory ceiling.
    - Anything else (e.g. ``PermissionError`` from setsid not having run)
      -> error.  Indicates the killpg-based teardown is broken and
      grandchildren are about to leak.
    """
    pid = proc.pid
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        logger.warning(
            f"_kill_process_group: pgid={pid} already gone "
            f"(process exited before our kill ran)"
        )
    except Exception as e:
        logger.error(
            f"_kill_process_group: killpg(pgid={pid}, SIGKILL) failed - "
            f"descendants may be orphaned: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# Module-level deadline watchdog
# ---------------------------------------------------------------------------
# A single daemon thread polls every live ``LeanProcess`` instance and
# SIGKILLs the REPL process group of any whose tactic deadline has expired.
# Independent of any asyncio loop or request coroutine, so a coroutine
# cancelled mid-await still gets the kill - the cancellation race that
# previously left runaway tactics is impossible by construction.
#
# Why module-level (not per-process or per-pool):
# - Per-process thread = N threads for N processes; trivially wasteful at
#   N=64.
# - Per-pool thread = bare LeanProcess instances (project.py, tests) get
#   no protection.
# - One module-level thread, started lazily on first LeanProcess creation,
#   gives uniform coverage at a fixed cost.

_DEADLINE_WATCHDOG_INTERVAL = 0.25  # seconds

_live_processes: "weakref.WeakSet[LeanProcess]" = weakref.WeakSet()
_watchdog_lock = threading.Lock()
_watchdog_thread: threading.Thread | None = None
_watchdog_logger = logging.getLogger(__name__ + ".watchdog")


def _ensure_watchdog_started() -> None:
    """Start the module-level deadline watchdog thread if it isn't running.
    Called from ``LeanProcess.__init__``; the thread is daemon and lives
    for the rest of the Python process's lifetime."""
    global _watchdog_thread
    with _watchdog_lock:
        if _watchdog_thread is not None:
            return
        _watchdog_thread = threading.Thread(
            target=_deadline_watchdog_loop,
            name="LeanProcess-deadline-watchdog",
            daemon=True,
        )
        _watchdog_thread.start()


def _deadline_watchdog_loop() -> None:
    while True:
        time.sleep(_DEADLINE_WATCHDOG_INTERVAL)
        now = time.monotonic()
        try:
            procs = list(_live_processes)
        except RuntimeError:
            # WeakSet mutated during iteration (GC of a dead LeanProcess).
            # Skip this tick; we'll catch any expiries on the next one.
            continue
        for proc in procs:
            d = proc._deadline_until
            if d is None:
                continue
            deadline, gen = d
            if now <= deadline:
                continue
            # Recheck right before kill - if the deadline was cleared in
            # the window between our scan and now (the read returned and
            # the coroutine ran its `finally`), generation/value will
            # differ and we suppress the kill.  Without this guard a
            # late-firing watchdog would land on a process that's about
            # to serve the next request.
            cur = proc._deadline_until
            if cur is None or cur != (deadline, gen):
                continue
            try:
                proc.kill_group()
            except Exception as e:
                _watchdog_logger.error(
                    f"deadline watchdog kill_group raised "
                    f"{type(e).__name__}: {e}"
                )


# TODO!: maybe not all sorries are reported: see https://github.com/leanprover-community/repl/issues/4
#  E.g. in: `simpa using sorry`, the sorry is not detected

# TODO: add some way to flush envIds and proofIds - maybe take inspiration from the itp-interface project

@dataclass
class RunnableUnit:
    span: FileSpan
    proof_mask: list[FileSpan] | None = None
    theorem: LeanTheorem | None = None


@dataclass
class RunnableFile:
    path: Path
    units: list[RunnableUnit]

    @classmethod
    def from_lean_file(cls, file: LeanFile):
        units = []
        curr_position = FilePosition.beginning_of_file()
        for thm in file.theorems:
            if isinstance(thm, StoredError):
                continue
            # Note: It is tempting here to instead send `before_theorem + theorem_with_sorries` as one command. However,
            # this can lead to types in goals being reported differently, which breaks the verification.
            # Take as example the theorem Algebra.LinearRecurrence.geom_sol_iff_root_charPoly. Its root state has type:
            #
            # `(E.IsSolution fun n => q ^ n) ↔ E.charPoly.IsRoot q`
            #
            # It can be written like this because Lean already recognizes `charPoly` as a structure field of E thanks
            # to this definition placed right above the theorem:
            #
            # def charPoly : α[X] :=
            #   Polynomial.monomial E.order 1 - ∑ i : Fin E.order, Polynomial.monomial i (E.coeffs i)
            #
            # When we instead send the definition and the theorem at once, the type will be written differently as:
            #
            # `(E.IsSolution fun n => q ^ n) ↔ IsRoot (@LinearRecurrence.charPoly α inst✝ E) q`
            before_theorem = FileSpan(curr_position, thm.span.start)
            units.append(RunnableUnit(
                span=before_theorem,
            ))

            units.append(RunnableUnit(
                span=thm.span,
                proof_mask=[block.span for block in thm.by_blocks],
                theorem=thm,
            ))

            curr_position = thm.span.finish
        return RunnableFile(
            file.path,
            units,
        )


@dataclass
class LeanEnvironmentCheckpoint:
    env_id: int


@dataclass
class PickledEnv:
    path: Path


# TODO: maybe replace with `print axioms` to also catch `apply?`/`admit`
# Matches `:= sorry` / `:=\n  sorry` so we can rewrite it to `:= by sorry`.
# The `:=` is always preceded by whitespace in normal sources, so no `\b`
# prefix (which only fires between word/non-word chars and thus silently
# failed to match in the presence of that leading space).
_eq_sorry_pattern = re.compile(r':=\s*sorry\b')


class LeanProcess:
    def __init__(
        self,
        repl_exe: Path,
        project_path: Path,
        logger: logging.Logger | None = None,
        rss_hard_limit: int | None = None,
    ):
        self.repl_exe = repl_exe
        self.project_path = project_path
        self.logger = logger if logger else logging.getLogger(__name__)
        # Hard per-subprocess address-space ceiling enforced via RLIMIT_AS in
        # ``start_async``'s preexec_fn.  ``None`` disables the limit.
        self.rss_hard_limit = rss_hard_limit

        self._proc = None
        self._env_id = None
        self._stderr_buffer = deque(maxlen=50)
        self._stderr_task = None
        # Serializes _send_to_repl_async calls so two coroutines on the same
        # event loop can't interleave a write/read pair against the same
        # subprocess (which corrupts the StreamReader state and surfaces as
        # `RuntimeError: readuntil() called while another coroutine is
        # already waiting for incoming data`).  Created lazily because some
        # Python versions tie the lock to the running event loop.
        self._io_lock = None
        # Per-process tactic deadline.  ``None`` while no tactic is in
        # flight; ``(monotonic_deadline, generation)`` while one is.  Read
        # by the module-level deadline-watchdog thread, which fires
        # ``kill_group()`` on expiry independent of any request coroutine.
        # Generation-tagged so a watchdog firing in the window between read
        # success and ``finally``'s clear doesn't kill a process about to
        # serve the next request.  See ``_send_to_repl_async`` for arming.
        self._deadline_until: tuple[float, int] | None = None
        self._deadline_generation: int = 0

        # Register with the module-level watchdog.  The WeakSet entry is
        # auto-removed when this LeanProcess is GC'd, so no cleanup hook
        # is needed.  ``_ensure_watchdog_started`` is a no-op after the
        # first call.
        _live_processes.add(self)
        _ensure_watchdog_started()

    async def _describe_ended_process(self) -> str:
        """Human-readable explanation of why the REPL subprocess ended.

        Used when we observe EOF on stdout - the precise cause is the single
        most useful piece of information for debugging a run (e.g. "is our
        RLIMIT_AS too strict?" is only answerable if the server log shows
        SIGKILL specifically).  Combines:
        - the subprocess exit code / signal (sign-negated is the signal),
        - the last few lines the subprocess wrote to stderr (already
          captured by `_monitor_stderr`),
        - a best-guess tag for common causes (SIGKILL -> probably RLIMIT_AS
          or the OOM killer).
        """
        stderr_tail = list(self._stderr_buffer)[-10:]

        # Lean prints `INTERNAL PANIC: out of memory` on allocation failure
        # and then exits; the exit code we race to observe is just noise in
        # that case. Surface the real cause directly. `cannot allocate memory`
        # catches libgmp's `GNU MP: Cannot allocate memory` abort path, which
        # bypasses Lean's own panic message.
        for line in stderr_tail:
            low = line.lower()
            if (
                "out of memory" in low
                or "std::bad_alloc" in low
                or "cannot allocate memory" in low
            ):
                return (
                    "Lean REPL ran out of memory (INTERNAL PANIC) - tactic "
                    "allocation exceeded available memory; reduce tactic "
                    "size or raise --max-process-memory-gb"
                )
            if "stack overflow detected" in low:
                return (
                    "Lean REPL stack overflow during tactic elaboration - "
                    "pathological tactic (deep combinator nesting / recursive "
                    "macro); subprocess aborted"
                )

        # stdout EOF beats wait() to the punch fairly often: the kernel
        # closes the pipe FDs as part of teardown, but `returncode` only
        # gets set once asyncio has reaped the child. Give wait() a brief
        # window to resolve so we can report the real exit code/signal
        # instead of the "still running" race tag.
        rc = getattr(self._proc, "returncode", None)
        if rc is None:
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=0.5)
            except (asyncio.TimeoutError, AttributeError):
                pass
            rc = getattr(self._proc, "returncode", None)
        if rc is None:
            exit_desc = "still running (race: stdout closed before wait() saw the exit)"
        elif rc >= 0:
            exit_desc = f"exited with code {rc}"
        else:
            sig = -rc
            try:
                sig_name = signal.Signals(sig).name
            except (ValueError, AttributeError):
                sig_name = f"signal {sig}"
            hint = ""
            if sig == signal.SIGKILL:
                # SIGKILL has three common causes in this system:
                # 1. Our own kill-on-deadline watchdog (caller re-tags).
                # 2. The kernel enforcing RLIMIT_AS (our 8 GiB ceiling).
                # 3. The kernel OOM killer (host is out of memory).
                hint = (
                    " - likely RLIMIT_AS ceiling hit or kernel OOM killer; "
                    "if this recurs, consider raising --max-process-memory-gb"
                )
            elif sig == signal.SIGSEGV:
                hint = " - Lean segfault (likely tactic bug in the REPL)"
            exit_desc = f"killed by {sig_name}{hint}"
        stderr_part = f" - last stderr: {stderr_tail!r}" if stderr_tail else ""
        return f"Lean REPL process ended unexpectedly ({exit_desc}){stderr_part}"

    async def _monitor_stderr(self):
        """Read stderr in the background and buffer the last few lines."""
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='replace').strip()
                if decoded_line:
                    self._stderr_buffer.append(decoded_line)
                    # Also log to debug so it's not lost if not an error
                    self.logger.debug(f"REPL STDERR: {decoded_line}")
        except Exception as e:
            self.logger.warning(f"Error reading stderr: {e}")

    async def start_async(self):
        """Start the Lean REPL asynchronously."""
        assert self._proc is None
        self._stderr_buffer.clear()
        cmd = ["lake", "env", str(self.repl_exe)]

        self.logger.debug(f"Starting Lean REPL with command: {cmd} (working directory: {self.project_path})")
        # preexec_fn applies RLIMIT_AS + PR_SET_PDEATHSIG to the child between
        # fork() and execve(), so the Lean subprocess (and everything it
        # exec's into via lake) inherits a hard memory ceiling and dies with
        # the parent.  See _make_subprocess_preexec_fn.
        preexec_fn = _make_subprocess_preexec_fn(self.rss_hard_limit)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.project_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # The limit argument sets the buffer limit for StreamReader wrappers for Process.stdout and Process.stderr.
            limit=16 * 1024 * 1024,  # 16 MB
            preexec_fn=preexec_fn,
        )
        self._stderr_task = asyncio.create_task(self._monitor_stderr())

    start = to_sync(start_async)

    async def stop_async(self):
        """Stop the Lean REPL asynchronously."""
        assert self._proc is not None
        _kill_process_group(self._proc, self.logger)
        # See https://github.com/python/cpython/issues/119710#issuecomment-2425168469
        # and https://github.com/python/cpython/issues/88050
        # on why this line is necessary (otherwise the wait() call hangs).
        self._proc._transport.close()
        # D1: bounded wait.  Without a timeout this `await wait()` can park
        # the asyncio event loop indefinitely if the kernel can't reap the
        # subprocess promptly (NFS hangs and disk thrash under heavy load
        # both observed in production).  A stuck reap here cascades into
        # every other handler thread because they all cross into the same
        # loop via `_run_async`.  After SIGKILL the kernel should reap
        # within milliseconds; 5 s is a generous ceiling.  If we hit it we
        # log and move on - leaking one already-dead subprocess in zombie
        # state is much cheaper than freezing the server.
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Lean subprocess pid={self._proc.pid} did not reap within 5s after SIGKILL - "
                "leaving as a zombie to keep the event loop responsive"
            )

        if self._stderr_task:
            try:
                await asyncio.wait_for(self._stderr_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._stderr_task.cancel()
            self._stderr_task = None

        self._proc = None

    stop = to_sync(stop_async)

    async def stop_async_safe(self):
        try:
            await self.stop_async()
        except (LeanProcessException, ProcessLookupError):
            self._proc = None
            pass

    stop_safe = to_sync(stop_async_safe)

    async def restart_async_safe(self):
        await self.stop_async_safe()
        await self.start_async()
        self._env_id = None

    restart_safe = to_sync(restart_async_safe)

    async def __aenter__(self) -> Self:
        if not self._proc:
            await self.start_async()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.stop_async()

    def __enter__(self) -> Self:
        """Synchronous context manager entry."""
        if not self._proc:
            self.start()
        return self

    def __exit__(self, *args, **kwargs):
        """Synchronous context manager exit."""
        self.stop()

    def checkpoint(self) -> LeanEnvironmentCheckpoint:
        return LeanEnvironmentCheckpoint(self._env_id)

    def rollback_to(self, checkpoint: LeanEnvironmentCheckpoint):
        self._env_id = checkpoint.env_id

    def set_deadline(self, seconds: float | None) -> None:
        """Arm or clear the per-process tactic deadline.  Read by the
        module-level deadline-watchdog thread, which fires ``kill_group()``
        on expiry independently of the awaiting coroutine.

        The generation counter advances on every call so the watchdog can
        recheck immediately before kill: if the value it sampled doesn't
        match the current value, the deadline was cleared (the read
        returned cleanly) and the kill is suppressed.

        ``seconds=None`` clears the deadline.  Idempotent.
        """
        # Plain attribute write under the GIL is atomic; no lock needed.
        # The generation counter ensures the watchdog can detect a stale
        # sample without coordination.
        self._deadline_generation += 1
        if seconds is None:
            self._deadline_until = None
        else:
            self._deadline_until = (time.monotonic() + seconds, self._deadline_generation)

    def kill_group(self) -> None:
        """SIGKILL the entire REPL process group via ``killpg``.  Wraps
        ``_kill_process_group`` so external callers (the pool watchdog)
        don't need to import a private module helper.  Relies on
        ``setsid()`` in the preexec_fn (commit e2a7390); without setsid
        this would only signal the immediate ``lake`` child and the
        ``repl`` grandchild would survive as a runaway orphan.
        Idempotent and never raises.
        """
        if self._proc is not None:
            _kill_process_group(self._proc, self.logger)

    async def _send_to_repl_async(self, data: dict, timeout: float | None = 300) -> dict:
        """Send data to the REPL asynchronously and return the response.

        Args:
            data: JSON-serializable dict to send to the REPL.
            timeout: Maximum seconds to wait for a complete response.
                     ``None`` means no limit.

        Implementation note - timeout is enforced by SIGKILL-on-deadline,
        not ``asyncio.wait_for``.  Cancelling a task that's parked inside
        ``StreamReader.readline()`` does NOT reliably reset the reader's
        internal ``_waiter`` (CPython issue 30289).  In practice this means
        a single timed-out read corrupts the stream so that the next read
        on the same process raises
        ``RuntimeError: readuntil() called while another coroutine is
        already waiting for incoming data``.  When that propagates out of
        a leanserver request handler it deadlocks the asyncio event loop,
        and the whole leanserver eventually stops answering HTTP.

        Kill-on-deadline runs in the module-level watchdog thread, NOT in
        a per-call ``loop.call_later``.  The old per-call timer was armed
        and cancelled inside this coroutine, so a coroutine cancelled
        before its ``await`` resumed (CancelledError, request hangup,
        outer ``_run_async_op`` timing out) cancelled the timer in
        ``finally`` and left the REPL grinding with no watchdog.  The
        thread-based watchdog is independent of this coroutine's
        lifetime.  ``readline()`` sees EOF when the watchdog kills, the
        inner loop converts that to ``LeanProcessException``, and the
        post-read deadline check below re-tags it as a timeout.
        """
        self._assert_started()
        serialized = json.dumps(data, ensure_ascii=False) + "\n\n"

        self.logger.debug(f"Sending to REPL: '{serialized[:-2]}'")

        if self._io_lock is None:
            self._io_lock = asyncio.Lock()

        response_lines = []
        deadline_at: float | None = (
            time.monotonic() + timeout if timeout is not None else None
        )

        async def _read_response():
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    # The subprocess's stdout closed - it has exited (or is
                    # about to).  Build a richer diagnostic so we can tell
                    # *why* from the server log alone.  Cases we care about:
                    #   - returncode == -9 (SIGKILL): RLIMIT_AS ceiling hit,
                    #     kernel OOM killer, or our own kill-on-deadline.
                    #   - returncode == -11 (SIGSEGV): Lean segfault.
                    #   - returncode > 0: Lean exited on its own.
                    # The post-read deadline check tags the timeout case
                    # specifically; this path covers all the others.
                    raise LeanProcessException(await self._describe_ended_process())
                decoded_line = line.decode('utf-8')
                if decoded_line.strip() == "":
                    break
                response_lines.append(decoded_line)

        async with self._io_lock:
            try:
                self._proc.stdin.write(serialized.encode('utf-8'))
                await self._proc.stdin.drain()

                self.set_deadline(timeout)
                try:
                    await _read_response()
                finally:
                    self.set_deadline(None)
            except LeanProcessException:
                if deadline_at is not None and time.monotonic() > deadline_at:
                    raise LeanProcessException(
                        f"Lean REPL did not respond within {timeout}s"
                    )
                raise
            except (BrokenPipeError, ValueError, OSError) as e:
                raise LeanProcessException(f"Failed to send data to REPL: {e}") from e

        response_str = "".join(response_lines)
        self._log_repl_response(response_str)
        response = json.loads(response_str)

        messages = response.get("messages", [])
        errors = [m for m in messages if m["severity"] == "error"]
        # TODO: handle warnings
        warnings = [m for m in messages if m["severity"] == "warning" and m["data"] != "declaration uses 'sorry'"]
        if len(errors) > 0:
            raise LeanInteractionException(f"REPL returned error messages: {json.dumps(errors, ensure_ascii=False)}")

        message = response.get("message")
        if message == "Operation timed out":
            raise LeanInteractionException("Tactic application timed out.")
        if message and message.startswith("Lean error:"):
            raise LeanInteractionException(f"REPL returned error: {message}")

        return response

    async def drain_repl_output_async(self):
        """Drain any not-yet-read REPL output to prevent garbage in subsequent reads."""
        self._assert_started()
        try:
            while True:
                try:
                    # Try to read up to 1024 bytes, but time out immediately if no data
                    data = await asyncio.wait_for(self._proc.stdout.read(1024), timeout=0)
                except asyncio.TimeoutError:
                    # no data is ready right now
                    break
                if not data:
                    # EOF
                    break
        except Exception as e:
            self.logger.warning(f"Error while draining REPL output: {e}")

    def _log_repl_response(self, response_str: str):
        to_filter = ["goalInfo", "goalInfos", "mctxBefore", "mctxAfter", "infotree", "infoTree"]

        def filter_data(data: dict | list):
            keys = data.keys() if isinstance(data, dict) else range(len(data))
            for k in keys:
                if k in to_filter:
                    data[k] = "<HIDDEN>"
                elif isinstance(data[k], dict) or isinstance(data[k], list):
                    filter_data(data[k])

        try:
            response = json.loads(response_str)
            filter_data(response)
            self.logger.debug(f"Received from REPL: '{json.dumps(response, ensure_ascii=False)}'")
        except json.JSONDecodeError:
            self.logger.debug(f"Received from REPL (could not parse): '{response_str}'")

    async def send_command_async(
        self,
        command: str,
        proof_trees: bool = False,
        info_trees: bool = False,
        timeout: float | None = 300.0,
    ) -> dict:
        """Send a command to the REPL asynchronously and return the response.

        Args:
            command: the Lean code to send to the REPL.
            timeout: seconds to wait for the REPL response.  Default 300s
                matches ``_send_to_repl_async``'s default.  Callers that send
                expensive long-running commands (e.g. ``import Mathlib`` at
                warmup under heavy system load - observed to take >5 min on
                loaded hosts) should pass a larger value.
        """
        self._assert_started()

        # Note: This is a temporary hack to avoid sending sorry without "by".
        command = self._eliminate_sorry_without_by(command)

        data = {"cmd": command}
        if self._env_id is not None:
            data["env"] = self._env_id
        if proof_trees:
            data["proofTrees"] = True
        if info_trees:
            data["infotree"] = "no_children"

        response = await self._send_to_repl_async(data, timeout=timeout)

        if "env" not in response:
            # REPL returned a fatal error without advancing the environment.
            # Typical cause: a parse or elaboration failure severe enough that
            # no new env snapshot was produced. Surface the payload as a
            # LeanInteractionException so callers (e.g., proof_from_sorry)
            # can report it cleanly instead of asserting.
            message = response.get("message")
            raise LeanInteractionException(
                f"No `env` in REPL response. message={message!r} response={response!r}"
            )
        self._env_id = response["env"]
        return response

    send_command = to_sync(send_command_async)

    @staticmethod
    def _eliminate_sorry_without_by(text: str) -> str:
        def repl(m: re.Match) -> str:
            s = str(m.group(0))
            return s.replace("sorry", "by sorry")

        return _eq_sorry_pattern.sub(repl, text)

    async def is_valid_source_async(self, source: str) -> bool:
        """Check if the source is valid Lean code."""
        try:
            await self.send_command_async(source)
            return True
        except LeanInteractionException:
            return False

    is_valid_source = to_sync(is_valid_source_async)

    async def pickle_async(self, path: Path | str) -> PickledEnv:
        """Pickle the current REPL environment to a file asynchronously."""
        self._assert_started()
        data = {"pickleTo": str(path)}
        if self._env_id is not None:
            data["env"] = self._env_id

        await self._send_to_repl_async(data)
        return PickledEnv(Path(path))

    pickle = to_sync(pickle_async)

    async def unpickle_async(self, path: Path | str):
        """Unpickle a REPL environment from a file asynchronously."""
        self._assert_started()
        response = await self._send_to_repl_async({"unpickleEnvFrom": str(path)})

        assert "env" in response, f"no `env` in REPL response with keys: {response.keys()}"
        self._env_id = response["env"]

    unpickle = to_sync(unpickle_async)

    def _assert_started(self):
        if self._proc is None:
            raise LeanProcessException(
                "Subprocess not started. Use 'with LeanProcess(...) as env:' or 'async with LeanProcess(...) as env:'"
            )
        if self._proc.returncode is not None:
            stderr_tail = "\n".join(self._stderr_buffer)
            raise LeanProcessException(
                f"Subprocess has terminated with exit code {self._proc.returncode}.\n"
                f"Stderr output:\n{stderr_tail}\n"
                "Use 'with LeanProcess(...) as env:' or 'async with LeanProcess(...) as env:'"
            )

    async def proofs_from_sorries_async(self, theorem_with_sorries: str) -> "AsyncIterator[LeanProofBranch]":
        """Start proofs from sorries asynchronously."""
        self._assert_started()
        response = await self.send_command_async(theorem_with_sorries)
        if "sorries" not in response:
            raise Exception(f"No `sorries` in REPL response. Make sure your theorem contains a 'sorry' keyword.")
        sorries = response["sorries"]
        goals = [ReplGoalInfo.goal_from_repl_data(sorry_data["goalInfo"]) for sorry_data in sorries]
        for sorry_data, goal in zip(sorries, goals):
            yield LeanProofBranch(self, sorry_data["proofState"], goal)

    proofs_from_sorries = to_sync_iterator(proofs_from_sorries_async)

    async def proof_from_sorry_async(self, theorem_with_sorry: str) -> "LeanProofBranch":
        """Start a proof from a sorry asynchronously."""
        proofs = [branch async for branch in self.proofs_from_sorries_async(theorem_with_sorry)]
        if len(proofs) != 1:
            raise Exception(f"{len(proofs)} occurrences of `sorry` in the theorem (expected 1).")
        return proofs[0]

    proof_from_sorry = to_sync(proof_from_sorry_async)

    async def file_proofs_async(
            self,
            file: LeanFile,
    ) -> "AsyncIterator[tuple[LeanTheorem, list[LeanProofBranch] | Exception]]":
        """Start file proofs asynchronously."""
        async for unit, sorry_branches in self.runnable_proofs_async(RunnableFile.from_lean_file(file)):
            if isinstance(sorry_branches, Exception):
                yield unit.theorem, sorry_branches
                continue
            if unit.theorem is not None:
                if len(sorry_branches) != len(unit.theorem.by_blocks):
                    yield unit.theorem, Exception(
                        f"Mismatch between REPL sorry branches ({len(sorry_branches)}) "
                        f"and tree-builder by-blocks ({len(unit.theorem.by_blocks)})."
                    )
                    continue
                yield unit.theorem, sorry_branches

    file_proofs = to_sync_iterator(file_proofs_async)

    async def runnable_proofs_async(
            self,
            file: RunnableFile,
    ) -> "AsyncIterator[tuple[RunnableUnit, list[LeanProofBranch] | Exception]]":
        """Start runnable proofs asynchronously."""
        self._assert_started()
        with open(file.path, "r") as f:
            file_content = f.read()
        for unit in file.units:
            source = unit.span.read_from_string(file_content)
            # We do not send comment-only statements because sending them seems to sometimes break the REPL (and it is
            # not necessary).
            if is_just_comments(source):
                continue

            if unit.proof_mask:
                try:
                    source_with_sorries = get_source_with_sorries(unit.span, unit.proof_mask, file_content=file_content)
                    response = await self.send_command_async(source_with_sorries)
                except (AssertionError, LeanInteractionException) as e:
                    yield unit, e
                    if unit.proof_mask:
                        await self.send_command_async(source)
                    continue
            else:
                response = await self.send_command_async(source)
            sorry_branches = []
            if "sorries" in response:
                sorries = response["sorries"]
                goals = [ReplGoalInfo.goal_from_repl_data(sorry_data["goalInfo"]) for sorry_data in sorries]
                for sorry_data, goal in zip(sorries, goals):
                    sorry_branches.append(LeanProofBranch(self, sorry_data["proofState"], goal))
            yield unit, sorry_branches

    runnable_proofs = to_sync_iterator(runnable_proofs_async)

    async def full_proofs_async(self, file: LeanFile) -> "AsyncIterator[tuple[LeanTheorem, LeanProofBranch]]":
        """Start full proofs asynchronously."""
        pass  # TODO: Implement this method

    async def take_control_async(self) -> None:
        """
        Asynchronously hands control of the subprocess to the user for debugging purposes.
        """
        if self._proc is None:
            raise Exception("Subprocess not started. Use 'async with LeanProcess(...) as env:'")

        async def read_and_print_stream(stream, print_prefix=""):
            while True:
                line = await stream.readline()
                if not line:
                    break
                print(f"{print_prefix}{line.decode('utf-8')}", end="")

        # Create tasks to read from stdout and stderr
        stdout_task = asyncio.create_task(read_and_print_stream(self._proc.stdout))
        stderr_task = asyncio.create_task(read_and_print_stream(self._proc.stderr, "STDERR: "))

        # Read user input and send it to subprocess stdin
        try:
            loop = asyncio.get_event_loop()
            while self._proc.returncode is None:
                # Use a thread to get user input without blocking the event loop
                user_input = await loop.run_in_executor(None, input)
                if not user_input:
                    break
                self._proc.stdin.write((user_input + "\n").encode('utf-8'))
                await self._proc.stdin.drain()
        except (EOFError, BrokenPipeError, KeyboardInterrupt):
            print("User interrupted input or pipe broken. Exiting control mode.")
        finally:
            # Wait for the subprocess to exit
            if self._proc.returncode is None:
                _kill_process_group(self._proc, self.logger)
            await self._proc.wait()
            # Cancel the reading tasks
            stdout_task.cancel()
            stderr_task.cancel()
            try:
                await stdout_task
            except asyncio.CancelledError:
                pass
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

    take_control = to_sync(take_control_async)

    def _memory_usage(self, get_memory) -> int:
        self._assert_started()
        try:
            process = psutil.Process(self._proc.pid)
            total = get_memory(process)
            for child in process.children(recursive=True):
                total += get_memory(child)
            return total
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error) as e:
            raise LeanProcessException(f"Failed to get memory usage", e)

    def memory_usage_pss(self) -> int:
        """
        Returns the PSS (Proportional Set Size) memory usage of the Lean REPL process
        and all its children in bytes. PSS divides shared pages by the number of
        processes sharing them, avoiding the overcounting that RSS suffers from when
        many processes share memory-mapped files (e.g. Mathlib .olean files).
        """
        return self._memory_usage(lambda proc: proc.memory_full_info().pss)

    def memory_usage_rss(self) -> int:
        return self._memory_usage(lambda proc: proc.memory_info().rss)

    async def send_theorem_async(self, theorem_str: str) -> ReplCompilationUnit:
        """Send a theorem to the REPL asynchronously."""
        line_lengths = [len(line) for line in theorem_str.splitlines(keepends=True)]
        response = await self.send_command_async(
            theorem_str,
            proof_trees=True,
            info_trees=True,
        )
        assert len(response["proofTreeEdges"]) == len(response["infotree"])
        # TODO: what if there are more compilation units

        proof_steps = response["proofTreeEdges"][0]
        root_info_tree = response["infotree"][0]

        proof_steps = [ReplProofStepInfo.from_repl_data(step, line_lengths) for step in proof_steps]

        src_range = root_info_tree["node"]["stx"]["range"]
        pretty_print = root_info_tree["node"]["stx"]["pp"]
        if pretty_print:
            pretty_print = utils.remove_empty_lines(utils.remove_comments(pretty_print))
        span = FilePositionParser.create_file_span(src_range, line_lengths)

        return ReplCompilationUnit(
            proof_steps,
            pretty_print,
            span,
            None,
        )

    send_theorem = to_sync(send_theorem_async)

    @classmethod
    def _goals_from_response(cls, response: dict) -> list[LeanGoal]:
        """Extract goals from REPL response."""
        return [ReplGoalInfo.goal_from_repl_data(goal_info) for goal_info in response["goalInfos"]]


class LeanProofBranch(ProofBranch[LeanGoal, LeanTactic]):
    def __init__(self, env: LeanProcess, proof_state_id: int, all_goals: list[LeanGoal] | LeanGoal,
                 goals_mask: list[bool] = None):
        self._env = env
        self._proof_state_id = proof_state_id
        self._all_goals = all_goals if isinstance(all_goals, list) else [all_goals]
        self._goals_mask = goals_mask
        assert self._goals_mask is None or len(self._all_goals) == len(self._goals_mask)

    def __str__(self):
        return f"LeanProofBranch[proofState={self._proof_state_id},state={self.state}]"

    @property
    def state(self) -> LeanProofState:
        if self._goals_mask is None:
            return LeanProofState([goal for goal in self._all_goals])
        return LeanProofState([goal for goal, visible in zip(self._all_goals, self._goals_mask) if visible])

    @property
    def is_solved(self) -> bool:
        return self.state.is_solved()

    async def _send_tactic_async(self, tactic: str, timeout: int | None = 5000) -> dict:
        data = {
            "tactic": tactic,
            "proofState": self._proof_state_id,
        }
        if timeout is not None:
            data["timeout"] = timeout

        response = await self._env._send_to_repl_async(data)
        return response

    async def _delete_masked_goals_async(self):
        """
        Gets rid of all masked goals so that a tactic cannot affect them and that we do not get confused about what
        needs to be proven. Called lazily before tactic execution.
        Must not change order of the non-hidden goals.
        """
        if self._goals_mask is None or all(self._goals_mask):
            return
        old_state = self.state

        masked_spans = []
        i = 0
        while i < len(self._all_goals):
            if self._goals_mask[i]:
                # Non-masked goal.
                i += 1
                continue
            span_start = i
            while i < len(self._all_goals) and not self._goals_mask[i]:
                i += 1
            masked_spans.append((span_start, i))
        assert len(masked_spans) > 0

        tactics = []
        i = 0
        for start, end in masked_spans:
            if start != i:
                # Skip non-masked goals.
                tactics.append(f"rotate_left {start - i}")
            # Get rid of masked goals.
            tactics.append(f"iterate {end - start} sorry")
            i = end
        if i < len(self._all_goals):
            # Make sure the order of non-masked goals is not changed.
            tactics.append(f"rotate_left {len(self._all_goals) - i}")

        response = None
        for tactic in tactics:
            response = await self._send_tactic_async(tactic)
            self._proof_state_id = response["proofState"]
        assert response is not None
        final_goals = LeanProcess._goals_from_response(response)
        assert old_state.semantic_equals(LeanProofState(final_goals))

        self._all_goals = final_goals
        self._goals_mask = None

    async def apply_tactic_async(
            self,
            tactic: LeanTactic | str,
            # Tactics rw?, apply?, exact?, grind? technically close the main goal, but the proof is invalid. Setting
            # ban_search_tactics disallows these. Consider e.g.:
            # example : 1 = 0 := by
            #   apply?
            ban_search_tactics: bool = True,
            timeout: int | None = 5000,
    ) -> list[Self]:
        assert not self.state.is_solved(), "This proof branch is already solved."
        if isinstance(tactic, LeanTactic):
            tactic = tactic.tactic
        self._check_tactic(tactic, ban_search_tactics)

        # Normalize the proof state by removing masked goals.
        await self._delete_masked_goals_async()

        response = await self._send_tactic_async(tactic, timeout=timeout)
        if "goals" not in response:
            raise LeanInteractionException(f"Could not apply tactic in REPL: {json.dumps(response)}")
        new_proof_state = response["proofState"]
        step_error = self.step_error_from_response(response)
        if step_error:
            raise LeanInteractionException(f"Step verification error: {step_error}")
        new_goals = LeanProcess._goals_from_response(response)
        metavar_graph = MetavarGraph.from_dict(response["mctxAfter"])

        next_states = []
        for branch_goals in metavar_graph.partition_independent_goals(new_goals):
            next_states.append(LeanProofBranch(
                self._env,
                new_proof_state,
                new_goals,
                goals_mask=[g in branch_goals for g in new_goals],
            ))

        # `sorries` can be generated e.g. when executing a `have` tactic. They create an entirely new proofState with a
        # single goal.
        for sorry_data in response.get("sorries", []):
            goal = ReplGoalInfo.goal_from_repl_data(sorry_data["goalInfo"])
            next_states.append(LeanProofBranch(self._env, sorry_data["proofState"], goal))

        # This is a temporary hack to disallow things like "exact (by _ : _)" which currently break the REPL verification.
        for next_state in next_states:
            for goal in next_state.state.goals:
                if goal.type.startswith("?") and " " not in goal.type:
                    raise LeanInteractionException("Metavariable-only goal types are not allowed.")

        return next_states

    apply_tactic = to_sync(apply_tactic_async)

    # TODO: def apply_tactics

    async def try_apply_tactic_async(self, tactic: LeanTactic | str, timeout: int | None = 5000) -> ValueOrError[list[Self]]:
        try:
            return ValueOrError.from_success(await self.apply_tactic_async(tactic, timeout=timeout))
        except (LeanInteractionException, AssertionError) as e:
            return ValueOrError.from_error(e)

    try_apply_tactic = to_sync(try_apply_tactic_async)

    @classmethod
    def _check_tactic(cls, tactic: str, ban_search_tactics: bool):
        tactic = tactic.strip()
        # `have` without specifying the hypothesis type is accepted by the REPL but not by Lean.
        if tactic.startswith("have ") or tactic.startswith("haveI") or tactic.startswith("have'"):
            if ":" not in tactic:
                raise LeanInteractionException("`have` must specify the hypothesis type")
        if tactic.startswith("simpa ") and "sorry" in tactic:
            # As of now, the REPL does no correctly detect `sorry` in a `simpa ... using` tactic.
            raise LeanInteractionException("`sorry` not allowed in `simpa`")
        if ban_search_tactics and any(tactic.startswith(banned) for banned in ["apply?", "rw?", "exact?", "grind?"]):
            raise LeanInteractionException("Search tactics (apply?, rw?, exact?, grind?) are not allowed.")

    @classmethod
    def step_error_from_response(cls, response: dict) -> str | None:
        status = response["stepVerification"]
        if status == "OK":
            return None
        return status


class LeanInteractionException(Exception):
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.__cause__ = cause


class LeanProcessException(Exception):
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.__cause__ = cause
