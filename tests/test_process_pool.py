"""Integration tests for ``LeanProcessPool`` against real Lean REPL processes.

These tests use the in-loop variants (``start_async``/``shutdown_async``)
because pytest runs each test in its own asyncio loop and we want the
janitor task to live on that same loop.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from leantree.repl_adapter.process_pool import LeanProcessPool
from leantree.repl_adapter.interaction import LeanProcess

REPL_EXE = Path("../lean-repl/.lake/build/bin/repl")


def get_project_path():
    project_path = Path("leantree_project")
    if not project_path.exists():
        project_path = Path("../leantree_project")
    if not project_path.exists():
        raise FileNotFoundError(
            "leantree_project not found. Follow the Development section in README to create it."
        )
    return project_path


async def _make_pool(max_processes: int = 2, env_setup_async=None) -> LeanProcessPool:
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=get_project_path(),
        max_processes=max_processes,
        env_setup_async=env_setup_async,
        # Disable PSS recycling for these tests so return paths are
        # deterministic (no off-thread /proc reads, no recycle decisions).
        pss_recycle_limit=None,
    )
    await pool.start_async()
    return pool


@pytest.mark.asyncio
async def test_basic_acquire_return():
    """Acquire one entry, send a command, return - capacity recovers."""
    pool = await _make_pool(max_processes=2)
    try:
        async with await pool.acquire_async() as entry:
            assert entry.process is not None
            assert pool.stats().used_processes == 1
            await entry.process.send_command_async("#check Nat")
        # Healthy return: entry goes back to idle but stays alive in _live.
        s = pool.stats()
        assert s.total_processes == 1
        assert s.available_processes == 1
        assert s.used_processes == 0
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_multiple_processes_concurrent():
    """Two concurrent acquisitions get two distinct processes."""
    pool = await _make_pool(max_processes=2)
    try:
        async with await pool.acquire_async() as e1:
            async with await pool.acquire_async() as e2:
                assert e1.process is not e2.process
                assert pool.stats().used_processes == 2
                await asyncio.gather(
                    e1.process.send_command_async("#check Nat"),
                    e2.process.send_command_async("#check Int"),
                )
        s = pool.stats()
        assert s.used_processes == 0
        assert s.available_processes == 2
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_warmup_pre_spawns_to_capacity():
    pool = await _make_pool(max_processes=2)
    try:
        await pool.warmup_async()
        s = pool.stats()
        assert s.total_processes == s.max_processes
        assert s.available_processes == s.max_processes
        assert s.used_processes == 0
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_acquire_non_blocking_returns_none_at_capacity():
    pool = await _make_pool(max_processes=1)
    try:
        e1 = await pool.acquire_async()
        try:
            assert e1 is not None
            # At capacity now; non-blocking should return None.
            e2 = await pool.acquire_async(timeout=0)
            assert e2 is None
        finally:
            pool.release(e1, recycle=True, reason="test cleanup")
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_acquire_blocking_wakes_on_release():
    pool = await _make_pool(max_processes=1)
    try:
        e1 = await pool.acquire_async()
        assert e1 is not None
        # At capacity. Start a second acquire that must wait.
        wait_task = asyncio.create_task(pool.acquire_async(timeout=10.0))
        await asyncio.sleep(0.1)
        assert not wait_task.done()
        # Recycle e1 -> janitor frees the slot -> wait_task wakes.
        pool.release(e1, recycle=True, reason="test handoff")
        e2 = await asyncio.wait_for(wait_task, timeout=10.0)
        assert e2 is not None
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_env_setup_async_runs_per_spawn():
    setup_called = []

    async def setup_func(process: LeanProcess):
        setup_called.append(process)
        await process.send_command_async("#check Nat")

    pool = await _make_pool(max_processes=1, env_setup_async=setup_func)
    try:
        async with await pool.acquire_async() as entry:
            assert len(setup_called) == 1
            assert setup_called[0] is entry.process
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_process_reuse_across_acquisitions():
    """Returning to idle and re-acquiring gives the same underlying process."""
    pool = await _make_pool(max_processes=1)
    try:
        async with await pool.acquire_async() as e1:
            first_pid = e1.pid
        async with await pool.acquire_async() as e2:
            assert e2.pid == first_pid, "process should be reused after healthy return"
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_shutdown_cleans_up_all_processes():
    pool = await _make_pool(max_processes=2)
    await pool.warmup_async()
    async with await pool.acquire_async() as e:
        await e.process.send_command_async("#check Nat")
    assert pool.stats().total_processes == 2
    await pool.shutdown_async()
    assert pool.stats().total_processes == 0


@pytest.mark.asyncio
async def test_concurrent_acquire_release_under_load():
    """10 concurrent tasks contending for 2 slots; nothing leaks."""
    pool = await _make_pool(max_processes=2)
    try:
        async def use_once(task_id: int):
            async with await pool.acquire_async() as entry:
                await entry.process.send_command_async("#check Nat")
                await asyncio.sleep(0.01)
            return task_id

        results = await asyncio.gather(*(use_once(i) for i in range(10)))
        assert set(results) == set(range(10))
        # All released cleanly.
        assert pool.stats().used_processes == 0
    finally:
        await pool.shutdown_async()


@pytest.mark.asyncio
async def test_spawn_semaphore_caps_parallel_spawns():
    """``spawn_concurrency_limit`` must serialize ``_spawn_async`` calls so
    a runtime recycle wave can't fan out into N parallel Mathlib imports
    that all blow past the 40s outer acquire deadline. Uses a slow
    ``env_setup_async`` to make the spawn long enough to observe overlap.
    """
    in_flight = 0
    peak = 0

    async def slow_setup(process):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.3)
        finally:
            in_flight -= 1

    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=get_project_path(),
        max_processes=4,
        env_setup_async=slow_setup,
        pss_recycle_limit=None,
        spawn_concurrency_limit=2,
    )
    await pool.start_async()
    try:
        entries = await asyncio.gather(*(pool.acquire_async() for _ in range(4)))
        assert peak == 2, (
            f"spawn_concurrency_limit=2 should cap parallel spawns at 2; "
            f"observed peak in-flight {peak}"
        )
        for e in entries:
            pool.release(e, recycle=True)
    finally:
        await pool.shutdown_async()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
