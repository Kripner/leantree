"""
tests:
- start process pool, request server, import mathlib, return server, request again, import mathlib again and measure
  that the second import does not take longer than a few milliseconds
"""
import asyncio
import sys
import time
from pathlib import Path
from typing import Callable

from leantree.repl_adapter.process_pool import LeanProcessPool
from leantree.repl_adapter.interaction import LeanProcess

# Get REPL_EXE from conftest pattern
REPL_EXE = Path("../lean-repl/.lake/build/bin/repl")


def get_project_path():
    """Get the project path for testing."""
    project_path = Path("leantree_project")
    if not project_path.exists():
        raise FileNotFoundError(
            f"Project path {project_path} does not exist. Please follow the Development section in README to create it."
        )
    return project_path


async def test_mathlib_import_caching(project_path: Path):
    """
    Test that importing mathlib twice is fast on the second import due to environment caching.
    """
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )
    try:
        # Get a process from the pool
        async with await pool.get_process_async() as process1:
            assert process1 is not None

            # First import - this should take some time
            start_time = time.time()
            await process1.send_command_async("import Mathlib")
            first_import_time = time.time() - start_time

        assert len(pool.available_processes) == 1

        # Get a process again (should be the same one or a new one)
        async with await pool.get_process_async() as process2:
            assert process2 is not None
            assert id(process1) == id(process2), "Process should be reused"

            # Second import - should be fast if using the same process with cached environment
            start_time = time.time()
            await process2.send_command_async("import Mathlib")
            second_import_time = time.time() - start_time

        assert len(pool.available_processes) == 1

        print(f"First import took {first_import_time:.3f}s, second import took {second_import_time:.3f}s")

        # The second import should be much faster (less than 100ms for cached import)
        # Note: This is a heuristic - actual times may vary
        assert second_import_time < 0.1, f"Second import took {second_import_time:.3f}s, expected < 0.1s"
    finally:
        await pool.shutdown_async()


async def test_basic_get_return_process(project_path: Path):
    """Test basic get and return operations."""
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )
    try:
        # Get a process using context manager
        async with await pool.get_process_async() as process:
            assert process is not None
            assert pool._num_used_processes == 1

            # Use the process
            await process.send_command_async("#check Nat")

        # Process should be returned automatically
        assert pool._num_used_processes == 0
        assert len(pool.available_processes) == 1
    finally:
        await pool.shutdown_async()


async def test_multiple_processes(project_path: Path):
    """Test that the pool can handle multiple processes."""
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )
    try:
        # Get two processes using context managers
        async with await pool.get_process_async() as process1:
            async with await pool.get_process_async() as process2:
                assert process1 is not None
                assert process2 is not None
                assert process1 is not process2  # Should be different processes
                assert pool._num_used_processes == 2

                # Use both processes concurrently
                await asyncio.gather(
                    process1.send_command_async("#check Nat"),
                    process2.send_command_async("#check Int"),
                )

        # Both processes should be returned automatically
        assert pool._num_used_processes == 0
        assert len(pool.available_processes) == 2
    finally:
        await pool.shutdown_async()


async def test_max_out_processes(project_path: Path):
    """Test max_out_processes_async creates processes up to max_processes."""
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )
    try:
        # Initially no processes
        assert len(pool.available_processes) == 0
        assert pool._num_used_processes == 0

        # Max out processes
        await pool.max_out_processes_async()

        # Should have max_processes available
        assert len(pool.available_processes) == pool.max_processes
        assert pool._num_used_processes == 0
    finally:
        await pool.shutdown_async()


async def test_get_process_non_blocking(project_path: Path):
    """Test non-blocking get_process returns None when no processes available."""
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )
    try:
        # Max out the pool
        await pool.max_out_processes_async()

        # Get all processes using context managers
        processes = []
        for _ in range(pool.max_processes):
            process = await pool.get_process_async(blocking=False)
            assert process is not None
            processes.append(process)

        # Now try to get another one non-blocking - should return None
        process = await pool.get_process_async(blocking=False)
        assert process is None

        # Return one process manually (we got it without context manager)
        await pool.return_process_async(processes[0])

        # Now should be able to get one using context manager
        process = await pool.get_process_async(blocking=False)
        assert process is not None
        async with process:
            pass  # Process will be returned automatically when exiting context

        # Return remaining processes
        for p in processes[1:]:
            await pool.return_process_async(p)
    finally:
        await pool.shutdown_async()


async def test_get_process_blocking(project_path: Path):
    """Test blocking get_process waits for a process to become available."""
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )
    try:
        # Max out the pool and get all processes
        await pool.max_out_processes_async()
        processes = []
        for _ in range(pool.max_processes):
            process = await pool.get_process_async(blocking=False)
            processes.append(process)

        # Start a task that will wait for a process using context manager
        async def get_process_task():
            async with await pool.get_process_async(blocking=True) as process:
                return process

        get_task = asyncio.create_task(get_process_task())

        # Wait a bit to ensure the task is waiting
        await asyncio.sleep(0.1)
        assert not get_task.done()

        # Return a process - should unblock the waiting task
        await pool.return_process_async(processes[0])

        # The task should complete quickly
        returned_process = await asyncio.wait_for(get_task, timeout=2.0)
        assert returned_process is not None

        # Return remaining processes
        for p in processes[1:]:
            await pool.return_process_async(p)
    finally:
        await pool.shutdown_async()


async def test_context_manager(project_path: Path):
    """Test that processes from pool can be used as a context manager."""
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )
    try:
        # Use the recommended pattern: async with await pool.get_process_async()
        async with await pool.get_process_async() as process:
            assert process is not None
            await process.send_command_async("#check Nat")
            # Process should be returned automatically when exiting context

        # Process should be back in the pool
        assert len(pool.available_processes) == 1
        assert pool._num_used_processes == 0
    finally:
        await pool.shutdown_async()


async def test_env_setup_async(project_path: Path):
    """Test that env_setup_async is called when creating new processes."""
    setup_called = []

    async def setup_func(process: LeanProcess):
        setup_called.append(process)
        await process.send_command_async("#check Nat")

    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=1,
        env_setup_async=setup_func,
    )

    try:
        # Get a process using context manager - should trigger setup
        async with await pool.get_process_async() as process:
            assert len(setup_called) == 1
            assert setup_called[0] is process
    finally:
        await pool.shutdown_async()


async def test_shutdown_cleans_up_processes(project_path: Path):
    """Test that shutdown properly cleans up all processes."""
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    # Create and use some processes
    await pool.max_out_processes_async()
    async with await pool.get_process_async() as process:
        await process.send_command_async("#check Nat")

    # Shutdown should clean up all processes
    await pool.shutdown_async()

    # All processes should be stopped
    assert len(pool.available_processes) == 0


async def test_process_reuse(project_path: Path):
    """Test that processes are reused when returned to the pool."""
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )
    try:
        # Get a process using context manager
        async with await pool.get_process_async() as process1:
            process1_id = id(process1)

        # Get another process - should be the same one
        async with await pool.get_process_async() as process2:
            process2_id = id(process2)
            assert process1_id == process2_id, "Process should be reused"
    finally:
        await pool.shutdown_async()


async def test_concurrent_operations(project_path: Path):
    """Test concurrent get/return operations."""
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )
    try:
        async def use_process(task_id: int):
            async with await pool.get_process_async() as process:
                await process.send_command_async(f"#check Nat")
                await asyncio.sleep(0.01)  # Simulate some work
            return task_id

        # Run multiple tasks concurrently
        tasks = [use_process(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert set(results) == set(range(10))

        # All processes should be returned
        assert pool._num_used_processes == 0
    finally:
        await pool.shutdown_async()


async def run_all_tests():
    """Run all tests sequentially, creating fresh resources for each."""
    project_path = get_project_path()

    # List of all test functions
    tests = [
        ("test_mathlib_import_caching", test_mathlib_import_caching),
        ("test_basic_get_return_process", test_basic_get_return_process),
        ("test_multiple_processes", test_multiple_processes),
        ("test_max_out_processes", test_max_out_processes),
        ("test_get_process_non_blocking", test_get_process_non_blocking),
        ("test_get_process_blocking", test_get_process_blocking),
        ("test_context_manager", test_context_manager),
        ("test_env_setup_async", test_env_setup_async),
        ("test_shutdown_cleans_up_processes", test_shutdown_cleans_up_processes),
        ("test_process_reuse", test_process_reuse),
        ("test_concurrent_operations", test_concurrent_operations),
    ]

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running {test_name}...")
        print(f"{'=' * 60}")

        try:
            # Run the test (each test creates its own pool)
            await test_func(project_path)
            print(f"✓ {test_name} passed")
        except AssertionError as e:
            print(f"\n❌ {test_name} failed with assertion error:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            return 1
        except Exception as e:
            print(f"\n❌ {test_name} failed with error:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n{'=' * 60}")
    print("All tests passed!")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
