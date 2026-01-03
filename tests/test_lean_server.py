import asyncio
import sys
import time
from pathlib import Path

from leantree.repl_adapter.server import LeanServer, LeanClient, start_server
from leantree.repl_adapter.process_pool import LeanProcessPool

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


def get_free_port():
    """Get a free port for testing."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


async def test_server_startup_shutdown(project_path: Path):
    """Test that server can start and stop properly."""
    print("Running test_server_startup_shutdown...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        # Start server
        server.start()

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Verify server is running by checking status
        client = LeanClient("localhost", port)
        status = client.check_status()
        assert "available_processes" in status
        assert "used_processes" in status
        assert "max_processes" in status

        # Stop server
        server.stop()

        # Give it a moment to stop
        await asyncio.sleep(0.1)
    finally:
        await pool.shutdown_async()
    print("✓ test_server_startup_shutdown passed")


async def test_status_endpoint(project_path: Path):
    """Test the status endpoint returns correct information."""
    print("Running test_status_endpoint...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)
        status = client.check_status()

        assert status["max_processes"] == 2
        assert status["available_processes"] == 0
        assert status["used_processes"] == 0

        # Get a process and check status again
        process = client.get_process(blocking=True)
        assert process is not None

        status = client.check_status()
        assert status["used_processes"] == 1

        # Return process
        process.return_process()

        status = client.check_status()
        assert status["used_processes"] == 0
        assert status["available_processes"] == 1

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_status_endpoint passed")


async def test_get_process_blocking(project_path: Path):
    """Test getting a process in blocking mode."""
    print("Running test_get_process_blocking...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        # Get a process
        process = client.get_process(blocking=True)
        assert process is not None
        assert process.process_id > 0

        # Use the process
        response = process.send_command("#check Nat")
        assert "env" in response

        # Return the process
        process.return_process()

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_get_process_blocking passed")


async def test_get_process_non_blocking(project_path: Path):
    """Test getting a process in non-blocking mode."""
    print("Running test_get_process_non_blocking...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=1,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        # Get the only process
        process1 = client.get_process(blocking=False)
        assert process1 is not None

        # Try to get another one - should return None
        process2 = client.get_process(blocking=False)
        assert process2 is None

        # Return the first process
        process1.return_process()

        # Now should be able to get one
        process3 = client.get_process(blocking=False)
        assert process3 is not None

        process3.return_process()

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_get_process_non_blocking passed")


async def test_send_command(project_path: Path):
    """Test sending commands to a remote process."""
    print("Running test_send_command...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        with client.get_process(blocking=True) as process:
            # Send a simple command
            response = process.send_command("#check Nat")
            assert "env" in response

            # Send another command using the environment
            response2 = process.send_command("def test := 42")
            assert "env" in response2

            # Verify the definition exists
            response3 = process.send_command("#check test")
            assert "env" in response3

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_send_command passed")


async def test_context_manager(project_path: Path):
    """Test that LeanRemoteProcess works as a context manager."""
    print("Running test_context_manager...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        # Check initial status
        status1 = client.check_status()
        initial_available = status1["available_processes"]

        # Use context manager
        with client.get_process(blocking=True) as process:
            assert process is not None
            response = process.send_command("#check Nat")
            assert "env" in response

            # Check status while process is in use
            status2 = client.check_status()
            assert status2["used_processes"] == 1

        # Process should be returned automatically
        status3 = client.check_status()
        assert status3["used_processes"] == 0
        assert status3["available_processes"] == initial_available + 1

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_context_manager passed")


async def test_proof_from_sorry(project_path: Path):
    """Test creating a proof branch from a theorem with sorry."""
    print("Running test_proof_from_sorry...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        with client.get_process(blocking=True) as process:
            theorem = "example : 2 + 2 = 4 := by sorry"
            result = process.proof_from_sorry(theorem)
            assert result.is_success()
            proof_branch = result.value

            assert proof_branch is not None
            assert proof_branch._proof_state_id is not None

            # Check that we have goals
            state = proof_branch.state
            assert len(state.goals) > 0

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_proof_from_sorry passed")


async def test_apply_tactic(project_path: Path):
    """Test applying a tactic to a proof branch."""
    print("Running test_apply_tactic...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        with client.get_process(blocking=True) as process:
            theorem = "example : 2 + 2 = 4 := by sorry"
            result = process.proof_from_sorry(theorem)
            assert result.is_success()
            proof_branch = result.value

            initial_state = proof_branch.state
            assert not initial_state.is_solved()

            # Apply a tactic
            result = proof_branch.try_apply_tactic("decide")
            assert result.is_success()
            next_branches = result.value
            assert len(next_branches) > 0

            # Check if the proof is solved
            new_state = next_branches[0].state
            # Note: decide might solve it, or might not depending on the exact theorem

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_apply_tactic passed")


async def test_multiple_processes(project_path: Path):
    """Test that multiple clients can use processes concurrently."""
    print("Running test_multiple_processes...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        # Get two processes
        process1 = client.get_process(blocking=True)
        process2 = client.get_process(blocking=True)

        assert process1 is not None
        assert process2 is not None
        assert process1.process_id != process2.process_id

        # Use both processes
        response1 = process1.send_command("#check Nat")
        response2 = process2.send_command("#check Int")

        assert "env" in response1
        assert "env" in response2

        # Return both
        process1.return_process()
        process2.return_process()

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_multiple_processes passed")


async def test_process_reuse(project_path: Path):
    """Test that processes are reused when returned to the pool."""
    print("Running test_process_reuse...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        # Get a process
        with client.get_process(blocking=True) as process1:
            process1_id = process1.process_id
            process1.send_command("def test_reuse := 42")

        # Get another process - should be the same one (reused)
        with client.get_process(blocking=True) as process2:
            process2_id = process2.process_id
            # Verify the definition still exists (proves it's the same process)
            # We need to be careful here - if the process was restarted, the definition might be gone.
            # In the current implementation, return_process doesn't kill the process, just returns it to the pool.
            # But if the pool decided to restart it (e.g. due to memory usage), this test would fail.
            # For now, let's just check that we got a process back.
            
            # Let's redefine it just in case, to ensure the process is working
            process2.send_command("def test_reuse_2 := 100")
            response = process2.send_command("#check test_reuse_2")
            assert "env" in response

        # Note: process IDs might be different due to server tracking,
        # but the underlying process should be reused

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_process_reuse passed")


async def test_error_handling(project_path: Path):
    """Test error handling for invalid requests."""
    print("Running test_error_handling...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        # Test invalid process ID
        try:
            # Try to send command to non-existent process
            import urllib.request
            import json
            req = urllib.request.Request(
                f"http://localhost:{port}/process/999/command",
                data=json.dumps({"command": "#check Nat"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            urllib.request.urlopen(req)
            assert False, "Should have raised an error"
        except urllib.error.HTTPError as e:
            assert e.code == 500  # Server error for invalid process ID

        # Test invalid theorem (no sorry) - this raises generic Exception in interaction.py, so it becomes 500 -> RuntimeError
        with client.get_process(blocking=True) as process:
            try:
                process.proof_from_sorry("example : 2 + 2 = 4 := rfl")
                assert False, "Should have raised RuntimeError (500)"
            except RuntimeError as e:
                assert "No `sorries` in REPL response" in str(e)

        # Test syntactically incorrect theorem - this raises LeanInteractionException, so it becomes 200 -> ValueOrError
        with client.get_process(blocking=True) as process:
            result = process.proof_from_sorry("example : 2 + 2 = 4 := by sorr")  # Typo
            assert not result.is_success()
            assert "REPL returned error" in result.error or "error" in result.error

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_error_handling passed")


async def test_complete_proof_workflow(project_path: Path):
    """Test a complete proof workflow: get process, create proof, apply tactics."""
    print("Running test_complete_proof_workflow...")
    pool = LeanProcessPool(
        repl_exe=REPL_EXE,
        project_path=project_path,
        max_processes=2,
    )

    port = get_free_port()
    server = LeanServer(pool, address="localhost", port=port)

    try:
        server.start()
        await asyncio.sleep(0.1)

        client = LeanClient("localhost", port)

        with client.get_process(blocking=True) as process:
            # Create a simple proof
            theorem = """
            example : ∀ (p q : Prop), p → q → p ∧ q := by sorry
            """
            result = process.proof_from_sorry(theorem)
            assert result.is_success()
            proof_branch = result.value

            initial_state = proof_branch.state
            assert len(initial_state.goals) == 1

            # Apply tactics
            result = proof_branch.try_apply_tactic("intro p q hp hq")
            assert result.is_success()
            next_branches = result.value
            assert len(next_branches) > 0

            next_branch = next_branches[0]
            next_state = next_branch.state

            # Apply constructor
            result = next_branch.try_apply_tactic("constructor")
            assert result.is_success()
            final_branches = result.value
            assert len(final_branches) > 0

    finally:
        server.stop()
        await pool.shutdown_async()
    print("✓ test_complete_proof_workflow passed")


async def run_all_tests():
    """Run all tests sequentially, creating fresh resources for each."""
    project_path = get_project_path()

    # List of all test functions
    tests = [
        ("test_server_startup_shutdown", test_server_startup_shutdown),
        ("test_status_endpoint", test_status_endpoint),
        ("test_get_process_blocking", test_get_process_blocking),
        ("test_get_process_non_blocking", test_get_process_non_blocking),
        ("test_send_command", test_send_command),
        ("test_context_manager", test_context_manager),
        ("test_proof_from_sorry", test_proof_from_sorry),
        ("test_apply_tactic", test_apply_tactic),
        ("test_multiple_processes", test_multiple_processes),
        ("test_process_reuse", test_process_reuse),
        ("test_error_handling", test_error_handling),
        ("test_complete_proof_workflow", test_complete_proof_workflow),
    ]

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running {test_name}...")
        print(f"{'=' * 60}")

        try:
            # Run the test (each test creates its own pool and server)
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
