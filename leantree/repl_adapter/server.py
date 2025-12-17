import asyncio
import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Self
import urllib.request
import urllib.error

from leantree.repl_adapter.interaction import LeanProcess, LeanProofBranch, LeanInteractionException
from leantree.repl_adapter.process_pool import LeanProcessPool
from leantree.core.lean import LeanProofState, LeanTactic, LeanGoal
from leantree.utils import serialize_exception, deserialize_exception, ValueOrError


class LeanServer:
    """Manages a LeanProcessPool and exposes it over a HTTP port."""

    def __init__(self, pool: LeanProcessPool, address: str = "localhost", port: int = 8000, log_level: str = "INFO"):
        self.pool = pool
        self.address = address
        self.port = port
        self.log_level = log_level
        self.server = None
        self.server_thread = None

        # Setup logger
        self.logger = logging.getLogger("LeanServer")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

        self._process_id_counter = 0
        self._process_id_to_process = {}  # Maps process_id to LeanProcess
        self._process_to_id = {}  # Maps LeanProcess to process_id
        self._lock = threading.Lock()
        self._event_loop = None
        self._loop_thread = None

    def _get_process_id(self, process: LeanProcess) -> int:
        """Get or create a process ID for a LeanProcess."""
        with self._lock:
            if process in self._process_to_id:
                return self._process_to_id[process]
            self._process_id_counter += 1
            process_id = self._process_id_counter
            self._process_to_id[process] = process_id
            self._process_id_to_process[process_id] = process
            return process_id

    def _get_process(self, process_id: int) -> LeanProcess:
        """Get a LeanProcess by its ID."""
        with self._lock:
            if process_id not in self._process_id_to_process:
                raise ValueError(f"Process {process_id} not found")
            return self._process_id_to_process[process_id]

    def _remove_process(self, process_id: int):
        """Remove a process from tracking."""
        with self._lock:
            if process_id in self._process_id_to_process:
                process = self._process_id_to_process[process_id]
                del self._process_id_to_process[process_id]
                if process in self._process_to_id:
                    del self._process_to_id[process]

    def _run_async(self, coro):
        """Run an async coroutine in the event loop."""
        if self._event_loop is None:
            raise RuntimeError("Server not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return future.result()

    def start(self):
        """Start the HTTP server."""
        if self.server is not None:
            raise RuntimeError("Server already started")

        # Start event loop in a separate thread
        def run_event_loop():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()

        self._loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self._loop_thread.start()

        # Wait for event loop to be ready
        while self._event_loop is None:
            threading.Event().wait(0.01)

        handler = self._create_handler()
        self.server = HTTPServer((self.address, self.port), handler)

        def run_server():
            self.server.serve_forever()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def stop(self):
        """Stop the HTTP server."""
        if self.server is not None:
            self.server.shutdown()
            self.server = None
        if self._event_loop is not None:
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            self._event_loop = None

    def _create_handler(self):
        server = self

        class LeanServerHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.logger.debug(f"GET request to {self.path} from {self.client_address}")

                if self.path == "/status":
                    self._handle_status()
                else:
                    self._send_error(404, "Not Found")

            def do_POST(self):
                server.logger.debug(f"POST request to {self.path} from {self.client_address}")

                if self.path == "/process/get":
                    self._handle_get_process()
                elif self.path.startswith("/process/"):
                    parts = self.path.split("/")
                    if len(parts) >= 4:
                        process_id = int(parts[2])
                        action = parts[3]
                        if action == "command":
                            self._handle_command(process_id)
                        elif action == "return":
                            self._handle_return_process(process_id)
                        elif action == "proof_from_sorry":
                            self._handle_proof_from_sorry(process_id)
                        elif action == "proof" and len(parts) >= 6:
                            proof_state_id = int(parts[4])
                            if parts[5] == "apply_tactic":
                                self._handle_apply_tactic(process_id, proof_state_id)
                            elif parts[5] == "try_apply_tactic":
                                self._handle_try_apply_tactic(process_id, proof_state_id)
                            elif parts[5] == "state":
                                self._handle_proof_state(process_id, proof_state_id)
                            else:
                                self._send_error(404, "Not Found")
                        else:
                            self._send_error(404, "Not Found")
                    else:
                        self._send_error(404, "Not Found")
                else:
                    self._send_error(404, "Not Found")

            def _handle_status(self):
                pool = server.pool
                status = {
                    "available_processes": len(pool.available_processes),
                    "used_processes": pool._num_used_processes,
                    "max_processes": pool.max_processes,
                }
                self._send_json(200, status)

            def _handle_get_process(self):
                data = self._read_json()
                blocking = data.get("blocking", True)

                try:
                    process = server._run_async(server.pool.get_process_async(blocking=blocking))
                    if process is None:
                        self._send_json(200, {"process_id": None})
                    else:
                        process_id = server._get_process_id(process)
                        self._send_json(200, {"process_id": process_id})
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_command(self, process_id: int):
                try:
                    process = server._get_process(process_id)
                    data = self._read_json()
                    command = data["command"]

                    response = server._run_async(process.send_command_async(command))
                    self._send_json(200, response)
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_return_process(self, process_id: int):
                try:
                    process = server._get_process(process_id)
                    server._run_async(server.pool.return_process_async(process))
                    server._remove_process(process_id)
                    self._send_json(200, {"status": "ok"})
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_proof_from_sorry(self, process_id: int):
                try:
                    process = server._get_process(process_id)
                    data = self._read_json()
                    theorem_with_sorry = data["theorem_with_sorry"]

                    # Get the first proof branch
                    # proofs_from_sorries_async returns an AsyncIterator, so we need to collect it
                    async def collect_proof_branches():
                        return [branch async for branch in process.proofs_from_sorries_async(theorem_with_sorry)]

                    try:
                        proof_branches = server._run_async(collect_proof_branches())
                    except LeanInteractionException as e:
                        self._send_json(200, {"error": str(e)})
                        return

                    if len(proof_branches) == 0:
                        self._send_json(200, {"error": "No sorries found in theorem"})
                        return

                    proof_branch = proof_branches[0]
                    proof_state_id = proof_branch._proof_state_id
                    goals = [goal.serialize() for goal in proof_branch._all_goals]

                    # Store proof branch info (simplified - in real implementation might want to track these)
                    response = {
                        "proof_state_id": proof_state_id,
                        "goals": goals,
                    }
                    self._send_json(200, {"value": response})
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_apply_tactic(self, process_id: int, proof_state_id: int):
                try:
                    process = server._get_process(process_id)
                    data = self._read_json()
                    tactic = data["tactic"]

                    # Create a minimal proof branch to apply tactic
                    # In a real implementation, we'd track proof branches on the server
                    response_dict = server._run_async(process._send_to_repl_async({
                        "tactic": tactic,
                        "proofState": proof_state_id,
                    }))

                    # Extract goals and new proof state
                    new_proof_state_id = response_dict.get("proofState")
                    goals = LeanProcess._goals_from_response(response_dict)

                    response = {
                        "proof_state_id": new_proof_state_id,
                        "goals": [goal.serialize() for goal in goals],
                    }
                    self._send_json(200, response)
                except Exception as e:
                    self._send_error(500, str(e), exception=e)

            def _handle_try_apply_tactic(self, process_id: int, proof_state_id: int):
                try:
                    process = server._get_process(process_id)
                    data = self._read_json()
                    tactic = data["tactic"]

                    # Create a minimal proof branch to apply tactic
                    response_dict = server._run_async(process._send_to_repl_async({
                        "tactic": tactic,
                        "proofState": proof_state_id,
                    }))

                    step_error = LeanProofBranch.step_error_from_response(response_dict)
                    if step_error:
                        self._send_json(200, {"error": f"Step verification error: {step_error}"})
                        return

                    if "goals" not in response_dict:
                        self._send_json(200, {"error": f"Could not apply tactic in REPL: {json.dumps(response_dict)}"})
                        return

                    # Extract goals and new proof state
                    new_proof_state_id = response_dict.get("proofState")
                    goals = LeanProcess._goals_from_response(response_dict)

                    # Return a list of branches (simplified to 1 for now)
                    branch_data = {
                        "proof_state_id": new_proof_state_id,
                        "goals": [goal.serialize() for goal in goals],
                    }

                    self._send_json(200, {"value": [branch_data]})
                except Exception as e:
                    self._send_json(200, {"error": str(e)})

            def _handle_proof_state(self, process_id: int, proof_state_id: int):
                # This would need to track proof branches on the server
                # For now, return a simple response
                self._send_json(200, {"proof_state_id": proof_state_id})

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
                """Send an error response, optionally including a pickled exception."""
                error_data = {"error": message}

                if exception is not None:
                    # Serialize the exception using the utility function
                    exception_data = serialize_exception(exception)
                    error_data.update(exception_data)

                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(error_data).encode("utf-8"))

            def log_message(self, format, *args):
                # Suppress default logging
                pass

        return LeanServerHandler


class LeanClient:
    """Connects to a LeanServer."""

    def __init__(self, address: str, port: int):
        self.address = address
        self.port = port
        self.base_url = f"http://{address}:{port}"

    def _request(self, method: str, path: str, data: dict = None) -> dict:
        """Make an HTTP request to the server."""
        url = f"{self.base_url}{path}"
        if data is not None:
            data_bytes = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=data_bytes, method=method)
            req.add_header("Content-Type", "application/json")
        else:
            req = urllib.request.Request(url, method=method)

        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_data = json.loads(error_body)
                error_message = error_data.get("error", str(e))

                # Deserialize the exception using the utility function
                raise deserialize_exception(error_data, f"Error from LeanServer: {error_message}")
            except json.JSONDecodeError:
                raise RuntimeError(f"Error from LeanServer: {str(e)}")

    def check_status(self) -> dict:
        """Check the status of the server."""
        return self._request("GET", "/status")

    def get_process(self, blocking: bool = True) -> "LeanRemoteProcess | None":
        """Get a process from the server."""
        response = self._request("POST", "/process/get", {"blocking": blocking})
        process_id = response.get("process_id")
        if process_id is None:
            return None
        return LeanRemoteProcess(self, process_id)


class LeanRemoteProcess:
    """A remote Lean process managed by a LeanServer."""

    def __init__(self, client: LeanClient, process_id: int):
        self.client = client
        self.process_id = process_id

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs):
        """Return the process to the pool when exiting context."""
        self.return_process()

    def send_command(self, command: str) -> dict:
        """Send a command to the remote process."""
        return self.client._request(
            "POST",
            f"/process/{self.process_id}/command",
            {"command": command}
        )

    def return_process(self):
        """Return the process to the pool."""
        self.client._request("POST", f"/process/{self.process_id}/return")

    def proof_from_sorry(self, theorem_with_sorry: str) -> ValueOrError["RemoteLeanProofBranch"]:
        """Create a proof branch from a theorem with sorry."""
        response = self.client._request(
            "POST",
            f"/process/{self.process_id}/proof_from_sorry",
            {"theorem_with_sorry": theorem_with_sorry}
        )

        if "error" in response:
            return ValueOrError.from_error(response["error"])

        value = response["value"]
        return ValueOrError.from_success(RemoteLeanProofBranch(
            self.client,
            self.process_id,
            value["proof_state_id"],
            value["goals"]
        ))


class RemoteLeanProofBranch:
    """A remote proof branch managed by a LeanServer."""

    def __init__(self, client: LeanClient, process_id: int, proof_state_id: int, goals: list[dict]):
        self.client = client
        self.process_id = process_id
        self._proof_state_id = proof_state_id
        self._goals = [LeanGoal.deserialize(g) for g in goals]

    @property
    def state(self) -> LeanProofState:
        """Get the current proof state."""
        return LeanProofState(self._goals)

    def try_apply_tactic(
            self,
            tactic: LeanTactic | str,
            ban_search_tactics: bool = True,
    ) -> ValueOrError[list["RemoteLeanProofBranch"]]:
        """Apply a tactic to the proof branch."""
        if isinstance(tactic, LeanTactic):
            tactic = tactic.tactic

        response = self.client._request(
            "POST",
            f"/process/{self.process_id}/proof/{self._proof_state_id}/try_apply_tactic",
            {"tactic": tactic}
        )

        if "error" in response:
            return ValueOrError.from_error(response["error"])

        branches_data = response["value"]
        branches = []
        for branch_data in branches_data:
            branches.append(RemoteLeanProofBranch(
                self.client,
                self.process_id,
                branch_data["proof_state_id"],
                branch_data["goals"]
            ))

        return ValueOrError.from_success(branches)


def start_server(
        pool: LeanProcessPool,
        address: str = "localhost",
        port: int = 8000,
        log_level: str = "INFO"
) -> LeanServer:
    """Start a LeanServer with the given pool."""
    server = LeanServer(pool, address, port, log_level)
    server.start()
    return server


if __name__ == "__main__":
    from pathlib import Path
    from leantree.repl_adapter.process_pool import LeanProcessPool

    # Example usage
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
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        pool.shutdown()
