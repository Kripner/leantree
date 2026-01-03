import asyncio
import json
import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self, Any, AsyncIterator

import psutil

from leantree import utils
from leantree.core.abstraction import ProofBranch
from leantree.core.lean import LeanGoal, LeanTactic, LeanProofState
from leantree.core.lean_file import LeanTheorem, LeanFile, StoredError
from leantree.file_span import FilePosition, FileSpan
from leantree.metavar_graph import MetavarGraph
from leantree.repl_adapter.data import ReplGoalInfo, ReplCompilationUnit, FilePositionParser, ReplProofStepInfo
from leantree.utils import is_just_comments, ValueOrError, get_source_with_sorries, to_sync, to_sync_iterator


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
_eq_sorry_pattern = re.compile(r'\b:=\s*sorry\b')


class LeanProcess:
    def __init__(self, repl_exe: Path, project_path: Path, logger: utils.Logger = None, pool: Any = None):
        self.repl_exe = repl_exe
        self.project_path = project_path
        self.logger = logger if logger else utils.NullLogger()
        self.pool = pool

        self._proc = None
        self._env_id = None
        self._stderr_buffer = deque(maxlen=50)
        self._stderr_task = None

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
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.project_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # The limit argument sets the buffer limit for StreamReader wrappers for Process.stdout and Process.stderr.
            limit=16 * 1024 * 1024,  # 16 MB
        )
        self._stderr_task = asyncio.create_task(self._monitor_stderr())

    start = to_sync(start_async)

    async def stop_async(self):
        """Stop the Lean REPL asynchronously."""
        assert self._proc is not None
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass
        # See https://github.com/python/cpython/issues/119710#issuecomment-2425168469
        # and https://github.com/python/cpython/issues/88050
        # on why this line is necessary (otherwise the wait() call hangs).
        self._proc._transport.close()
        await self._proc.wait()

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
        if self.pool:
            # If this is a managed process, return it to the pool instead of terminating.
            await self.pool.return_process_async(self)
        else:
            await self.stop_async()

    def __enter__(self) -> Self:
        """Synchronous context manager entry."""
        if not self._proc:
            self.start()
        return self

    def __exit__(self, *args, **kwargs):
        """Synchronous context manager exit."""
        if self.pool:
            # If this is a managed process, return it to the pool instead of terminating.
            self.pool.return_process(self)
        else:
            self.stop()

    def checkpoint(self) -> LeanEnvironmentCheckpoint:
        return LeanEnvironmentCheckpoint(self._env_id)

    def rollback_to(self, checkpoint: LeanEnvironmentCheckpoint):
        self._env_id = checkpoint.env_id

    async def _send_to_repl_async(self, data: dict) -> dict:
        """Send data to the REPL asynchronously and return the response."""
        self._assert_started()
        serialized = json.dumps(data, ensure_ascii=False) + "\n\n"

        self.logger.debug(f"Sending to REPL: '{serialized[:-2]}'")
        try:
            self._proc.stdin.write(serialized.encode('utf-8'))
            await self._proc.stdin.drain()

            response_lines = []
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    # EOF.
                    raise LeanProcessException("Lean REPL process ended unexpectedly")

                decoded_line = line.decode('utf-8')
                if decoded_line.strip() == "":
                    # Empty line marks end of response.
                    break
                response_lines.append(decoded_line)
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

    async def send_command_async(self, command: str, proof_trees: bool = False, info_trees: bool = False) -> dict:
        """Send a command to the REPL asynchronously and return the response."""
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

        response = await self._send_to_repl_async(data)

        assert "env" in response, f"no `env` in REPL response with keys: {response.keys()}"
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
            raise Exception(
                "Subprocess not started. Use 'with LeanProcess(...) as env:' or 'async with LeanProcess(...) as env:'"
            )
        if self._proc.returncode is not None:
            stderr_tail = "\n".join(self._stderr_buffer)
            raise Exception(
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
                assert len(sorry_branches) == len(unit.theorem.by_blocks)
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
                self._proc.kill()
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

    def virtual_memory_usage(self) -> int:
        """
        Returns the virtual memory size (VMS) of the Lean REPL process in bytes.
        """
        self._assert_started()
        try:
            process = psutil.Process(self._proc.pid)
            mem_info = process.memory_info()
            return mem_info.vms
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error) as e:
            raise LeanProcessException(f"Failed to get memory usage", e)

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

    async def _send_tactic_async(self, tactic: str, timeout: int | None = 1000) -> dict:
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
            # Tactics rw?, apply?, exact? technically close the main goal, but the proof is invalid. Setting
            # ban_search_tactics disallows these. Consider e.g.:
            # example : 1 = 0 := by
            #   apply?
            ban_search_tactics: bool = True,
            timeout: int | None = 1000,
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

    async def try_apply_tactic_async(self, tactic: LeanTactic | str, timeout: int | None = 1000) -> ValueOrError[list[Self]]:
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
        # TODO: a better solution would be to report the `sorry` introduced by `apply?` and allow it (it seems that
        #  apply? creates a sorry internally)
        if ban_search_tactics and any(tactic.startswith(banned) for banned in ["apply?", "rw?", "exact?"]):
            raise LeanInteractionException("Search tactics (apply?, rw?, exact?) are not allowed.")

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
