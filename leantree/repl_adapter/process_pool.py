import time
import asyncio
from pathlib import Path
from typing import Callable, Coroutine
import psutil
from leantree.repl_adapter.interaction import LeanProcess
from leantree.utils import Logger, NullLogger, to_sync

# TODO:
#  - caching based on imported libraries

class LeanProcessPool:
    """
    A pool of LeanProcess instances for parallel processing.

    This class manages a pool of LeanProcess instances, handling their creation,
    allocation, and recycling. It also monitors memory usage and restarts processes
    that exceed memory thresholds.
    """

    def __init__(
            self,
            repl_exe: Path,
            project_path: Path,
            max_processes: int,
            max_memory_utilization: float = 80.0,  # percentage
            env_setup_async: Callable[[LeanProcess], Coroutine] | None = None,
            logger: Logger | None = None,
    ):
        """
        Initialize the process pool.

        Args:
            repl_exe: Path to the Lean REPL executable
            project_path: Path to the Lean project
            max_processes: Maximum number of parallel processes
            max_memory_utilization: Maximum memory utilization as a percentage
            logger: Optional logger
        """
        self.repl_exe = repl_exe
        self.project_path = project_path
        self.max_processes = max_processes
        self.max_memory_utilization = max_memory_utilization
        self.logger = logger if logger else NullLogger()

        # Pool state
        self.available_processes: list[LeanProcess] = []
        self._num_used_processes: int = 0
        self.lock = asyncio.Lock()  # Use asyncio.Lock instead of threading.RLock
        self.process_available_event = asyncio.Event()
        self.env_setup_async = env_setup_async
        # Calculate memory threshold per server based on total system memory
        total_memory = psutil.virtual_memory().total
        self.memory_threshold_per_process = int(total_memory * (self.max_memory_utilization / 100) / self.max_processes)

        self._was_shutdown = False

    async def _create_process_async(self) -> LeanProcess:
        """Create a new LeanProcess instance."""
        process = LeanProcess(
            self.repl_exe,
            self.project_path,
            self.logger,
            pool=self,
        )
        await process.start_async()
        if self.env_setup_async:
            await self.env_setup_async(process)
        return process

    async def max_out_processes_async(self):
        """
        Start processes in parallel until we reach max_processes capacity.
        
        This method ensures that len(self.available_processes) + self._num_used_processes 
        equals self.max_processes by starting new processes in parallel.
        """
        async with self.lock:
            processes_to_start = self.max_processes - (len(self.available_processes) + self._num_used_processes)
            if processes_to_start <= 0:
                return
            
            self.logger.info(f"Starting {processes_to_start} processes in parallel")
            
            # Start processes in parallel.
            tasks = [self._create_process_async() for _ in range(processes_to_start)]
            new_processes = await asyncio.gather(*tasks)
            
            self.available_processes.extend(new_processes)
            
            if self.available_processes:
                self.process_available_event.set()
                
            self.logger.info(f"Started {len(new_processes)} processes. Available: {len(self.available_processes)}, Used: {self._num_used_processes}")

    async def get_process_async(self, blocking: bool = True) -> LeanProcess | None:
        """
        Get a process from the pool asynchronously.

        Args:
            blocking: If True, wait until a process is available. If False, return None if no process is available.

        Returns:
            A LeanProcess instance if available, None otherwise (only if blocking=False)
        """
        async with self.lock:
            if self.available_processes:
                process = self.available_processes.pop()
                if not self.available_processes:
                    self.process_available_event.clear()
                self._num_used_processes += 1
                return process

            # If we haven't reached max processes, create a new one.
            if self._num_used_processes < self.max_processes:
                process = await self._create_process_async()
                self._num_used_processes += 1
                return process

        # No processes available and at max capacity
        if not blocking:
            return None

        # Wait for a process to become available asynchronously
        while True:
            try:
                # Wait for the event to be set with a timeout
                await asyncio.wait_for(self.process_available_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Continue waiting if timeout occurs
                pass

            async with self.lock:
                if self.available_processes:
                    process = self.available_processes.pop()
                    self._num_used_processes += 1
                    if not self.available_processes:
                        self.process_available_event.clear()
                    return process

    async def return_process_async(self, process: LeanProcess):
        """
        Return a process to the pool.

        If the process's memory usage exceeds the threshold, it will be terminated
        instead of being returned to the pool.

        Args:
            process: The LeanProcess instance to return
        """

        async with self.lock:
            should_terminate = False
            if self._was_shutdown:
                should_terminate = True
            else:
                try:
                    memory_usage = process.virtual_memory_usage()
                    if self.memory_threshold_per_process and memory_usage > self.memory_threshold_per_process:
                        self.logger.debug(
                            f"Process memory usage ({memory_usage / (1024 * 1024):.2f} MB) exceeds threshold "
                            f"({self.memory_threshold_per_process / (1024 * 1024):.2f} MB). Terminating."
                        )
                        should_terminate = True
                except Exception as e:
                    self.logger.warning(f"Error checking process memory: {e}. Terminating process.")
                    should_terminate = True

            if should_terminate:
                await process.stop_async()
                process = None

            if process is not None:
                await process.drain_repl_output_async()

            assert self._num_used_processes > 0, "No processes in use"
            self._num_used_processes -= 1

            if process is not None:
                # Add back to available processes
                self.available_processes.append(process)
                # Set the event to notify waiting coroutines
                self.process_available_event.set()

    return_process = to_sync(return_process_async)

    async def shutdown_async(self):
        """Shut down all processes in the pool asynchronously."""
        async with self.lock:
            if self._was_shutdown:
                return
            self._was_shutdown = True

            # Shut down available processes
            for process in self.available_processes:
                try:
                    await process.stop_async()
                except Exception as e:
                    self.logger.warning(f"Error shutting down process: {e}")
            self.available_processes = []

    shutdown = to_sync(shutdown_async)