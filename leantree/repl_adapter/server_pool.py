import time
import asyncio
from pathlib import Path
from typing import Callable, Coroutine
import psutil
from leantree.repl_adapter.interaction import LeanServer
from leantree.utils import Logger, NullLogger, to_sync


class LeanServerPool:
    """
    A pool of LeanServer instances for parallel processing.

    This class manages a pool of LeanServer instances, handling their creation,
    allocation, and recycling. It also monitors memory usage and restarts servers
    that exceed memory thresholds.
    """

    def __init__(
            self,
            repl_exe: Path,
            project_path: Path,
            max_servers: int,
            max_memory_utilization: float = 80.0,  # percentage
            env_setup_async: Callable[[LeanServer], Coroutine] | None = None,
            logger: Logger | None = None,
    ):
        """
        Initialize the server pool.

        Args:
            repl_exe: Path to the Lean REPL executable
            project_path: Path to the Lean project
            max_servers: Maximum number of parallel servers
            max_memory_utilization: Maximum memory utilization as a percentage
            logger: Optional logger
        """
        self.repl_exe = repl_exe
        self.project_path = project_path
        self.max_servers = max_servers
        self.max_memory_utilization = max_memory_utilization
        self.logger = logger if logger else NullLogger()

        # Pool state
        self.available_servers: list[LeanServer] = []
        self._num_used_servers: int = 0
        self.lock = asyncio.Lock()  # Use asyncio.Lock instead of threading.RLock
        self.server_available_event = asyncio.Event()
        self.env_setup_async = env_setup_async
        # Calculate memory threshold per server based on total system memory
        total_memory = psutil.virtual_memory().total
        self.memory_threshold_per_server = int(total_memory * (self.max_memory_utilization / 100) / self.max_servers)

    async def _create_server_async(self) -> LeanServer:
        """Create a new LeanServer instance."""
        server = LeanServer(
            self.repl_exe,
            self.project_path,
            self.logger,
            pool=self,
        )
        await server.start_async()
        if self.env_setup_async:
            await self.env_setup_async(server)
        return server

    async def max_out_servers_async(self):
        """
        Start servers in parallel until we reach max_servers capacity.
        
        This method ensures that len(self.available_servers) + self._num_used_servers 
        equals self.max_servers by starting new servers in parallel.
        """
        async with self.lock:
            servers_to_start = self.max_servers - (len(self.available_servers) + self._num_used_servers)
            if servers_to_start <= 0:
                return
            
            self.logger.info(f"Starting {servers_to_start} servers in parallel")
            
            # Start servers in parallel.
            tasks = [self._create_server_async() for _ in range(servers_to_start)]
            new_servers = await asyncio.gather(*tasks)
            
            self.available_servers.extend(new_servers)
            
            if self.available_servers:
                self.server_available_event.set()
                
            self.logger.info(f"Started {len(new_servers)} servers. Available: {len(self.available_servers)}, Used: {self._num_used_servers}")

    async def get_server_async(self, blocking: bool = True) -> LeanServer | None:
        """
        Get a server from the pool asynchronously.

        Args:
            blocking: If True, wait until a server is available. If False, return None if no server is available.

        Returns:
            A LeanServer instance if available, None otherwise (only if blocking=False)
        """
        async with self.lock:
            if self.available_servers:
                server = self.available_servers.pop()
                if not self.available_servers:
                    self.server_available_event.clear()
                self._num_used_servers += 1
                return server

            # If we haven't reached max servers, create a new one.
            if self._num_used_servers < self.max_servers:
                server = await self._create_server_async()
                self._num_used_servers += 1
                return server

        # No servers available and at max capacity
        if not blocking:
            return None

        # Wait for a server to become available asynchronously
        while True:
            try:
                # Wait for the event to be set with a timeout
                await asyncio.wait_for(self.server_available_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Continue waiting if timeout occurs
                pass

            async with self.lock:
                if self.available_servers:
                    server = self.available_servers.pop()
                    self._num_used_servers += 1
                    if not self.available_servers:
                        self.server_available_event.clear()
                    return server

    async def return_server_async(self, server: LeanServer):
        """
        Return a server to the pool.

        If the server's memory usage exceeds the threshold, it will be terminated
        instead of being returned to the pool.

        Args:
            server: The LeanServer instance to return
        """

        # TODO: as a temporary solution, we always restart the server
        await server.stop_async()
        server = None

        # try:
        #     memory_usage = server.virtual_memory_usage()
        #     if self.memory_threshold_per_server and memory_usage > self.memory_threshold_per_server:
        #         self.logger.debug(
        #             f"Server memory usage ({memory_usage / (1024 * 1024):.2f} MB) exceeds threshold "
        #             f"({self.memory_threshold_per_server / (1024 * 1024):.2f} MB). Terminating."
        #         )
        #         await server.stop_async()
        #         server = None
        # except Exception as e:
        #     self.logger.warning(f"Error checking server memory: {e}. Terminating server.")
        #     await server.stop_async()
        #     server = None

        # if server is not None:
        #     await server.drain_repl_output_async()

        async with self.lock:
            assert self._num_used_servers > 0, "No servers in use"
            self._num_used_servers -= 1

            if server is not None:
                # Add back to available servers
                self.available_servers.append(server)
                # Set the event to notify waiting coroutines
                self.server_available_event.set()

    return_server = to_sync(return_server_async)

    def shutdown(self):
        """Shut down all servers in the pool."""

        async def _shutdown_async():
            async with self.lock:
                # Shut down available servers
                for server in self.available_servers:
                    try:
                        server.stop()
                    except Exception as e:
                        self.logger.warning(f"Error shutting down server: {e}")

                self.available_servers = []

        # Run the coroutine in the current event loop or create a new one if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there's no event loop in the current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async function synchronously
        loop.run_until_complete(_shutdown_async())