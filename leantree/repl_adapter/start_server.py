"""CLI script to start the Lean server."""

import argparse
import os
import signal
import sys
from pathlib import Path

from leantree.repl_adapter.server import start_server
from leantree.repl_adapter.process_pool import LeanProcessPool
from leantree.utils import Logger, LogLevel


def main():
    """CLI entry point for the Lean server."""
    parser = argparse.ArgumentParser(description="Start a Lean server")
    parser.add_argument(
        "--address",
        type=str,
        default="localhost",
        help="Server address (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--repl-exe",
        type=str,
        default=None,
        help="Path to Lean REPL executable (default: from LEAN_REPL_EXE)"
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Path to Lean project (default: from LEAN_PROJECT_PATH env or ./leantree_project)"
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=8,
        help="Maximum number of parallel processes (default: 2)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--imports",
        type=str,
        nargs="*",
        help="List of Lean packages to import"
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Pre-start all processes to max capacity before accepting requests"
    )

    args = parser.parse_args()

    # Determine repl_exe path
    if args.repl_exe:
        repl_exe = Path(args.repl_exe)
    elif os.getenv("LEAN_REPL_EXE"):
        repl_exe = Path(os.getenv("LEAN_REPL_EXE"))
    else:
        raise ValueError("REPL executable not specified")

    if not repl_exe.exists():
        print(f"Error: REPL executable not found at {repl_exe}", file=sys.stderr)
        print("Please specify --repl-exe or set LEAN_REPL_EXE environment variable", file=sys.stderr)
        sys.exit(1)

    # Determine project_path
    if args.project_path:
        project_path = Path(args.project_path)
    elif os.getenv("LEAN_PROJECT_PATH"):
        project_path = Path(os.getenv("LEAN_PROJECT_PATH"))
    else:
        # Default relative to current working directory
        project_path = Path("leantree_project").resolve()

    if not project_path.exists():
        print(f"Error: Project path not found at {project_path}", file=sys.stderr)
        print("Please specify --project-path or set LEAN_PROJECT_PATH environment variable", file=sys.stderr)
        sys.exit(1)

    # Create process pool
    env_setup_async = None
    if args.imports:
        async def setup_imports(process):
            imports_str = "\n".join(f"import {imp}" for imp in args.imports)
            await process.send_command_async(imports_str)
        env_setup_async = setup_imports

    pool = LeanProcessPool(
        repl_exe=repl_exe,
        project_path=project_path,
        max_processes=args.max_processes,
        logger=Logger(LogLevel.DEBUG) if args.log_level == "DEBUG" else None,
        env_setup_async=env_setup_async,
    )

    # Start server
    server = start_server(
        pool,
        address=args.address,
        port=args.port,
        log_level=args.log_level
    )

    # Warmup: pre-start all processes if requested (must be after server starts to use its event loop)
    if args.warmup:
        print(f"Warming up {args.max_processes} processes...")
        server._run_async(pool.max_out_processes_async())
        print("Warmup complete.")
    print(f"Lean project: {project_path}")
    print(f"REPL executable: {repl_exe}")
    if args.imports:
        print(f"Importing packages: {", ".join(args.imports)}")
    print(f"Server started on http://{args.address}:{args.port} with log level {args.log_level}")

    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        # Shut down pool on the server's event loop before stopping it
        # (pool's asyncio primitives are bound to that loop)
        server._run_async(pool.shutdown_async())
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
