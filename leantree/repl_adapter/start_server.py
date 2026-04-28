"""CLI script to start the Lean server."""

import argparse
import atexit
import faulthandler
import logging
import os
import resource
import signal
import sys
import termios
import threading
from pathlib import Path
import time

from leantree.repl_adapter.server import start_server, LeanClient
from leantree.repl_adapter.process_pool import LeanProcessPool


logger = logging.getLogger(__name__)

# The OS default of 1024 is too tight for the REPL subprocesses + concurrent HTTP clients + their transient TIME_WAIT sockets.
DEFAULT_NOFILE_SOFT_LIMIT = 65536


def _raise_nofile_soft_limit(target: int = DEFAULT_NOFILE_SOFT_LIMIT) -> None:
    """Raise RLIMIT_NOFILE up to ``min(target, current_hard_limit)``.

    No-op on platforms without the resource module (e.g. Windows).
    Logs the before/after values so the actual limit is visible at startup.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (AttributeError, ValueError, OSError) as e:
        logger.error(f"Could not query RLIMIT_NOFILE: {e}")
        return
    new_soft = min(target, hard)
    if new_soft <= soft:
        logger.debug(f"RLIMIT_NOFILE soft limit already at {soft} (>= requested {target}); leaving as-is")
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError) as e:
        logger.error(f"Could not raise RLIMIT_NOFILE from {soft} to {new_soft}: {e}")
        return
    logger.debug(f"Raised RLIMIT_NOFILE soft limit from {soft} to {new_soft} (hard={hard})")

# Terminal settings preservation.
# The keyboard_monitor thread uses input() to read keypresses, which can modify
# terminal settings (e.g., disabling echo mode). If the process is killed via
# Ctrl+C while input() is active, these settings may not be restored, leaving
# the terminal in a broken state (characters you type are not displayed).
# We save the original settings at startup and restore them on exit.
_original_terminal_settings = None
try:
    _original_terminal_settings = termios.tcgetattr(sys.stdin)
except (termios.error, AttributeError):
    pass  # Not a terminal or termios not available


def _restore_terminal():
    """Restore terminal settings to their original state."""
    if _original_terminal_settings is not None:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _original_terminal_settings)
        except (termios.error, AttributeError):
            pass


atexit.register(_restore_terminal)


def _print_status(client: LeanClient):
    """Fetch and print server status. Used by both the Enter-key monitor and SIGUSR2."""
    try:
        status = client.check_status()
        print(f"\n=== Server Status ===")
        # ``used`` is processes held by actors right now; the
        # rest are pool states.  ``stopping`` is the janitor
        # backlog - persistently >0 means subprocess teardown
        # is wedged.
        print(
            f"Pool: {status['used_processes']} used, "
            f"{status['available_processes']} available, "
            f"{status['starting_processes']} starting, "
            f"{status['stopping_processes']} stopping  "
            f"({status['total_processes']}/{status['max_processes']} alive)"
        )
        print(f"RAM: {status['ram']['percent']:.1f}% used "
              f"({status['ram']['used_bytes'] / (1024**3):.1f}GB / "
              f"{status['ram']['total_bytes'] / (1024**3):.1f}GB)")
        avg_cpu = sum(status['cpu_percent_per_core']) / len(status['cpu_percent_per_core'])
        print(f"CPU: {avg_cpu:.1f}% average across {len(status['cpu_percent_per_core'])} cores")

        idle_too_long = status.get('idle_too_long_60s', 0)
        inactive_br = status.get('inactive_branches_60s', 0)
        total_br = status.get('total_branches', 0)
        print(
            f"Idle >60s: {idle_too_long}/{status['total_processes']} pool entries, "
            f"{inactive_br}/{total_br} branches"
        )

        # Per-process memory. Snapshot pool.checkpoints.keys() so we don't iterate
        # a dict being mutated by the event loop. Skip processes that fail to report
        # (may be mid-shutdown) and keep RSS/PSS aligned by appending together.
        # rss_values: list[float] = []
        # pss_values: list[float] = []
        # for process in list(pool.checkpoints.keys()):
        #     try:
        #         rss = process.memory_usage_rss()
        #         pss = process.memory_usage_pss()
        #     except Exception:
        #         continue
        #     rss_values.append(rss / (1024**3))
        #     pss_values.append(pss / (1024**3))
        # sorted_pairs = sorted(zip(rss_values, pss_values), key=lambda p: p[0], reverse=True)
        # print(f"PSS (GB): {' '.join(f'{p:.2f}' for _, p in sorted_pairs)}")
        # print(f"RSS (GB): {' '.join(f'{r:.2f}' for r, _ in sorted_pairs)}")

        # Show active requests
        active_requests = status.get('active_requests', [])
        if active_requests:
            print(f"Active requests ({len(active_requests)}):")
            for req in active_requests:
                print(f"  - {req['path']} ({req['duration_seconds']}s, thread: {req['thread']})")
        else:
            print("Active requests: none")
        print()
    except Exception as e:
        print(f"Error getting status: {e}")


def keyboard_monitor(address: str, port: int, pool: LeanProcessPool):
    """Print server status each time the user hits Enter."""
    client = LeanClient(address, port)
    while True:
        try:
            input()  # Wait for Enter key
            _print_status(client)
        except EOFError:
            # stdin closed, exit the thread
            break


def main():
    """CLI entry point for the Lean server."""
    # Line-buffer stdout so nohup-redirected log files show diagnostic prints immediately (RLIMIT_NOFILE raise, RLIMIT_AS
    # default, warmup progress, SIGUSR1 registration) rather than waiting for block-buffer flush. Without this, empty log
    # files mid-run masked useful startup info.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

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
    parser.add_argument(
        "--warmup-batch-size",
        type=int,
        default=32,
        help="Start warmup processes in batches of this size to limit resource contention; <=0 disables batching (default: 32)"
    )
    parser.add_argument(
        "--warmup-timeout",
        type=float,
        default=300.0,
        help="Per-process timeout in seconds for the env-setup `import` command run during warmup. May need raising when many subprocesses race the disk for Mathlib oleans (default: 300)"
    )
    parser.add_argument(
        "--rss-hard-limit-gib",
        type=float,
        default=32.0,
        help="Per-subprocess RLIMIT_AS ceiling in GiB; <=0 disables (default: 32)"
    )
    parser.add_argument(
        "--pss-recycle-limit-gib",
        type=float,
        default=4.0,
        help="Recycle a process whose PSS on return exceeds this, in GiB; <=0 disables (default: 4)"
    )

    args = parser.parse_args()

    # Apply --log-level to the root logger so every module-level logger
    # (process_pool, interaction, server, ...) inherits it.
    logging.getLogger().setLevel(args.log_level)

    # FD limit must be raised BEFORE the HTTP socket is opened or any subprocess
    # is spawned, so do it as early as possible after argparse.
    _raise_nofile_soft_limit()

    # Make `kill -USR1 <pid>` dump a Python traceback for every thread to stderr.
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        logger.debug("Registered SIGUSR1 handler: kill -USR1 <pid> dumps all-thread tracebacks to stderr")
    except (AttributeError, ValueError, OSError) as e:
        logger.error(f"Could not register SIGUSR1 faulthandler: {e}")

    # Make `kill -USR2 <pid>` print server status (same output as pressing Enter).
    # The handler dispatches to a thread because check_status() does HTTP I/O,
    # which is unsafe to run inside a signal handler.
    def _usr2_print_status(sig, frame):
        threading.Thread(
            target=_print_status,
            args=(LeanClient(args.address, args.port),),
            daemon=True,
        ).start()
    try:
        signal.signal(signal.SIGUSR2, _usr2_print_status)
        logger.debug("Registered SIGUSR2 handler: kill -USR2 <pid> prints server status")
    except (AttributeError, ValueError, OSError) as e:
        logger.error(f"Could not register SIGUSR2 handler: {e}")

    # Determine repl_exe path
    if args.repl_exe:
        repl_exe = Path(args.repl_exe)
    elif os.getenv("LEAN_REPL_EXE"):
        repl_exe = Path(os.getenv("LEAN_REPL_EXE"))
    else:
        raise ValueError("REPL executable not specified")

    if not repl_exe.exists():
        logger.error(f"REPL executable not found at {repl_exe}")
        logger.error("Please specify --repl-exe or set LEAN_REPL_EXE environment variable")
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
        logger.error(f"Project path not found at {project_path}")
        logger.error("Please specify --project-path or set LEAN_PROJECT_PATH environment variable")
        sys.exit(1)

    # Prepare imports if requested
    env_setup_async = None
    if args.imports:
        async def setup_imports(process):
            imports_str = "\n".join(f"import {imp}" for imp in args.imports)
            await process.send_command_async(imports_str, timeout=args.warmup_timeout)
        env_setup_async = setup_imports

    # Create process pool
    rss_hard_limit = int(args.rss_hard_limit_gib * 1024**3) if args.rss_hard_limit_gib > 0 else None
    pss_recycle_limit = int(args.pss_recycle_limit_gib * 1024**3) if args.pss_recycle_limit_gib > 0 else None
    pool = LeanProcessPool(
        repl_exe=repl_exe,
        project_path=project_path,
        max_processes=args.max_processes,
        env_setup_async=env_setup_async,
        rss_hard_limit=rss_hard_limit,
        pss_recycle_limit=pss_recycle_limit,
    )

    # Start server
    logger.info(f"Lean project: {project_path}")
    logger.info(f"REPL executable: {repl_exe}")
    if args.imports:
        logger.info(f"Importing packages: {", ".join(args.imports)}")
    warmup_batch_size = args.warmup_batch_size if args.warmup_batch_size and args.warmup_batch_size > 0 else None
    if args.warmup:
        if warmup_batch_size:
            logger.info(f"Warming up {args.max_processes} processes in batches of {warmup_batch_size} before accepting requests")
        else:
            logger.info(f"Warming up {args.max_processes} processes before accepting requests")
    server = start_server(
        pool,
        address=args.address,
        port=args.port,
        log_level=args.log_level,
        warmup=args.warmup,
        warmup_batch_size=warmup_batch_size,
    )
    logger.info(f"Server accepting requests on http://{args.address}:{args.port} with log level {args.log_level}")

    # Handle shutdown gracefully
    _shutting_down = False

    def signal_handler(sig, frame):
        nonlocal _shutting_down
        if _shutting_down:
            print("\nForced shutdown.")
            _restore_terminal()
            os._exit(1)
        _shutting_down = True
        print("\nShutting down server... (press Ctrl+C again to force quit)")
        # server.stop() stops both checked-out and available processes
        server.stop()
        _restore_terminal()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start keyboard monitoring thread
    keyboard_thread = threading.Thread(
        target=keyboard_monitor,
        args=(args.address, args.port, pool),
        daemon=True,
    )
    keyboard_thread.start()
    print("Press Enter to show server status")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
