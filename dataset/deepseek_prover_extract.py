import argparse
import traceback
from pathlib import Path
from typing import Any
import json
import time

from tqdm import tqdm
from datasets import load_dataset

from leantree import utils
from leantree.core.lean_file import LeanTheorem
from leantree.core.project import LeanProject
from leantree.repl_adapter.interaction import LeanProcessException, LeanServer
from leantree.utils import Logger, LogLevel


# Note: you have to be logged in using huggingface-cli.


# TODO: Errors like these could also be ignored, since we probably don't want such samples in the dataset:
# Could not apply tactic in REPL: {"message": "Lean error:\n(deterministic) timeout at `aesop`, maximum number of heartbeats (200000) has been rea
# ched\nUse `set_option maxHeartbeats <num>` to set the limit.(invalid MessageData.lazy, missing context)"}

# TODO: project.load_theorem reports errors differently from project.load_file

def create_environment(project: LeanProject, header: str) -> Any:
    """Create a new environment and initialize it with the header"""
    # env = project.environment(repl_path, Logger(LogLevel.DEBUG)).__enter__()
    env = project.environment().__enter__()
    if header:
        env.send_command(header)
    return env


def process_theorem(
        project: LeanProject,
        env: Any,
        name: str,
        statement: str,
        proof: str,
        timeout: int
) -> tuple[LeanTheorem | None, str | None]:
    """Process a single theorem, using utils.timeout for timeouts."""
    full_theorem = f"{statement}{proof}"
    try:
        if timeout > 0:
            # run load_theorem under our timeout wrapper
            theorem = utils.timeout(lambda: project.load_theorem(full_theorem, env), timeout)
        else:
            theorem = project.load_theorem(full_theorem, env)

        theorem.name = name
        return theorem, None
    except TimeoutError:
        return None, f"timeout after {timeout}s"
    except LeanProcessException:
        raise
    except Exception as e:
        traceback.print_exc()
        return None, str(e)


def write_result(result: dict, out_file):
    """Write successful result to output file"""
    json.dump(result, out_file, ensure_ascii=False)
    out_file.write("\n")


def write_error(sample: dict, error: str, err_file):
    """Write error record to errors file"""
    error_record = {
        "name": sample["name"],
        "header": sample["header"],
        "statement": sample["formal_statement"],
        "proof": sample["formal_proof"],
        "error": error
    }
    json.dump(error_record, err_file, ensure_ascii=False)
    err_file.write("\n")


def is_skipped_theorem(error: str) -> bool:
    """
    Check if a theorem should be considered skipped based on error message. True if:
    - the errors are due to `unknown identifier`, which means the proof is is broken due to different Lean version
    """
    prefix = "REPL returned error messages: "
    if not error.startswith(prefix):
        return False
    json_str = error[len(prefix):]

    try:
        messages = json.loads(json_str)
    except json.JSONDecodeError:
        return False
    if not isinstance(messages, list):
        return False

    # Check all error messages are "unknown identifier"
    for msg in messages:
        if msg.get("severity") == "error" and not msg.get("data", "").startswith("unknown identifier"):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract and process DeepSeek Prover dataset")
    parser.add_argument("output", type=Path, help="Output JSONL file path")
    parser.add_argument("--repl_path", type=Path, required=True, help="Path to Lean REPL executable")
    parser.add_argument("--project_path", type=Path, required=True, help="Path to Lean project")
    parser.add_argument("--max_env_steps", type=int, default=100,
                        help="Maximum number of theorems to process before recreating environment")
    parser.add_argument("--force", action="store_true", help="Overwrite output files if they exist")
    parser.add_argument("--dataset", type=str, default="deepseek-ai/DeepSeek-Prover-V1",
                        help="HuggingFace dataset name")
    parser.add_argument("--only", nargs="+", help="Only process theorems with these names")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout in seconds for load_theorem; 0 = no timeout")
    args = parser.parse_args()

    output_path = args.output
    errors_path = output_path.with_suffix(".errors")

    # Check if output files exist
    if not args.force:
        if output_path.exists():
            print(f"Error: Output file {output_path} already exists. Use --force to overwrite.")
            return
        if errors_path.exists():
            print(f"Error: Errors file {errors_path} already exists. Use --force to overwrite.")
            return

    # Setup project and load dataset
    project = LeanProject(args.project_path, args.repl_path, Logger(LogLevel.INFO))
    dataset = load_dataset(args.dataset, split="train")

    # Initialize counters
    total = successful = failed = skipped = timed_out = 0
    current_header = None
    env: LeanServer | None = None
    env_steps = 0

    def restart_env():
        nonlocal env, current_header, env_steps
        print(f"Creating new LeanEnvironment.")
        if env is not None:
            env.__exit__(None, None, None)
        env = create_environment(project, sample["header"])
        current_header = sample["header"]
        env_steps = 0

    # Setup tracking for specified theorems
    remaining_theorems = set(args.only) if args.only else None

    with open(output_path, "w") as out_file, open(errors_path, "w") as err_file:
        try:
            for sample in tqdm(dataset):
                total += 1

                if args.only:
                    if sample["name"] not in remaining_theorems:
                        continue

                # Check if we need to create new environment
                if (env is None or
                        sample["header"] != current_header or
                        env_steps >= args.max_env_steps):
                    restart_env()
                print(sample["name"])
                try:
                    theorem, error = process_theorem(
                        project, env,
                        sample["name"],
                        sample["formal_statement"],
                        sample["formal_proof"],
                        args.timeout,
                    )
                except LeanProcessException as e:
                    print(e)
                    restart_env()
                env_steps += 1

                # Handle result
                if theorem is not None:
                    successful += 1
                    write_result(theorem.serialize(), out_file)
                    print(f"OK")
                    if args.only:
                        print(theorem.by_blocks[0].tree.pretty_print())
                elif error.startswith("timeout"):
                    timed_out += 1
                    print(f"TIMEOUT: {error}")
                elif is_skipped_theorem(error):
                    skipped += 1
                    print(f"SKIPPED: {error}")
                else:
                    failed += 1
                    write_error(sample, error, err_file)
                    print(f"ERROR: {error}")

                # Print progress periodically
                if total % 10 == 0:
                    success_rate = (successful / total) * 100
                    skip_rate = (skipped / total) * 100
                    fail_rate = (failed / total) * 100
                    tout_rate = (timed_out / total) * 100
                    print(
                        f"Progress: {successful}/{total} successful ({success_rate:.2f}%), "
                        f"{skipped}/{total} skipped ({skip_rate:.2f}%), "
                        f"{failed}/{total} failed ({fail_rate:.2f}%), "
                        f"{timed_out}/{total} timed out ({tout_rate:.2f}%)"
                    )

                out_file.flush()
                err_file.flush()

                if args.only:
                    remaining_theorems.remove(sample["name"])
                    if not remaining_theorems:
                        print("All specified theorems have been processed.")
                        break

        finally:
            if env is not None:
                env.__exit__(None, None, None)

    # Print final statistics
    print(f"\n* Final results *")
    print(f"Total samples:  {total}")
    print(f"Successful:     {successful} ({(successful/total)*100:.2f}%)")
    print(f"Skipped:        {skipped} ({(skipped/total)*100:.2f}%)")
    print(f"Timed out:      {timed_out} ({(timed_out/total)*100:.2f}%)")
    print(f"Failed:         {failed} ({(failed/total)*100:.2f}%)")


if __name__ == "__main__":
    main()
