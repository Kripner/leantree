import argparse
import itertools
import json
import multiprocessing as mp
import signal
import sys
from collections import defaultdict
import traceback
from pathlib import Path
import os
from typing import Iterable, Optional

from tqdm import tqdm

from leantree import utils
from leantree.core.lean_file import LeanFile, StoredError
from leantree.core.project import LeanProject


# TODO:
#  Think about what should be in:
#  - supervised training examples --> just tactic-style proofs (NOTE: these can also occur in e.g. `def`! So this is not a subset of evaluation data!)
#  - online training examples  --> all theorem statements
#  - evaluation examples  --> all theorem statements (but provide the info whether tactic-style proof exists)
#  Possible solution:
#  - for each theorem, give also its "statement" (the initial goal)
#  - include even theorems with len(proof_tree_edges) == 0  (which thus have a purely-term-style proof)
#  - give whether the theorem has purely-tactic-style proof
#    - maybe by looking at whether the statement of the theorem and the state_before of the root are the same?
#    - or syntactically (looking for ':= by' after the theorem statement)

# TODO Problems:
#  - `theorem coeff` in Algebra/CubicDiscriminant.lean
#    - `omega` closes all goals, but then another goal is present when doing `repeat' rw [zero_add]`

def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    generate = subparsers.add_parser("generate")
    view_trees = subparsers.add_parser("view_trees")
    view_stats = subparsers.add_parser("view_stats")
    merge_shards = subparsers.add_parser("merge_shards")
    error_stats = subparsers.add_parser("error_stats")
    show_errors = subparsers.add_parser("show_errors")

    generate.add_argument("--project_path", type=Path, required=True)
    generate.add_argument("--source_files", type=Path, default="mathlib/Mathlib")
    generate.add_argument("--output_dir", type=Path, default="leantree_generated")
    generate.add_argument("--repl_path", type=Path, default="lean-repl/.lake/build/bin/repl")
    generate.add_argument("--skip_until", type=Path)
    generate.add_argument("--force", action="store_true", help="Override the output file if it already exists.")
    generate.add_argument("--use_repl_cache", action="store_true")

    generate.add_argument("--num_workers", type=int, default=1,
                          help="Number of parallel worker processes (work-stealing).")
    generate.add_argument("--file_timeout", type=int, default=1800,
                          help="Per-file processing timeout in seconds (default 1800 = 30 min). "
                               "Files exceeding this limit are recorded as errors.")
    # For SLURM/multi-node: static sharding
    generate.add_argument("--total_workers", type=int, help="[Advanced] Total shards across nodes.")
    generate.add_argument("--worker_id", type=int, help="[Advanced] Shard ID for this invocation.")

    view_trees.add_argument("dataset_path", type=Path)
    view_trees.add_argument("--limit", type=int, default=100)

    view_stats.add_argument("dataset_path", type=Path)
    view_stats.add_argument("--project_path", type=Path, required=True)
    view_stats.add_argument("--source_files", type=Path, default="mathlib/Mathlib")

    merge_shards.add_argument("shard_directory", type=Path)
    merge_shards.add_argument("--output_dir", type=Path, required=True)
    merge_shards.add_argument("--shards_count", type=int, default=128)
    merge_shards.add_argument("--force", action="store_true", help="Override the output file if it already exists.")

    error_stats.add_argument("error_files", type=Path, nargs="+")
    error_stats.add_argument("--output_dir", type=Path, required=True)

    show_errors.add_argument("dataset_path", type=Path)
    show_errors.add_argument("--kind", type=str)

    return parser


def identify_lean_files(args: argparse.Namespace, source_files_path: Path) -> Iterable[Path]:
    if source_files_path.is_file():
        assert str(source_files_path).endswith(".lean")
        yield source_files_path
        return
    assert source_files_path.is_dir()
    skipping = hasattr(args, "skip_until") and args.skip_until is not None
    for root, dirs, files in os.walk(source_files_path):
        for file_name in sorted(files):
            if file_name.endswith(".lean"):
                absolute_file_name = Path(root) / file_name
                if skipping:
                    if str(args.skip_until) == str(absolute_file_name):
                        skipping = False
                    else:
                        print(f"Skipping: {absolute_file_name}")
                        print(args.skip_until, absolute_file_name)
                        continue
                # TODO: is this true?
                # REPL process is created anew for each file since there are massive memory leaks, making reuse impossible.
                yield absolute_file_name


def generate_dataset(args: argparse.Namespace):
    metadata = {k: str(getattr(args, k)) for k in ["source_files"] if getattr(args, k) is not None}
    descriptor = utils.get_args_descriptor(
        args,
        param_whitelist=set(metadata.keys()),
        include_time=False,
        include_slurm_id=False,
    )

    # Shard suffix for multi-node setups; default single shard.
    shard_id = args.worker_id if args.worker_id is not None else 0
    total_shards = args.total_workers if args.total_workers is not None else 1

    out_file = args.output_dir / f"leantree-{descriptor}-{shard_id}.jsonl"
    if not args.force:
        if out_file.exists():
            print(f"Exiting because output file already exists: {out_file}")
            return

    errors_file = Path(str(out_file) + ".errors")
    if errors_file.is_file():
        errors_file.unlink()

    source_files_path = args.project_path / ".lake/packages" / args.source_files
    print(f"Identifying Lean files: {source_files_path}")
    paths = list(identify_lean_files(args, source_files_path))

    if total_shards > 1:
        batch_size = len(paths) // total_shards
        offset = shard_id * batch_size
        end = offset + batch_size if shard_id < total_shards - 1 else len(paths)
        paths = paths[offset:end]
        print(f"Shard {shard_id}/{total_shards}: processing {len(paths)} files (offset {offset})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_file}")
    print(f"Workers: {args.num_workers}")

    if args.num_workers <= 1:
        # Sequential mode (legacy)
        generator = DatasetGenerator(args, out_file, errors_file)
        generator.generate(paths)
    else:
        _generate_parallel(args, paths, out_file, errors_file)


# ----- Parallel worker pool -----

# Per-worker cached project (initialized lazily).
_WORKER_PROJECT = None


def _worker_init(project_path: str, repl_path: str):
    """Runs once per worker process to initialize the LeanProject."""
    global _WORKER_PROJECT
    logger = utils.Logger(utils.LogLevel.SUPPRESS)
    _WORKER_PROJECT = LeanProject(Path(project_path), repl_path=Path(repl_path), logger=logger)


class _FileTimeout(Exception):
    pass


def _relative_to_packages(project_path: Path, file_path: Path) -> Path:
    packages_dir = project_path / ".lake" / "packages"
    if file_path.is_relative_to(packages_dir):
        return file_path.relative_to(packages_dir)
    return file_path


def _crashed_file_entry(project_path: Path, file_path: Path, error_msg: str) -> str:
    """Build a JSONL entry for a file that crashed during processing."""
    return json.dumps({
        "path": str(_relative_to_packages(project_path, file_path)),
        "imports": [],
        "theorems": [],
        "error": error_msg,
    }, ensure_ascii=False)


def _worker_process_file(task):
    """Process a single file in a worker. Returns (path, main_line, err_lines, exc, stats)."""
    path_str, use_cache, timeout_sec = task
    path = Path(path_str)

    def _alarm(signum, frame):
        raise _FileTimeout(f"Per-file timeout of {timeout_sec}s exceeded")

    if timeout_sec and timeout_sec > 0:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout_sec)
    try:
        loaded = _WORKER_PROJECT.load_file(path, use_cache=use_cache)
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        return path_str, None, [], traceback.format_exc(), None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)

    main_line = json.dumps(loaded.serialize(), ensure_ascii=False)
    err_lines = []
    total_thms = len(loaded.theorems)
    good_thms = 0
    good_blocks = 0
    for thm in loaded.theorems:
        if isinstance(thm, StoredError):
            err_lines.append(json.dumps(thm.serialize(), ensure_ascii=False))
        else:
            good_thms += 1
            for by_block in thm.by_blocks:
                if isinstance(by_block.tree, StoredError):
                    err_lines.append(json.dumps(by_block.tree.serialize(), ensure_ascii=False))
                else:
                    good_blocks += 1
    return path_str, main_line, err_lines, None, (total_thms, good_thms, good_blocks)


def _generate_parallel(args, paths: list[Path], out_file: Path, errors_file: Path):
    tasks = [(str(p.absolute()), args.use_repl_cache, args.file_timeout) for p in paths]

    total_thms = 0
    good_thms = 0
    good_blocks = 0
    failed_files = 0

    # spawn is safer than fork for this complex codebase
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        args.num_workers,
        initializer=_worker_init,
        initargs=(str(args.project_path), str(args.repl_path)),
    ) as pool, open(out_file, "w") as fo, open(errors_file, "w") as fe:
        pbar = tqdm(total=len(tasks), desc=f"[{args.num_workers} workers]", unit="file")
        for result in pool.imap_unordered(_worker_process_file, tasks):
            path_str, main_line, err_lines, exc, *rest = result
            if main_line is None:
                failed_files += 1
                tqdm.write(f"FAILED: {path_str}\n{exc}", file=sys.stderr)
                # Still record the file with an error so it appears in the dataset.
                crashed_line = _crashed_file_entry(args.project_path, Path(path_str), exc or "Unknown error")
                fo.write(crashed_line + "\n")
                fo.flush()
            else:
                fo.write(main_line + "\n")
                fo.flush()
                for el in err_lines:
                    fe.write(el + "\n")
                fe.flush()
                t, gt, gb = rest[0]
                total_thms += t
                good_thms += gt
                good_blocks += gb
            pbar.set_postfix(
                thms=f"{good_thms}/{total_thms}",
                blocks=good_blocks,
                failed=failed_files,
            )
            pbar.update(1)
        pbar.close()

    print(
        f"\nDone. {len(tasks)} files, {failed_files} failed. "
        f"Theorems: {good_thms}/{total_thms}. Blocks: {good_blocks}."
    )


class DatasetGenerator:
    def __init__(self, args, out_path: Path, errors_path: Path):
        self.args = args
        self.out_path = out_path
        self.errors_path = errors_path

        self.total_theorems = 0
        self.good_theorems = 0
        self.good_blocks = 0

        # self.logger = utils.Logger(utils.LogLevel.SUPPRESS)
        # self.logger = utils.Logger(utils.LogLevel.DEBUG)
        self.logger = utils.Logger(utils.LogLevel.INFO)
        self.project = LeanProject(args.project_path, repl_path=args.repl_path, logger=self.logger)

    def generate(self, paths: list[Path]):
        total_files = 0
        failed_files = 0
        timeout = getattr(self.args, "file_timeout", 0)
        for i, path in enumerate(paths):
            print(f"Processing [{i + 1}/{len(paths)}]: {path}")
            total_files += 1

            def _alarm(_signum, _frame):
                raise _FileTimeout(f"Per-file timeout of {timeout}s exceeded")

            if timeout and timeout > 0:
                signal.signal(signal.SIGALRM, _alarm)
                signal.alarm(timeout)
            # noinspection PyBroadException
            try:
                loaded_file = self.load_file(path.absolute())
                self.store_file(loaded_file)
            except Exception:
                failed_files += 1
                print(f"Failed file: {path}", file=sys.stderr)
                err_msg = traceback.format_exc()
                traceback.print_exc()
                with open(self.out_path, "a") as f:
                    f.write(_crashed_file_entry(self.args.project_path, path.absolute(), err_msg) + "\n")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)

            print(self.get_stats())
            print(f"Failed files: {failed_files} / {total_files} ({failed_files / total_files:%})")

    def load_file(self, path: Path):
        assert path.is_file()
        return self.project.load_file(
            path,
            use_cache=self.args.use_repl_cache,
        )

    def store_file(self, loaded_file: LeanFile):
        self.total_theorems += len(loaded_file.theorems)

        with open(self.out_path, "a") as f:
            f.write(json.dumps(loaded_file.serialize(), ensure_ascii=False) + "\n")
        with open(self.errors_path, "a") as f:
            for thm in loaded_file.theorems:
                if isinstance(thm, StoredError):
                    # TODO: provide better info
                    f.write(json.dumps(thm.serialize(), ensure_ascii=False) + "\n")
                else:
                    self.good_theorems += 1
                    for by_block in thm.by_blocks:
                        if isinstance(by_block.tree, StoredError):
                            f.write(json.dumps(by_block.tree.serialize()) + "\n")
                        else:
                            self.good_blocks += 1
                            print(by_block.tree.pretty_print())

    def get_stats(self):
        if self.total_theorems == 0:
            return "Total theorems: 0 (cannot compute stats)"
        return (
            f"Total theorems: {self.total_theorems}"
            f"  |  good theorems: {self.good_theorems} ({self.good_theorems / self.total_theorems:%})"
            f"  |  failed theorems: {self.total_theorems - self.good_theorems} ({(self.total_theorems - self.good_theorems) / self.total_theorems:%})"
            f"  |  good blocks: {self.good_blocks}"
        )


def view_trees(args):
    with open(args.dataset_path) as f:
        for line in itertools.islice(f, 0, args.limit + 1):
            file = LeanFile.deserialize(json.loads(line))
            for thm in file.theorems:
                print(thm.get_source_without_tactic_blocks())
                print()
                for by_block in thm.by_blocks:
                    tree = by_block.tree
                    print(tree.pretty_print())
                print("\n---\n")


def view_stats(args):
    print("====== DATASET STATS ======")
    print(args.dataset_path)
    print()

    source_files_path = args.project_path / ".lake/packages" / args.source_files
    all_files = len(list(identify_lean_files(args, source_files_path)))

    print(f"Total files:\t\t{all_files}")

    all_theorems, all_proofs = 0, 0
    gen_files, gen_theorems, gen_proofs = 0, 0, 0
    conv, calc = 0, 0
    with open(args.dataset_path) as f:
        for line in tqdm(f):
            gen_files += 1
            file = LeanFile.deserialize(json.loads(line))
            for thm in file.theorems:
                all_theorems += 1
                if isinstance(thm, StoredError):
                    continue
                gen_theorems += 1
                for proof in thm.by_blocks:
                    all_proofs += 1
                    if isinstance(proof.tree, StoredError):
                        if get_error_category(proof.tree.error) == "conv":
                            conv += 1
                        elif get_error_category(proof.tree.error) == "calc":
                            calc += 1
                        continue
                    gen_proofs += 1
    print(f"Good files:\t{gen_files}/{all_files} ({gen_files / all_files:%})")
    print(f"Good theorems:\t{gen_theorems}/{all_theorems} ({gen_theorems / all_theorems:%})")
    print(f"Good proofs:\t{gen_proofs}/{all_proofs} ({gen_proofs / all_proofs:%})")
    bad_proofs = all_proofs - gen_proofs
    print(f"Calc-caused bad proofs:\t{calc}/{bad_proofs} ({calc / bad_proofs:%})")
    print(f"Conv-caused bad proofs:\t{conv}/{bad_proofs} ({conv / bad_proofs:%})")


def merge_shards(args):
    shard_dir = args.shard_directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "leantree-sf=mathlib_Mathlib.jsonl"
    if output_file.exists() and not args.force:
        raise Exception(f"Output file {output_file} already exists")

    print(f"Merging shards from {shard_dir} into {output_file}")

    with open(output_file, 'w') as out_f:
        for i in tqdm(range(args.shards_count)):
            shard_file = shard_dir / f"leantree-sf=mathlib_Mathlib-{i}.jsonl"
            if not shard_file.exists():
                raise Exception(f"Shard file {shard_file} does not exist")

            with open(shard_file, 'r') as in_f:
                for line in in_f:
                    out_f.write(line)

    print(f"Successfully merged {args.shards_count} shards into {output_file}")


def get_error_category(error: str) -> str:
    error = error.strip()
    if len(error) == 0:
        return "empty"
    if error == "`simp_rw` tactic is not yet supported.":
        return "simp_rw"
    if error == "`conv` tactic is not yet supported":
        return "conv"
    if error == "`calc` tactic is not yet supported":
        return "calc"
    if error == "`sorry` not allowed in `simpa`":
        return "sorry-in-simpa"
    if error == "Not enough singleton nodes to use in the proof.":
        return "not-enough-singleton-nodes"
    if error.startswith("REPL returned error: Lean error:"):
        if "unknown tactic" in error:
            return "unknown-tactic"
        if "unexpected end of input" in error:
            return "unexpected-end-of-input"
        if "failed to synthesize" in error:
            return "failed-to-synthesize"
        if "type mismatch" in error:
            return "type-mismatch"
        if "no goals to be solved" in error:
            return "no-goals-to-solve"
        if "internal exception #5" in error:
            return "internal-exception-5"
        if "expected tactic" in error:
            return "expected-tactic"
        if "unknown constant" in error:
            return "unknown-constant"
        if "maximum number of heartbeats (200000) has been reached" in error:
            return "maximum-heartbeats-reached"
    if error.startswith("REPL returned error messages:"):
        if "no goals to be solved" in error:
            return "no-goals-to-solve"
        if "unknown universe level" in error:
            return "unknown-universe-level"
        if "failed to synthesize" in error:
            return "failed-to-synthesize"
    if error.startswith("Step verification error"):
        return "step-verification"

    return "unknown"


def error_stats(args):
    counts = defaultdict(int)
    errors = defaultdict(list)
    for error_file in args.error_files:
        with open(error_file) as f:
            for line in f:
                error = json.loads(line)["error"]
                category = get_error_category(error)
                counts[category] += 1
                errors[category].append(error)

    total_errors = sum(counts.values())
    print(f"Total errors: {total_errors}")
    for category, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count} ({count / total_errors:%})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "error_stats.json", "w") as f:
        json.dump(counts, f)
    (args.output_dir / "categories").mkdir(parents=True, exist_ok=True)
    for category, errors in errors.items():
        with open(args.output_dir / "categories" / f"{category}.jsonl", "w") as f:
            for error in errors:
                f.write(json.dumps({"error": error}, ensure_ascii=False) + "\n")


def show_errors(args):
    def should_show(err: StoredError) -> bool:
        if args.kind is None:
            return True
        return get_error_category(err.error) == args.kind

    with open(args.dataset_path) as f:
        for line in f:
            data = json.loads(line)
            file = LeanFile.deserialize(data)
            for thm in file.theorems:
                if isinstance(thm, StoredError):
                    if should_show(thm):
                        print(f"THM error: {thm.error}")
                        print("----")
                    continue
                errors = [by_block.tree for by_block in thm.by_blocks if
                          isinstance(by_block.tree, StoredError) and should_show(by_block.tree)]
                if errors:
                    print(thm.load_source())
                    for error in errors:
                        print()
                        print(f"error: '{error.error}'")
                    print("----")


def main(args):
    # utils.resolve_paths(args)

    if args.action == "generate":
        generate_dataset(args)
    elif args.action == "view_trees":
        view_trees(args)
    elif args.action == "view_stats":
        view_stats(args)
    elif args.action == "merge_shards":
        merge_shards(args)
    elif args.action == "error_stats":
        error_stats(args)
    elif args.action == "show_errors":
        show_errors(args)
    else:
        raise Exception(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main(create_parser().parse_args())
