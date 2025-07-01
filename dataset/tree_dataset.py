import argparse
import itertools
import json
import sys
from collections import defaultdict
import traceback
from pathlib import Path
import os
from typing import Iterable, Optional

from tqdm import tqdm

from leantree import utils
from leantree.core.lean_file import LeanFile, StoredError
from leantree.core.project import LeanProject, LeanTheorem


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
    deepseek_convert = subparsers.add_parser("deepseek_convert")
    error_stats = subparsers.add_parser("error_stats")
    show_errors = subparsers.add_parser("show_errors")

    generate.add_argument("--project_path", type=Path, required=True)
    generate.add_argument("--source_files", type=Path, default="mathlib/Mathlib")
    generate.add_argument("--output_dir", type=Path, default="leantree_generated")
    generate.add_argument("--repl_path", type=Path, default="lean-repl/.lake/build/bin/repl")
    generate.add_argument("--skip_until", type=Path)
    generate.add_argument("--force", action="store_true", help="Override the output file if it already exists.")
    generate.add_argument("--use_repl_cache", action="store_true")

    generate.add_argument("--total_workers", type=int)
    generate.add_argument("--worker_id", type=int)

    view_trees.add_argument("dataset_path", type=Path)
    view_trees.add_argument("--limit", type=int, default=100)

    view_stats.add_argument("dataset_path", type=Path)
    view_stats.add_argument("--project_path", type=Path, required=True)
    view_stats.add_argument("--source_files", type=Path, default="mathlib/Mathlib")

    merge_shards.add_argument("shard_directory", type=Path)
    merge_shards.add_argument("--output_dir", type=Path, required=True)
    merge_shards.add_argument("--shards_count", type=int, default=128)
    merge_shards.add_argument("--force", action="store_true", help="Override the output file if it already exists.")

    deepseek_convert.add_argument("input_file", type=Path)
    deepseek_convert.add_argument("output_file", type=Path)
    deepseek_convert.add_argument("--force", action="store_true", help="Override the output file if it already exists.")

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
    if args.worker_id is None:
        args.worker_id = 0
        args.total_workers = 1

    out_file = args.output_dir / f"lean-trees-{descriptor}-{args.worker_id}.jsonl"
    if not args.force:
        if out_file.exists():
            print(f"Exiting because output file already exists: {out_file}")
            return

    errors_file = Path(str(out_file) + ".errors")
    if errors_file.is_file():
        errors_file.unlink()

    source_files_path = args.project_path / ".lake/packages" / args.source_files
    print(f"Identifying Lean files: {source_files_path}")
    paths = identify_lean_files(args, source_files_path)
    paths = list(paths)
    batch_size = len(paths) // args.total_workers
    offset = args.worker_id * batch_size
    paths = itertools.islice(paths, offset, offset + batch_size)
    paths = list(paths)
    print(
        f"Will process {len(paths)} paths (worker {args.worker_id} of {args.total_workers}, files {offset} - {offset + batch_size})."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {out_file}")

    generator = DatasetGenerator(
        args,
        out_file,
        errors_file,
    )
    generator.generate(paths)


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
        for i, path in enumerate(paths):
            print(f"Processing [{i + 1}/{len(paths)}]: {path}")
            total_files += 1
            # noinspection PyBroadException
            try:
                loaded_file = self.load_file(path.absolute())
                self.store_file(loaded_file)
            except Exception:
                failed_files += 1
                print(f"Failed file: {path}", file=sys.stderr)
                traceback.print_exc()

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

    output_file = output_dir / "lean-trees-sf=mathlib_Mathlib.jsonl"
    if output_file.exists() and not args.force:
        raise Exception(f"Output file {output_file} already exists")

    print(f"Merging shards from {shard_dir} into {output_file}")

    with open(output_file, 'w') as out_f:
        for i in tqdm(range(args.shards_count)):
            shard_file = shard_dir / f"lean-trees-sf=mathlib_Mathlib-{i}.jsonl"
            if not shard_file.exists():
                raise Exception(f"Shard file {shard_file} does not exist")

            with open(shard_file, 'r') as in_f:
                for line in in_f:
                    out_f.write(line)

    print(f"Successfully merged {args.shards_count} shards into {output_file}")


def deepseek_prover_convert(args):
    if args.output_file.exists() and not args.force:
        raise Exception(f"Output file {args.output_file} already exists")

    with (open(args.input_file) as f, open(args.output_file, 'w') as out_f):
        for line in tqdm(f):
            theorem = LeanTheorem.deserialize(json.loads(line))
            file = LeanFile(
                path=None,
                imports=["Mathlib", "Aesop"],
                theorems=[theorem],
            )
            theorem.file = file
            theorem.context = [
                "open BigOperators Real Nat Topology Rat"
            ]

            out_f.write(json.dumps(file.serialize(), ensure_ascii=False) + "\n")


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
    elif args.action == "deepseek_convert":
        deepseek_prover_convert(args)
    elif args.action == "error_stats":
        error_stats(args)
    elif args.action == "show_errors":
        show_errors(args)
    else:
        raise Exception(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main(create_parser().parse_args())
