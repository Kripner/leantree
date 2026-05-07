## Installation

[![arXiv](https://img.shields.io/badge/cs.LG-2507.14722-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2507.14722)

Clone the repository including submodules:

```bash
git clone --recurse-submodules https://github.com/Kripner/leantree.git
```

Ensure [Lean](https://docs.lean-lang.org/lean4/doc/setup.html) is installed via elan.

```bash
lake --version
```

LeanTree currently pins Lean **v4.27.0** (see `lean-repl/lean-toolchain`); other 4.27.x patch versions may work but are untested.

Make sure `pip` is installed.
Then, run:

```bash
make install
```

Alternatively, use Poetry explicitly:

```bash
pip install poetry==1.8.4
poetry install
```

The `make install` step also runs `lake build` against the bundled `lean-repl/` submodule, producing the REPL executable at `lean-repl/.lake/build/bin/repl`.

For running tests or experiments, refer to the [Development](#development) section.

## Basic Usage

> **Note:** You must create a Lean project to use Lean.
> See [Lean project guide](https://leanprover-community.github.io/install/project.html) for more information.

Start by creating or loading a Lean project:

```python
from leantree import LeanProject

# Create a fresh project (Mathlib is *not* added unless you ask for it):
project = LeanProject.create("path/to/project", lean_version="v4.27.0", libraries=["mathlib"])

# Or load an existing project:
project = LeanProject("path/to/project")

# Or decide automatically:
project = LeanProject("path/to/project", create=True)
```

If no `path` is provided to `LeanProject.create` or the constructor, the project is created in or loaded from the `leantree_project` subdirectory of the current directory.

`libraries` accepts either short names (currently only `"mathlib"` is recognised) or `LeanLibrary` instances; pin a particular revision with `LeanLibrary(name=..., scope=..., git=..., rev=...)`. When `lean_version` is set and Mathlib is requested without an explicit revision, LeanTree pins Mathlib to the matching tag.

### Starting a Proof

Using the environment, you can initialize a proof search by supplying a theorem with one or more `sorry` keywords.

```python
with project.environment() as env:
    env.send_command("import Mathlib\nopen BigOperators Real Nat Topology Rat")
    branch = env.proof_from_sorry("theorem succ_less_double_succ (n : Nat) : n > 0 → n < 2 * n := by sorry")
    zero, succ = branch.apply_tactic("cases n")
    print("Factorized proof states after `cases n`:")
    print(zero.state)
    print(succ.state)
    assert not zero.is_solved
    ...
```

`apply_tactic` returns one `LeanProofBranch` per *independent* sub-goal (computed from the metavariable graph), which is what makes the resulting proof states factorized rather than threaded through a single linear goal stack. Use `branch.try_apply_tactic(...)` to get a `ValueOrError` instead of an exception on failure.

By default, search-style tactics (`apply?`, `rw?`, `exact?`, `grind?`) are rejected because they can close the goal in the REPL while leaving the proof unsound. Pass `ban_search_tactics=False` to opt out:

```python
branch.apply_tactic("apply?", ban_search_tactics=False)
```

### Async API

```python
async with project.environment() as env:
    await env.send_command_async("import Mathlib\nopen BigOperators Real Nat Topology Rat")
    branch = await env.proof_from_sorry_async("theorem succ_less_double_succ (n : Nat) : n > 0 → n < 2 * n := by sorry")
    zero, succ = await branch.apply_tactic_async("cases n")
    ...
```

Most environment and branch methods have an `_async` counterpart that can be awaited inside an async context manager.

### Data Extraction

Using the project, you can parse a Lean file and build all proof trees.
Then, you can use the environment to start a proof for each tactic block in the file.

```python
from leantree import StoredError

file = project.load_file("Example.lean")

# Pretty-print all proof trees.
for thm in file.theorems:
    if isinstance(thm, StoredError):
        print(f"Skipping: {thm.error}")
        continue
    print(thm.load_source() + "\n")
    for by_block in thm.by_blocks:
        if isinstance(by_block.tree, StoredError):
            print(f"  tree extraction failed: {by_block.tree}")
            continue
        print(by_block.tree.pretty_print())
    print("-" * 100)

# Start proofs for each tactic block in the file.
for thm, branches in env.file_proofs(file):
    if isinstance(branches, Exception):
        print(f"Could not start theorem '{thm}' due to exception: {branches}")
        continue
    for branch in branches:
        ...
```

Theorem and tactic-block extraction can both fail independently (file parse error vs. tree-builder error on a single `by`-block); both failure modes surface as `StoredError` rather than exceptions, so you can iterate large corpora without try/except wrappers. A worked example lives in [examples/parse_file.py](examples/parse_file.py).

### Dataset Generation

You can easily generate a dataset containing all Lean files in a directory.
For production-ready dataset generation, see the [Datasets section](#datasets)

```python
import glob
import json

with open("dataset.json", "w") as f:
    for path in glob.glob('**/*.lean', recursive=True):
        file = project.load_file(path)
        f.write(json.dumps(file.serialize()) + "\n")
```

### Save/Restore Environment State

The environment state can be saved to disk and restored later.

```python

with project.environment() as env:
    env.send_command("import Mathlib\nopen BigOperators Real Nat Topology Rat")
    env.pickle("env.pkl")

with project.environment() as env:
    env.unpickle("env.pkl")
    branch = env.proof_from_sorry("theorem succ_less_double_succ (n : Nat) : n > 0 → n < 2 * n := by sorry")
    zero, succ = branch.apply_tactic("cases n")
    ...
```

For finer-grained control inside a single environment, use `env.checkpoint()` and `env.rollback_to(checkpoint)` instead of pickling.

## Datasets

Assuming you have already created a Lean project in `leantree_project`, you can recreate the whole Mathlib dataset by running:

```bash
python dataset/tree_dataset.py generate \
  --project_path leantree_project \
  --source_files mathlib/Mathlib \
  --output_dir leantree_generated \
  --num_workers 16
```

`--num_workers` defaults to `1` (sequential); raising it spins up a work-stealing pool of that many worker processes and shows a live `tqdm` progress bar with running totals of extracted theorems, blocks, and failures. Per-file processing is bounded by `--file_timeout` (default `1800` seconds); files that exceed it are recorded in the `.errors` sidecar.

For reference, regenerating against current Mathlib (`v4.27.0`, May 2026) yields roughly:

| files processed | theorems extracted | tactic blocks with proof trees | total proof-tree nodes |
| --- | --- | --- | --- |
| 7,516 | 92,163 (95.2%) | 92,903 (90.4%) | 287,132 |

The remaining ~5% of theorems and ~10% of `by`-blocks land in the `.errors` sidecar, typically because of REPL/tree-builder edge cases on exotic tactics.

#### Sharded Generation (multi-node)

For multi-node setups (e.g. SLURM array jobs), each node processes a disjoint shard
via `--total_workers` / `--worker_id`. Each shard writes its own JSONL file
(`leantree-sf=<source>-<worker_id>.jsonl`) and `.errors` file. Combine `--num_workers`
within each shard to use multiple local processes.

```bash
python dataset/tree_dataset.py generate \
  --project_path leantree_project \
  --source_files mathlib/Mathlib \
  --output_dir leantree_generated \
  --total_workers $SLURM_ARRAY_TASK_COUNT \
  --worker_id $SLURM_ARRAY_TASK_ID \
  --num_workers 16
```

The last worker (`worker_id == total_workers - 1`) picks up any remainder files
when `len(files)` isn't divisible by `total_workers`, so no files are dropped.

#### Merging Shards

After all shards finish, combine them into a single JSONL file:

```bash
python dataset/tree_dataset.py merge_shards leantree_generated \
  --output_dir leantree_merged \
  --shards_count $SLURM_ARRAY_TASK_COUNT
```

This produces `leantree_merged/leantree-sf=<source>.jsonl` by concatenating all shard
files in `worker_id` order. It errors out if any shard file is missing, so you're
guaranteed a complete dataset on success. Use `--force` to overwrite an existing
output file.

Note: `.errors` files are not merged; concatenate them manually if needed
(`cat leantree_generated/*.errors > leantree_merged/all.errors`).

The CLI also exposes `view_trees`, `view_stats`, `error_stats`, `show_errors`, and `deepseek_convert` subcommands for inspecting and converting generated datasets; run with `--help` for flags.

## Lean Server (Process Pool)

For long-running services that need to amortize REPL startup across many proof attempts, LeanTree ships a standalone HTTP server that wraps a recycling process pool. It is installed as the `leanserver` console script:

```bash
leanserver \
  --address localhost --port 8000 \
  --max-processes 8 \
  --imports Mathlib \
  --warmup --warmup-batch-size 32 --warmup-timeout 600 \
  --rss-hard-limit-gib 32 --pss-recycle-limit-gib 4
```

Key options:

* `--repl-exe` / `LEAN_REPL_EXE`: path to the `repl` binary (defaults to the bundled `lean-repl/.lake/build/bin/repl`).
* `--project-path` / `LEAN_PROJECT_PATH`: Lean project to run inside (defaults to `./leantree_project`).
* `--imports`: packages to `import` during warmup (e.g. `Mathlib`); skipped without `--warmup`.
* `--warmup` / `--warmup-batch-size` / `--warmup-timeout`: pre-start every worker with the given imports applied, in batches, before accepting requests.
* `--rss-hard-limit-gib`: `RLIMIT_AS` ceiling per subprocess (default `32`); `<= 0` disables.
* `--pss-recycle-limit-gib`: recycle a worker on return whose PSS exceeds this (default `4`); `<= 0` disables.

Once running, `kill -USR1 <pid>` dumps Python tracebacks for every thread, `kill -USR2 <pid>` (or pressing Enter in the foreground terminal) prints pool status.

## Development

Install all development and experiments dependencies:

```bash
make install-dev
```

Other Make targets: `make build` (poetry build), `make clean` (remove build artifacts).

### Running Tests

To run tests, first create a Lean v4.27.0 project called `leantree_project` in the LeanTree directory.
You can use LeanTree for that - in the leantree directory, run the following Python code:

```python
from leantree import LeanProject

project = LeanProject.create(lean_version="v4.27.0", libraries=["mathlib"])
```

After the project is created, run:

```bash
make test
```

The test suite covers the REPL adapter, process pool invariants, async timeout handling, file-span utilities, and the Lean server, in addition to the data-extraction pipeline.

### Debugging Tips

When working with a Lean environment, you can use `env.take_control()` to debug the underlying Lean REPL.
This method connects your stdin/stdout to the REPL's stdin/stdout.

For benchmarking, [scripts/bench_warmup.py](scripts/bench_warmup.py) measures end-to-end pool warmup time and [scripts/stress_test_server.py](scripts/stress_test_server.py) exercises the `leanserver` under load.

---

## Related Tools

* **[LeanDojo](https://github.com/lean-dojo)**
* **[Pantograph](https://github.com/stanford-centaur/PyPantograph)**

> For a detailed comparison, refer to the LeanTree paper.

## Reference

```
@inproceedings{
    kripner2025leantree,
    title={LeanTree: Accelerating White-Box Proof Search with Factorized States in Lean 4},
    author={Mat{\v{e}}j Kripner and Michal Sustr and Milan Straka},
    booktitle={2nd AI for Math Workshop @ ICML 2025},
    year={2025},
    url={https://openreview.net/forum?id=pROVJwTb6w}
}
```
