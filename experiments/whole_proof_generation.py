import argparse
import json
import time
from pathlib import Path
from enum import Enum
import asyncio
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

from leantree import utils
from leantree.repl_adapter.interaction import LeanServer, LeanInteractionException, LeanProcessException
from experiments.interlm_adapter import InterLMMiniF2FAdapter

MINIF2F_HEADER = (
    "import Mathlib\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology\n"
)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="EleutherAI/llemma_7b")
    parser.add_argument("--per_device_batch_size", type=int, default=8)

    parser.add_argument("--max_passes", type=int, default=10)

    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--benchmark_cache_dir", type=Path, default="benchmarks")
    parser.add_argument("--output_dir", type=Path, default="whole_proof_generation-lemma")

    parser.add_argument("--repl_exe", type=Path, required=True)
    parser.add_argument("--project_path", type=Path, required=True)

    parser.add_argument("--force", action="store_true")

    return parser


ARGS_WHITELIST = [
    "seed",
    "checkpoint",
    "max_passes",
    "max_output_tokens",
    "temperature",
]

class Logger:
    def __init__(self, log_dir: Path):
        self._model_outputs_path = log_dir / "model_outputs.txt"
        self._exceptions_path = log_dir / "exceptions.txt"
        self._stats_path = log_dir / "stats.json"
        self._final_proofs_path = log_dir / "proofs.lean"
        self._incomplete_proofs_path = log_dir / "incomplete_proofs.lean"
        self._error_proofs_path = log_dir / "error_proofs.lean"
        self._unknown_outputs_path = log_dir / "unknown_outputs.lean"
        self._start_time = time.time()

    def log_model_outputs(self, outputs: list[str]):
        with open(self._model_outputs_path, "a") as f:
            for output in outputs:
                quotes = '"""\n'
                f.write(f"{quotes}{output}\n{quotes}\n\n")

    def log_exception(self, theorem: str, exception: Exception, note: str):
        error_traceback = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        print(f"Unhandled exception ({note}): {exception}\n{error_traceback}")
        with open(self._exceptions_path, "a") as f:
            f.write(f"{theorem}\n")
            f.write(f"Exception: {exception}\n")
            f.write(f"Note: {note}\n")
            f.write(f"Traceback:\n{error_traceback}\n\n")

    def log_stats(self, completed_searches: "list[WholeProofSearch]"):
        stats = self._calculate_stats(completed_searches)
        for k, v in stats.items():
            print(f"{k}: {v}")
        self._stats_path.write_text(json.dumps(stats, indent=4))

    def log_final_proof(self, unit: str):
        if not self._final_proofs_path.exists():
            with open(self._final_proofs_path, "w") as f:
                f.write(MINIF2F_HEADER + "\n")
                f.write(
                    "-- Successful MiniF2F proofs found by whole proof generation\n\n"
                )

        with open(self._final_proofs_path, "a") as f:
            f.write(unit + "\n\n")

    def log_incomplete_proof(self, unit: str):
        if not self._incomplete_proofs_path.exists():
            with open(self._incomplete_proofs_path, "w") as f:
                f.write(MINIF2F_HEADER + "\n")
                f.write(
                    "-- Incomplete MiniF2F proofs found by whole proof generation\n\n"
                )

        with open(self._incomplete_proofs_path, "a") as f:
            f.write(unit + "\n\n")

    def log_error_proof(self, unit: str, error: LeanInteractionException | str | None):
        if not self._error_proofs_path.exists():
            with open(self._error_proofs_path, "w") as f:
                f.write(MINIF2F_HEADER + "\n")
                f.write(
                    "-- Failed MiniF2F proofs found by whole proof generation\n\n"
                )

        with open(self._error_proofs_path, "a") as f:
            f.write(unit + "\n")
            f.write(f"-- Error: {error}\n\n\n")

    def log_unknown_output(self, theorem: str, output: str):
        with open(self._unknown_outputs_path, "a") as f:
            quotes = '"""'
            f.write(theorem + "\n")
            f.write(f"{quotes}{output}{quotes}\n\n")

    def _calculate_stats(self, completed_searches: "list[WholeProofSearch]") -> dict:
        runtime = time.time() - self._start_time
        return {
            "completed": len(completed_searches),

            "proven": len([r for r in completed_searches if r.proven]),
            "proven_rate": len([r for r in completed_searches if r.proven]) / len(completed_searches) if completed_searches else 0,

            "runtime": runtime,
        }

prompt = """
Complete the given Lean 4 code.
Directly output the completed Lean 4 code without any additional text, comments, or reasoning.
""".strip()

class BanTokenLogitsProcessor(LogitsProcessor):
    """Processor that bans specific token IDs from being generated."""
    
    def __init__(self, banned_token_ids):
        self.banned_token_ids = set(banned_token_ids)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Set the logits for banned tokens to negative infinity
        for token_id in self.banned_token_ids:
            scores[:, token_id] = float('-inf')
        return scores

class Model:
    def __init__(
            self,
            args: argparse.Namespace,
            logger: Logger,
    ):
        self.args = args
        self.logger = logger
        self.model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ban tokens: "sorry", "admit", "--", "/-".
        self.logits_processors = LogitsProcessorList(
            [BanTokenLogitsProcessor([
                7423,  # ▁sorry
                20000,  # ▁admit
                489,  # --
                1192,  # ▁--
                24028,  # /-
            ])]
        )

    def generate(self, statement: str) -> list[str]:
        text = f"{prompt}\n\n```lean4\n{statement[:-len('sorry')]}\n  "
        encoded = self.tokenizer(
            [text], return_tensors="pt", padding=True, truncation=False
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        print("Generating...")
        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=self.args.max_output_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                stop_strings=["```"],
                tokenizer=self.tokenizer,
                num_return_sequences=self.args.max_passes,
                temperature=self.args.temperature,
                logits_processor=self.logits_processors,  # Add the logits processor
            )
        # Extract only the new tokens (exclude prompt tokens).
        output_tokens = outputs[:, encoded["input_ids"].shape[1]:].tolist()
        output_decoded = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        # Log the whole output including prompt and thinking.
        all_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.logger.log_model_outputs(all_decoded)

        del outputs
        del output_tokens
        torch.cuda.empty_cache()

        return ["  " + proof.rstrip("```") for proof in output_decoded]

class ProofResult(Enum):
    Incomplete = "incomplete"
    Completed = "completed"
    Error = "error"

class Verifier:
    def __init__(self, args: argparse.Namespace, logger: Logger):
        self.args = args
        self.logger = logger

        self.repl: LeanServer | None = None

    async def verify(self, theorem: str, proof: str) -> tuple[ProofResult, str, LeanInteractionException | str | None]:
        restarts = 0
        unit = self._create_verifiable_unit(theorem, proof)
        while True:
            if self.repl is None:
                if restarts >= 3:
                    return ProofResult.Error, unit, "Too many REPL restarts"
                restarts += 1
                await self._restart_repl()
            assert self.repl is not None
            # print(f"Verifying:\n'{unit}'")
            if "sorry" in proof or "admit" in proof:
                return ProofResult.Incomplete, unit, None
            try:
                response = await self.repl.send_command_async(unit)
                if any(m["data"] == "Goals accomplished!" for m in response["messages"]):
                    return ProofResult.Completed, unit, None
                else:
                    return ProofResult.Incomplete, unit, None
            except LeanInteractionException as e:
                return ProofResult.Error, unit, e
            except LeanProcessException as e:
                self.logger.log_exception(theorem, e, "Lean process exception")
                print(f"Lean process exception: {e}")
                print("Restarting REPL...")
                self.repl = None  # Will be restarted at the start of next iteration.

    def _create_verifiable_unit(self, theorem: str, proof: str) -> str:
        return (
            theorem[:-len("sorry")].strip() + "\n" +
            proof.rstrip()
        )

    async def _restart_repl(self):
        if self.repl:
            await self.repl.stop_async_safe()
            self.repl = None
        self.repl = LeanServer(
            self.args.repl_exe,
            self.args.project_path,
            # utils.Logger(utils.LogLevel.DEBUG),
        )
        await self.repl.start_async()
        await self.repl.send_command_async(MINIF2F_HEADER)

class WholeProofSearch:
    def __init__(self, args: argparse.Namespace, theorem: str, model: Model, verifier: Verifier, logger: Logger):
        self.args = args
        self.theorem = theorem
        self.model = model
        self.verifier = verifier
        self.logger = logger

        self.proven: bool | None = None

    async def try_prove(self):
        output_decoded = self.model.generate(self.theorem)

        for output in output_decoded:
            # proof = self._extract_proof(output)
            proof = output
            # if proof is None:
            #     self.logger.log_unknown_output(self.theorem, output)
            #     continue
            result, unit, error = await self.verifier.verify(self.theorem, proof)
            if result == ProofResult.Completed:
                self.proven = True
                self.logger.log_final_proof(unit)
                return
            elif result == ProofResult.Incomplete:
                self.logger.log_incomplete_proof(unit)
            else:
                assert result == ProofResult.Error
                self.logger.log_error_proof(unit, error)
        self.proven = False

    # def _extract_proof(self, output: str) -> str | None:
    #     match = re.search(r"```lean4?(.*?)```", output, re.DOTALL)
    #     if not match:
    #         return None
    #     proof = match.group(1)
    #     if ":= by" not in proof:
    #         return None
    #     proof = proof[proof.index(":= by") + len(":= by"):]
    #     proof = "".join(line for line in proof.splitlines(keepends=True) if line.strip())
    #     return proof

async def run_searches(args: argparse.Namespace, theorems: list[str], model: Model, logger: Logger) -> list[WholeProofSearch]:
    verifier = Verifier(args, logger)
    searches = []
    for theorem in tqdm(theorems):
        search = WholeProofSearch(args, theorem, model, verifier, logger)
        searches.append(search)
        await search.try_prove()
        logger.log_stats(searches)
    return searches

def mask_theorem_names(theorems: list[str]) -> list[str]:
    modified_theorems = []
    for theorem in theorems:
        words = theorem.split()
        assert len(words) > 2
        modified_theorems.append(
            "example" + theorem[len(words[0] + " " + words[1]):]
        )
    return modified_theorems

def main():
    args = get_parser().parse_args()
    utils.resolve_paths(args)
    utils.setup_seeds(args.seed)

    descriptor = utils.get_args_descriptor(args, param_whitelist=set(ARGS_WHITELIST))
    log_dir = args.output_dir / descriptor
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_dir}")
    logger = Logger(log_dir)

    utils.dump_args(args, log_dir)

    adapter = InterLMMiniF2FAdapter(args.benchmark_cache_dir)
    benchmark = adapter.fetch_minif2f()

    print(f"Loading models from {args.checkpoint}...")
    model = Model(args, logger)

    theorems = benchmark.test_theorems
    print(f"Will try to prove {len(theorems)} theorems.")

    # Avoid repeated definition + self-references.
    theorems = mask_theorem_names(theorems)

    start_time = time.time()
    completed_searches = asyncio.run(run_searches(args, theorems, model, logger))
    end_time = time.time()
    assert len(completed_searches) == len(theorems)

    print(f"Completed {len(completed_searches)} theorems.")
    print(f"Runtime: {end_time - start_time} seconds, {(end_time - start_time) / len(theorems)} seconds per theorem")
    logger.log_stats(completed_searches)


if __name__ == "__main__":
    main()