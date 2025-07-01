import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import re

import leantree.file_span
import leantree.utils
from leantree.repl_adapter.data import ReplCompilationUnit, ReplLoadedLeanFile, ReplProofStepInfo, FilePositionParser


class LeanFileParser:
    # TODO: use LeanEnvironment instead
    @classmethod
    def load_lean_file(
            cls,
            repl_exe_path: Path | str,
            project_path: Path | str,
            path: Path | str,
            use_cache: bool = True
    ) -> ReplLoadedLeanFile:
        # file_line_lengths = [len(line) for line in cls._preprocess_file_content(path.read_text()).splitlines(keepends=True)]
        file_line_lengths = [len(line) for line in path.read_text().splitlines(keepends=True)]

        units = LeanFileParser.load_compilation_units(
            repl_exe_path,
            project_path,
            path,
            file_line_lengths,
            use_cache,
        )
        imports = LeanFileParser.load_lean_imports(path)

        return ReplLoadedLeanFile(
            Path(path),
            units,
            imports,
            file_line_lengths,
        )

    @classmethod
    def load_compilation_units(
            cls,
            repl_exe_path: Path | str,
            project_path: Path | str,
            file: Path | str,
            file_line_lengths: list[int],
            use_cache: bool = True,
    ) -> list[ReplCompilationUnit]:
        data = cls.run_lean_on_file(repl_exe_path, project_path, file, use_cache)

        result = []
        global_context = GlobalContextTracker()
        for edges, root_info_tree in zip(data["proof_tree_edges"], data["info_trees"]):
            src_range = root_info_tree["node"]["stx"]["range"]
            if src_range["synthetic"]:
                assert len(edges) == 0
                continue
            pretty_print = root_info_tree["node"]["stx"]["pp"]
            if pretty_print:
                pretty_print = leantree.utils.remove_empty_lines(leantree.utils.remove_comments(pretty_print))
                global_context.next_compilation_unit(pretty_print)
            span = FilePositionParser.create_file_span(src_range, file_line_lengths)
            unit = ReplCompilationUnit(
                [ReplProofStepInfo.from_repl_data(step, file_line_lengths) for step in edges],
                pretty_print,
                span,
                global_context.get_context(),
            )
            result.append(unit)
        return result

    @classmethod
    def load_lean_imports(cls, path: Path | str) -> list[str]:
        assert str(path).endswith(".lean")
        imports = []
        with open(path) as f:
            for line in f:
                if line.startswith("import "):
                    imports.append(line[len("import "):].strip())
        return imports

    @classmethod
    def run_lean_on_file(
            cls, repl_exe_path: Path | str, project_path: Path | str, file: Path | str, use_cache: bool
    ) -> dict:
        if use_cache:
            cached_data = cls._load_cache(file)
            if cached_data:
                return cached_data

        repl_exe_path = Path(repl_exe_path).absolute()
        cmd = ["lake", "env", str(repl_exe_path)]
        # file_content = cls._preprocess_file_content(file.read_text())
        # input_data = json.dumps({"cmd": file_content, "proofTrees": True, "infotree": "no_children"}) + "\n\n\n"
        input_data = json.dumps({"path": str(file), "proofTrees": True, "infotree": "no_children"}) + "\n\n\n"
        # print(f"Running '{" ".join(cmd)}' in '{project_path}' with input '{input_data.strip()}' (stripped)")
        result = subprocess.run(
            cmd,
            cwd=project_path,
            input=input_data,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            is_empty_line_error = result.returncode == 1 and result.stderr.strip() == 'uncaught exception: {"message": "Could not parse JSON:\\noffset 1: unexpected end of input"}'
            if not is_empty_line_error:
                raise RuntimeError(f"Command failed: {result.stderr}")

        repl_data = json.loads(result.stdout)

        if "message" in repl_data:
            raise Exception(f"lean-repl returned an error: {repl_data["message"]}")

        assert len(repl_data["proofTreeEdges"]) == len(repl_data["infotree"])
        if any(msg["severity"] == "error" for msg in repl_data.get("messages", [])):
            raise Exception(f"lean-repl returned some errors: {repl_data["messages"]}")
        data = {
            "proof_tree_edges": repl_data["proofTreeEdges"],
            "info_trees": repl_data["infotree"],
            "messages": repl_data.get("messages")
        }
        if use_cache:
            cls._save_cache(file, data)
        return data

    # TODO: this is too unreliable
    # @classmethod
    # def _preprocess_file_content(cls, content: str) -> str:
    #     result = []
    #     for line in content.splitlines(keepends=True):
    #         exacts_match = re.match(r"([ \t]*)exacts \[([^\n]+)]", line.rstrip())
    #         if exacts_match:
    #             indent = exacts_match.group(1)
    #             for term in exacts_match.group(2).split(","):
    #                 result.append(f"{indent}exact {term.strip()}\n")
    #         else:
    #             result.append(line)
    #     return "".join(result)

    @classmethod
    def _load_cache(cls, path: Path | str, cache_extension: str = ".replcache") -> dict | None:
        cache_path = Path(str(path) + cache_extension)
        if not cache_path.is_file():
            return None
        with open(cache_path) as f:
            data = json.load(f)
        cache_timestamp = data["timestamp"]
        file_last_modified = Path(path).stat().st_mtime
        if cache_timestamp != file_last_modified:
            cache_path.unlink()
            return None
        del data["timestamp"]
        return data

    @classmethod
    def _save_cache(cls, path: Path | str, data: dict, cache_extension: str = ".replcache"):
        cache_path = Path(str(path) + cache_extension)
        file_last_modified = Path(path).stat().st_mtime
        print(f"Saving REPL data cache to {cache_path}")
        with open(cache_path, "w") as f:
            # noinspection PyTypeChecker
            json.dump({
                **data,
                "timestamp": file_last_modified,
            }, f)


class GlobalContextTracker:
    @dataclass
    class Section:
        name: str
        context: list[str] = field(default_factory=list)

    def __init__(self):
        self.sections_stack = [GlobalContextTracker.Section("")]

    def next_compilation_unit(self, source: str):
        context_clauses = ["open "]
        section_clauses = ["section", "noncomputable section", "namespace "]
        if any(source.startswith(c) for c in context_clauses):
            assert "\n\n" not in source
            self.sections_stack[-1].context.append(source)
        elif any(source.startswith(c) for c in section_clauses):
            assert "\n" not in source
            clause = next(c for c in section_clauses if source.startswith(c))
            section = source[len(clause):].strip()
            assert " " not in section
            for part in section.split("."):
                self.sections_stack.append(GlobalContextTracker.Section(part))
        elif source.startswith("end"):
            assert "\n" not in source
            section = source[len("end"):].strip()
            assert " " not in section
            for part in reversed(section.split(".")):
                assert part == "" or part == self.sections_stack[-1].name
                assert len(self.sections_stack) > 1
                self.sections_stack.pop()

    def get_context(self) -> list[str]:
        return [ctx for section in self.sections_stack for ctx in section.context]
