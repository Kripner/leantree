class LeanFileLoader:
    def load_file(self, path: Path, verbose: bool = False, use_repl_cache: bool = True) -> tuple[LeanFile, list[str]]:
        assert path.is_file()
        assert str(path).endswith(".lean")

        theorems = []
        failed_theorems = []
        parsed_file = LeanFile(file.path, file.imports, theorems)

        for unit in file.units:
            if len(unit.proof_steps) == 0:
                continue

            try:
                self._check_if_unit_supported(unit)
            except AssertionError as e:
                e.args += ("some features not supported yet", f"in file: {path}", f"in unit:\n'{unit.pretty_print}'")
                failed_theorems.append("\n".join(str(arg) for arg in e.args))
                continue

            try:
                trees = self._create_proof_trees(unit.proof_steps, create_file_span, verbose=verbose)
            except AssertionError as e:
                e.args += ("during proof tree creation", f"in file: {path}", f"in unit:\n'{unit.pretty_print}'")
                failed_theorems.append("\n".join(str(arg) for arg in e.args))
                continue

            assert len(trees) > 0

            by_blocks = [LeanTacticBlock.from_tree(t) for t in trees]

            # @classmethod
            # def from_tree(cls, tree: ProofTree) -> "LeanTacticBlock":
            #     return LeanTacticBlock(
            #         None,  # TODO: fix
            #         tree,
            #         FileSpan.get_containing_span([
            #             n.tactic.span for n in tree.get_nodes()
            #             if n.tactic is not None and n.tactic.span is not None
            #         ])
            #     )

            try:
                postprocessor = ProofTreePostprocessor(file, unit)
                for t in trees:
                    postprocessor.transform_proof_tree(t)
            except AssertionError as e:
                e.args += ("during proof tree transformation", f"in file: {path}", f"in unit:\n'{unit.pretty_print}'")
                failed_theorems.append("\n".join(str(arg) for arg in e.args))
                continue

            thm = LeanTheorem(parsed_file, unit.span, by_blocks, unit.global_context)
            for block in by_blocks:
                block.theorem = thm
            theorems.append(thm)
        return parsed_file, failed_theorems
