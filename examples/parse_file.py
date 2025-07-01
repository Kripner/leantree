from leantree import LeanProject, StoredError
from leantree import utils

project = LeanProject(create=True, logger=utils.Logger(utils.LogLevel.INFO))
file = project.load_file("Example3.lean", use_cache=False)

for thm in file.theorems:
    if isinstance(thm, StoredError):
        print(f"Error: {thm.error}")
        continue
    print(thm.load_source() + "\n")
    for by_block in thm.by_blocks:
        if isinstance(by_block.tree, StoredError):
            print(f"Error: {by_block.tree}")
        print(by_block.tree.pretty_print() + "\n")
    print("-" * 100)

