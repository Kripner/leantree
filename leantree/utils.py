import argparse
import asyncio
import datetime
import functools
import os
import re
from enum import Enum
from pathlib import Path
from typing import Set, Callable, AsyncIterator, Self

from PrettyPrint import PrettyPrintTree

from leantree.file_span import FileSpan


def to_sync(func):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If there's no event loop in the current thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return loop.run_until_complete(func(*args, **kwargs))
    return wrapper

class AsyncToSyncIterator:
    """
    A synchronous iterator wrapper for an asynchronous iterator.
    
    This class allows asynchronous iterators to be used in synchronous contexts
    by converting async iteration to sync iteration.
    """
    def __init__(self, async_iter: AsyncIterator, loop: asyncio.AbstractEventLoop):
        self.async_iter = async_iter
        self.loop = loop
        
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            # Use run_until_complete to get the next item synchronously
            return self.loop.run_until_complete(self.async_iter.__anext__())
        except StopAsyncIteration:
            raise StopIteration

def to_sync_iterator(func):
    """
    Decorator to convert an async iterator function to a sync iterator function.
    
    This decorator takes an async function that returns an AsyncIterator and
    converts it to a synchronous function that returns a regular Iterator.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there's no event loop in the current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        async_iter = func(*args, **kwargs)
        return AsyncToSyncIterator(async_iter, loop)
    
    return wrapper

def pretty_print_tree[TypeNode](
        root: TypeNode,
        get_children: Callable[[TypeNode], list[TypeNode]],
        node_to_str: Callable[[TypeNode], str],
        edge_to_str: Callable[[TypeNode], str | None] | None = None,
        max_label_len=55,
        max_edge_label_len=None,
) -> str:
    def trimmed_edge_to_str(e: TypeNode) -> str | None:
        if edge_to_str is None:
            return None
        s = edge_to_str(e)
        if max_edge_label_len is None:
            return s
        if s is None:
            return s
        if len(s) > max_edge_label_len:
            dots = "..."
            return s[:max_edge_label_len - len(dots)] + dots
        return s

    pt = PrettyPrintTree(
        get_children=get_children,
        get_val=node_to_str,
        get_label=trimmed_edge_to_str,
        return_instead_of_print=True,
        # border=True,
        trim=max_label_len,
    )
    return pt(root)



def get_args_descriptor(
        args_ns: argparse.Namespace,
        *args,
        **kwargs,
):
    return get_dict_descriptor(vars(args_ns), *args, **kwargs)


def get_dict_descriptor(
        args: dict,
        param_blacklist: Set[str] | None = None,
        param_whitelist: Set[str] | None = None,
        extra_args: dict[str, object] | None = None,
        include_slurm_id=True,
        include_time=True,
) -> str:
    if include_time:
        descriptor = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    else:
        descriptor = ""

    if include_slurm_id and "SLURM_JOB_ID" in os.environ:
        if len(descriptor) > 0:
            descriptor += "-"
        descriptor += f"id={os.environ['SLURM_JOB_ID']}"

    visible_args = {k: v for k, v in sorted(args.items())}
    if param_blacklist is not None:
        visible_args = {k: v for k, v in visible_args.items() if k not in param_blacklist}
    if param_whitelist is not None:
        visible_args = {k: v for k, v in visible_args.items() if k in param_whitelist}

    if extra_args is not None:
        visible_args = {**visible_args, **extra_args}

    def format_value(v: str) -> str:
        if isinstance(v, Path) or "/" in str(v):
            v = str(v)
            if v.endswith("/"):
                v = v[:-1]
            parts = [p for p in v.split("/") if len(p) != 0]
            return "_".join([v[:50] for v in parts[-2:]])
        if isinstance(v, str):
            return v.replace("<", "").replace(">", "")
        return str(v)

    if len(visible_args) > 0:
        if len(descriptor) > 0:
            descriptor += "-"
        descriptor += ",".join((
            "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), format_value(v))
            for k, v in visible_args.items()
        ))

    assert len(descriptor) > 0

    return descriptor


class ValueOrError[SomeValue]:
    def __init__(self, value: SomeValue | None, error: str | None):
        assert (value is None) != (error is None)
        self._value = value
        self._error = error

    @classmethod
    def from_success(cls, value: SomeValue) -> Self:
        return cls(value, None)

    @classmethod
    def from_error(cls, error: str) -> Self:
        return cls(None, error)

    def is_success(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> SomeValue:
        assert self.is_success()
        return self._value

    @property
    def error(self) -> SomeValue:
        assert not self.is_success()
        return self._error


class LogLevel(Enum):
    SUPPRESS = 0
    SUPPRESS_AND_STORE = 1
    INFO = 2
    DEBUG = 3


# TODO: replace with something built-in
class Logger:
    def __init__(self, log_level: LogLevel):
        self.log_level = log_level
        self._stored_messages = None
        if log_level == LogLevel.SUPPRESS_AND_STORE:
            self._stored_messages = []

    def info(self, msg: str):
        if self.log_level in [LogLevel.INFO, LogLevel.DEBUG]:
            print(msg)
        elif self.log_level == LogLevel.SUPPRESS_AND_STORE:
            self._stored_messages.append((msg, LogLevel.INFO))

    warning = info

    def debug(self, msg: str):
        if self.log_level == LogLevel.DEBUG:
            print(msg)
        elif self.log_level == LogLevel.SUPPRESS_AND_STORE:
            self._stored_messages.append((msg, LogLevel.DEBUG))

    def print_stored(self, log_level: LogLevel):
        assert log_level in [LogLevel.INFO, LogLevel.DEBUG]
        for msg, msg_level in self._stored_messages:
            if log_level.value >= msg_level.value:
                print(msg)
        self.delete_stored()

    def delete_stored(self):
        assert self._stored_messages is not None
        self._stored_messages = []


class NullLogger(Logger):
    def __init__(self):
        super().__init__(LogLevel.SUPPRESS)

    def info(self, msg: str):
        pass

    def debug(self, msg: str):
        pass

    def print_stored(self, log_level: LogLevel):
        pass

    def delete_stored(self):
        pass


# TODO: unit test
# TODO: fix this
def remove_comments(source: str) -> str:
    inside_comment = False
    result = []
    for line in source.splitlines():
        result_line = ""
        to_process = line
        while len(to_process) > 0:
            relevant_tokens = ["-/"] if inside_comment else ["--", "/-"]
            indices = [to_process.index(tok) for tok in relevant_tokens if tok in to_process]
            if not indices:
                if not inside_comment:
                    result_line += to_process
                break
            first_idx = min(indices)
            match to_process[first_idx:first_idx + 2]:
                case "-/":
                    inside_comment = False
                    to_process = to_process[first_idx + 2:]
                case "/-":
                    inside_comment = True
                    result_line += to_process[:first_idx]
                    to_process = to_process[first_idx + 2:]
                case "--":
                    result_line += to_process[:first_idx]
                    to_process = ""
        if inside_comment and not result_line:
            continue
        result.append(result_line)
    return "\n".join(result)


def remove_empty_lines(s: str) -> str:
    return "\n".join([l for l in s.splitlines() if l.strip()])


def is_just_comments(s: str) -> bool:
    return remove_empty_lines(remove_comments(s)).strip() == ""

def replace_with_sorries(theorem_str: str, sorries_mask: list[FileSpan]) -> str:
    return get_source_with_sorries(
        FileSpan.whole_file(theorem_str),
        sorries_mask,
        theorem_str,
    )

def get_source_with_sorries(
        span: FileSpan,
        sorries_mask: list[FileSpan] | None,
        file_content: str | None = None,
        file_path: Path | str | None = None,
) -> str:
    if file_content is None:
        with file_path.open("r", encoding="utf-8") as f:
            file_content = f.read()
    if not sorries_mask:
        return span.read_from_string(file_content)
    result = ""
    curr_position = span.start
    for mask_span in sorted(sorries_mask, key=lambda s: s.start):
        assert curr_position <= mask_span.start <= mask_span.finish <= span.finish
        result += FileSpan(curr_position, mask_span.start).read_from_string(file_content)
        result += "sorry"
        curr_position = mask_span.finish
    result += FileSpan(curr_position, span.finish).read_from_string(file_content)
    return result
