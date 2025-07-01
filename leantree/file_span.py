from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Self


@dataclass(frozen=True)
class FilePosition:
    # Character offset, assuming '\n' newlines.
    offset: int

    def __post_init__(self):
        assert self.offset >= 0

    @classmethod
    def beginning_of_file(cls) -> Self:
        return cls(0)

    @classmethod
    def end_of_file(cls, content: str) -> Self:
        return FilePosition(len(content))

    def relative_to(self, origin: Self) -> Self:
        return FilePosition(self.offset - origin.offset)

    def __lt__(self, other: Self):
        if not isinstance(other, FilePosition):
            return NotImplemented
        return self.offset < other.offset

    def __gt__(self, other: Self):
        if not isinstance(other, FilePosition):
            return NotImplemented
        return self.offset > other.offset

    def __le__(self, other: Self):
        if not isinstance(other, FilePosition):
            return NotImplemented
        return self.offset <= other.offset

    def __ge__(self, other: Self):
        if not isinstance(other, FilePosition):
            return NotImplemented
        return self.offset >= other.offset

    def __cmp__(self, other: Self):
        if not isinstance(other, FilePosition):
            return NotImplemented
        return self.offset.__cmp__(other.offset)


@dataclass(frozen=True)
class FileSpan:
    # The span is exclusive.
    start: FilePosition
    finish: FilePosition

    def serialize(self) -> dict:
        return {
            "start": self.start.offset,
            "finish": self.finish.offset,
        }

    @classmethod
    def deserialize(cls, data: dict) -> Self:
        return cls(
            FilePosition(data["start"]),
            FilePosition(data["finish"]),
        )

    @classmethod
    def whole_file(cls, content: str) -> Self:
        return cls(FilePosition.beginning_of_file(), FilePosition.end_of_file(content))

    def read_from_file(self, path: Path) -> str:
        with path.open("r", encoding="utf-8") as f:
            return self.read_from_string(f.read())

    def read_from_string(self, content: str) -> str:
        return content[self.start.offset:self.finish.offset]

    def contains(self, other: Self) -> bool:
        return self.start <= other.start <= other.finish <= self.finish

    def relative_to(self, origin: FilePosition) -> Self:
        return FileSpan(
            self.start.relative_to(origin),
            self.finish.relative_to(origin),
        )

    @classmethod
    def replace_spans(cls, base_string, replacement: str, spans: list[Self]) -> str:
        curr_position = FilePosition.beginning_of_file()
        result = []
        for span in sorted(spans, key=lambda s: s.start):
            assert curr_position <= span.start
            result.append(FileSpan(curr_position, span.start).read_from_string(base_string))
            result.append(replacement)
            curr_position = span.finish
        assert curr_position <= FilePosition.end_of_file(base_string)
        result.append(FileSpan(curr_position, FilePosition.end_of_file(base_string)).read_from_string(base_string))
        return "".join(result)

    @classmethod
    def get_containing_span(cls, spans: list[Self]) -> Self:
        min_start = min(span.start for span in spans)
        max_finish = max(span.finish for span in spans)
        return FileSpan(min_start, max_finish)

    @classmethod
    def merge_contiguous_spans(cls, spans: list[Self], content: str, should_merge: Callable[[str], bool]) -> list[Self]:
        if len(spans) == 0:
            return []
        spans = sorted(spans, key=lambda s: s.start)
        curr_span = spans[0]
        result = []
        for span in spans[1:]:
            if span.start <= curr_span.finish:
                curr_span = FileSpan(curr_span.start, span.finish)
                continue
            inbetween = FileSpan(curr_span.finish, span.start).read_from_string(content)
            if should_merge(inbetween):
                curr_span = FileSpan(curr_span.start, span.finish)
            else:
                result.append(curr_span)
                curr_span = span
        result.append(curr_span)
        return result
