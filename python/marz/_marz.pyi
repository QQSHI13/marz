"""Type stubs for the Marz native extension.

Hand-written because PyO3 cannot generate them. Keep in sync with
`python/src/lib.rs`.
"""

from collections.abc import Iterable, Mapping
from typing import Any, final

__version__: str

class QueryError(ValueError):
    """A query string could not be parsed."""

    query: str
    start: int
    end: int

class FormatError(ValueError):
    """Bytes are not a valid Marz binary index."""

def languages() -> list[str]:
    """Language codes this build supports."""

def tokenize(text: str, language: str) -> list[str]:
    """Split `text` the way the index would."""

def index_language(data: bytes) -> str:
    """Read an index's language code without loading it."""

@final
class Result:
    """One search hit."""

    @property
    def ref(self) -> str: ...
    @property
    def score(self) -> float: ...
    @property
    def matches(self) -> dict[str, dict[str, list[tuple[int, int]]]]: ...
    @property
    def terms(self) -> list[str]: ...

@final
class Index:
    """A built search index."""

    def search(self, query: str) -> list[Result]: ...
    def to_bytes(self, *, positions: bool = True) -> bytes: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_bytes(data: bytes, language: str | None = None) -> Index: ...
    @staticmethod
    def from_json(data: str, language: str) -> Index: ...
    @property
    def fields(self) -> list[str]: ...
    @property
    def document_count(self) -> int: ...
    @property
    def term_count(self) -> int: ...
    @property
    def language(self) -> str: ...
    def __len__(self) -> int: ...

@final
class IndexBuilder:
    """Builds a search index."""

    def __init__(
        self,
        language: str,
        *,
        ref_field: str = "id",
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None: ...
    def field(self, name: str, boost: float = 1.0) -> None: ...
    def add(self, doc: Mapping[str, Any], boost: float = 1.0) -> None: ...
    def add_many(
        self, docs: Iterable[Mapping[str, Any]], boost: float = 1.0
    ) -> None: ...
    def build(self) -> Index: ...
    def clear(self) -> None: ...
    @property
    def staged(self) -> int: ...
    @property
    def fields(self) -> list[str]: ...
    @property
    def ref_field(self) -> str: ...
    @property
    def language(self) -> str: ...
