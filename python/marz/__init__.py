"""Marz offline search engine."""

try:
    from ._marz import IndexBuilder
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Marz native extension is not built. Run `maturin build` in python/."
    ) from exc

__all__ = ["IndexBuilder"]
