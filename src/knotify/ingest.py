"""Import boundary for the future read-only GitHub client."""

from knotify.contracts import Issue
from knotify.errors import ComponentUnavailable


def import_repository(repository: str) -> tuple[Issue, ...]:
    """Reserved entry point; pagination, PR filtering and snapshots are not implemented."""
    raise ComponentUnavailable("GitHub importing is not implemented; no network request was made.")
