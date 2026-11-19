"""Explicit boundaries between missing implementations and real runtime failures."""


class ComponentUnavailable(RuntimeError):
    """A component has not been implemented or its required artifact is absent."""
