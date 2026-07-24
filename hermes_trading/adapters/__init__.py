"""Data adapters. Each exposes `async def fetch(...) -> dict` with a schema_version.

A schema mismatch raises SchemaError so the loop halts loudly rather than
trading on data it doesn't understand.
"""


class SchemaError(RuntimeError):
    """Raised when an adapter's payload doesn't match the expected schema_version."""


def require_schema(payload: dict, expected: int, source: str) -> dict:
    got = payload.get("schema_version")
    if got != expected:
        raise SchemaError(f"{source}: expected schema_version={expected}, got {got!r}")
    return payload
