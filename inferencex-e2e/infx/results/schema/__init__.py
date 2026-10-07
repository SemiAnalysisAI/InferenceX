"""Versioned contract for published result rows.

Producers on bare runner Python and in serving containers import this module, so it stays
stdlib-only; the Pydantic models live in ``models``.
"""

RESULT_SCHEMA_VERSION = 1
