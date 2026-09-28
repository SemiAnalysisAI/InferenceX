class UnsupportedOpError(Exception):
    """The backend doesn't support this (op_type, args); recorded as status="unsupported", not "error"."""
