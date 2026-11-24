class UnsupportedOpError(Exception):
    """The backend doesn't support this (op_type, args); recorded as status="unsupported", not "error"."""


class FreshProcessError(Exception):
    """The case needs a process that has built nothing yet (e.g. a second engine in one process
    can hang); a CI part restarts the process and runs the case first. Not a result."""
