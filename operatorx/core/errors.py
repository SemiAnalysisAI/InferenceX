class UnsupportedOpError(Exception):
    """Backend cannot execute this op (wrong dtype, wrong category, kernel missing, ...).

    Distinct from a runtime crash: the backend simply doesn't claim support for this
    (op_type, args) combination. Smoke tests record this as ``status="unsupported"``
    rather than ``"error"``.
    """
