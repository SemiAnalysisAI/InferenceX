"""TEMPORARY: restore ServerArgs.get_model_config for the pinned Dynamo wheel.

The image's SGLang moved model-config resolution out of ServerArgs, but the
pinned Dynamo wheel still calls ``server_args.get_model_config()`` while
parsing worker arguments. Re-attach the accessor so the wheel keeps working.

This file is picked up because its directory is on PYTHONPATH, so it is
imported by every interpreter in the job, including ones that never import
SGLang. Failing to import SGLang there is expected and must stay silent.

Remove this directory and its PYTHONPATH entry once the pinned Dynamo wheel
stops calling the accessor.
"""

try:
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.server_args import ServerArgs
except Exception:  # noqa: BLE001 - non-SGLang interpreters legitimately fail here
    pass
else:
    if not hasattr(ServerArgs, "get_model_config"):

        def get_model_config(self):
            return model_config_of(self)

        ServerArgs.get_model_config = get_model_config
