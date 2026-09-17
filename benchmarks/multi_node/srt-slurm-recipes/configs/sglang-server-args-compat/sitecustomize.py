"""TEMPORARY: restore ServerArgs.get_model_config for the pinned Dynamo wheel.

The reviewed SGLang branch these recipes import moved model-config resolution
out of ServerArgs, but the pinned Dynamo wheel still calls
``server_args.get_model_config()`` while parsing worker arguments. Re-attach
the accessor so the wheel keeps working against the newer source tree.

This file is picked up because its directory is on PYTHONPATH, so it is
imported by every interpreter in the job, including ones that never import
SGLang. Failing to import SGLang there is expected and must stay silent.

Remove this directory, its PYTHONPATH entry, and the rest of the temporary
source override once the optimizations ship in the pinned image.
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
