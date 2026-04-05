"""Auto-patch SGLang allreduce shape logging via import hook.

When AR_SHAPE_LOGGING=1, installs a meta-path finder that waits for
sglang.srt.distributed.parallel_state to be imported, then applies the
monkey-patch to log tensor shapes entering the custom allreduce kernel.
"""
import importlib
import os
import sys


if os.environ.get("AR_SHAPE_LOGGING") == "1":

    class _AllReducePatchFinder:
        """Meta-path finder that triggers patching after parallel_state is imported."""
        _target = "sglang.srt.distributed.parallel_state"
        _done = False

        def find_module(self, fullname, path=None):
            if not self._done and fullname == self._target:
                return self
            return None

        def load_module(self, fullname):
            # Remove ourselves so we don't recurse
            self._done = True
            if self in sys.meta_path:
                sys.meta_path.remove(self)

            # Let the real import happen
            if fullname in sys.modules:
                mod = sys.modules[fullname]
            else:
                mod = importlib.import_module(fullname)

            # Now apply the patch
            try:
                _patch_dir = os.path.dirname(os.path.abspath(__file__))
                if _patch_dir not in sys.path:
                    sys.path.insert(0, _patch_dir)
                import allreduce_shape_logger
                allreduce_shape_logger.patch()
            except Exception as e:
                print(f"[AR_SHAPE] Deferred patch failed: {e}", flush=True)

            return mod

    sys.meta_path.insert(0, _AllReducePatchFinder())
    print("[AR_SHAPE] Import hook installed, will patch after parallel_state loads", flush=True)
