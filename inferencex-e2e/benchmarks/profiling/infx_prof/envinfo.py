"""env/<rank>.json: versions, device identity (to join the clock samples) and engine config."""

import os
import time

from . import runtime


def source_tree(module):
    """The git checkout a package runs from (a dev image builds from a source tree whose reported version names only its base release): path, commit, branch, dirty."""
    path = os.path.dirname(os.path.abspath(getattr(module, "__file__", "") or ""))
    while path and path != os.path.dirname(path):
        dot_git = os.path.join(path, ".git")
        if os.path.exists(dot_git):
            break
        path = os.path.dirname(path)
    else:
        return None
    out = {"path": path}
    try:
        import subprocess

        def run(*args):
            return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        out["commit"] = run("rev-parse", "HEAD") or None
        out["branch"] = run("rev-parse", "--abbrev-ref", "HEAD") or None
        out["dirty"] = bool(run("status", "--porcelain", "--untracked-files=no"))
        if out["commit"]:
            return out
    except Exception:
        pass
    try:
        meta = dot_git
        if os.path.isfile(meta):
            with open(meta) as f:
                meta = os.path.join(path, f.read().split(":", 1)[1].strip())
        with open(os.path.join(meta, "HEAD")) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            out["branch"] = ref.rsplit("/", 1)[-1]
            ref_path = os.path.join(meta, ref)
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    out["commit"] = f.read().strip()
            else:
                with open(os.path.join(meta, "packed-refs")) as f:
                    out["commit"] = next((line.split()[0] for line in f if line.rstrip().endswith(ref)), None)
        else:
            out["commit"] = head
    except Exception as e:
        out["error"] = repr(e)
    return out


def write(rank, packages, env_prefixes, extra=None):
    try:
        import torch

        info = {"pid": os.getpid(), "rank": rank, "t_ns": time.time_ns()}
        for name in tuple(packages) + ("torch", "triton", "flashinfer", "deep_gemm"):
            try:
                module = __import__(name)
                info[f"{name}_version"] = getattr(module, "__version__", None)
                if name in packages:
                    info[f"{name}_source"] = source_tree(module)
            except Exception as e:
                info[f"{name}_version"] = f"unavailable: {e}"
        info["cuda_runtime"] = torch.version.cuda
        info["hip_runtime"] = getattr(torch.version, "hip", None)
        try:
            info["device_name"] = torch.cuda.get_device_name()
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            info["device_uuid"] = str(props.uuid)
            info["device_bdf"] = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        except Exception as e:
            info.setdefault("device_uuid", f"unavailable: {e}")
        info["env"] = {k: v for k, v in os.environ.items()
                       if k.startswith(tuple(env_prefixes) + ("INFX_", "TORCH", "CUDA", "NCCL", "INDUCTOR"))}
        info.update(extra or {})
        runtime.write_json(os.path.join("env", f"{rank}.json"), info)
    except Exception:
        runtime.write_error("write env")
