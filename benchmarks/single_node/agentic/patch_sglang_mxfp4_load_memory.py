"""Release dead shuffle temporaries in the pinned SGLang MXFP4 loader."""

import argparse
import hashlib
from pathlib import Path

UPSTREAM_SHA256 = "0ff4ca142e71ec0baabc00844b4854e554c607c1a356ffe13419c59b1f51813c"
ANCHOR = b"        layer.w2_weight = Parameter(torch.stack(g2_w), requires_grad=False)\n"
REPLACEMENT = (
    b"        # The completed w13 stacks own their storage; release copied inputs.\n"
    b"        del g1_w, g1_s\n" + ANCHOR
)


def apply_patch(path: Path) -> None:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if REPLACEMENT in source:
        original = source.replace(REPLACEMENT, ANCHOR, 1)
        if hashlib.sha256(original).hexdigest() == UPSTREAM_SHA256:
            print(f"MXFP4 load-memory patch already applied: {digest}")
            return
    if digest != UPSTREAM_SHA256:
        raise ValueError(f"Unrecognized SGLang MXFP4 loader SHA256: {digest}")
    if source.count(ANCHOR) != 1:
        raise ValueError("Expected exactly one MXFP4 shuffle allocation site")
    patched = source.replace(ANCHOR, REPLACEMENT, 1)
    compile(patched, str(path), "exec")
    path.write_bytes(patched)
    print(f"Applied MXFP4 load-memory patch: {hashlib.sha256(patched).hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    apply_patch(parser.parse_args().path)


if __name__ == "__main__":
    main()
