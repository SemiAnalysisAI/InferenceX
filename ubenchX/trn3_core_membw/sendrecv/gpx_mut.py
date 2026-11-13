"""Mutate bytes of every GPSIMD sendrecv instruction (Pool opcode 0xbf, 64 bytes) in a NEFF, on every core.

  python3 sendrecv/gpx_mut.py IN.neff OUT.neff OFF=VAL [OFF=VAL ...]     (OFF, VAL: byte offset in the instruction, new value)
  python3 sendrecv/gpx_mut.py IN.neff                                    (print the instructions)

Used to look for a field that names the partner core (see XSENG.md, "GPSIMD engine"). Layout as far as decoded:
  +0 opcode bf 10 | +4 wait: cond, semaphore | +6 on completion: action 0x13, semaphore | +8 wait value (u32)
  +12 src dtype, dst dtype | +16 src SBUF offset (u32), 6 x u16 shape/steps, +32 01, +33 partitions
  +34..+47 zero | +48 dst SBUF offset (u32), 6 x u16
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neffhack  # noqa: E402


def main():
    header, files, order = neffhack.unpack(sys.argv[1])
    edits = [tuple(int(v, 0) for v in a.split("=")) for a in sys.argv[3:]]
    for name in order:
        if not name.endswith("/Pool0.bin"):
            continue
        blob = bytearray(files[name])
        for i in range(0, len(blob), 64):
            if blob[i] == 0xBF:
                for off, val in edits:
                    blob[i + off] = val
                print(name, hex(i), blob[i:i + 64].hex())
        files[name] = bytes(blob)
    if len(sys.argv) > 2:
        neffhack.pack(sys.argv[2], header, files, order)


if __name__ == "__main__":
    main()
