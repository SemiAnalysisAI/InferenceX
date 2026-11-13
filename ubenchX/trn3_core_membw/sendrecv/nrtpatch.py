"""Byte-patch a COPY of libnrt.so (Neuron runtime 2.35.30.0) to lift checks that keep a DMA inside an LNC pair.

  python3 sendrecv/nrtpatch.py IN.so OUT.so PATCH [PATCH ...]

Patches are named; each is (virtual address, expected bytes, replacement). The copy is bind-mounted over the
library inside a container, so the host install and every other user's runtime are untouched.

  lnc    parse_vnc_config: on this instance type any LNC size other than 2 is refused
         ("Only LNC Size of 2 is supported at this moment on this instance type"). 0x50d80f: je -> jmp.
         Not enough on its own: tdrv's vtpb_ctx_init then refuses a logical core size above 2.
  remote kelf_load (inlined kelf_resolve_vtpb_remote_variables): a "remote" variable SB{j} must name a
         subgraph of the SAME logical core ("refers to sg 2, outside of subgraphs 0 to 1"). 0x6f178a: the
         je to the error is removed, so a core may name any subgraph of the loaded NEFF.
"""
import struct
import sys

PATCHES = {
    "lnc": [(0x50d80f, bytes.fromhex("74a1"), bytes.fromhex("eba1"))],
    "remote": [(0x6f178a, bytes.fromhex("0f84bf000000"), bytes.fromhex("909090909090"))],
}


def file_offset(data, vaddr):
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    for i in range(phnum):
        p_type, _, p_offset, p_vaddr, _, p_filesz = struct.unpack_from("<IIQQQQ", data, phoff + i * phentsize)
        if p_type == 1 and p_vaddr <= vaddr < p_vaddr + p_filesz:
            return vaddr - p_vaddr + p_offset
    raise SystemExit(f"address {vaddr:#x} is not in a loaded segment")


def main():
    src, dst, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    data = bytearray(open(src, "rb").read())
    for name in names:
        for vaddr, want, new in PATCHES[name]:
            off = file_offset(data, vaddr)
            got = bytes(data[off:off + len(want)])
            if got != want:
                raise SystemExit(f"{name}: at {vaddr:#x} expected {want.hex()}, found {got.hex()}; wrong library version")
            data[off:off + len(new)] = new
            print(f"{name}: {vaddr:#x} (file {off:#x}) {want.hex()} -> {new.hex()}")
    open(dst, "wb").write(data)


if __name__ == "__main__":
    main()
