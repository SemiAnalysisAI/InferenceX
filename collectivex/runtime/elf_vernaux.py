"""Retarget one version requirement of a shared library to another version of the same library.

Broadcom's host libbnxt_re is linked on Ubuntu 24.04 and needs GLIBC_2.38 for one symbol,
__isoc23_strtol; the ROCm images ship glibc 2.35. After that symbol is unversioned and supplied by a
shim, the GLIBC_2.38 entry in .gnu.version_r still makes the loader refuse the library. This rewrites
that entry's name and hash to a version libc does provide, in place; nothing else in the file moves.

usage: elf_vernaux.py LIBRARY FROM_VERSION TO_VERSION
"""
import struct
import sys

SHT_GNU_VERNEED = 0x6FFFFFFE


def elf_hash(name: bytes) -> int:
    h = 0
    for byte in name:
        h = ((h << 4) + byte) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def retarget(path: str, old: str, new: str) -> None:
    with open(path, "r+b") as stream:
        data = bytearray(stream.read())
        if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
            raise SystemExit("not a little-endian ELF64 file")
        shoff, = struct.unpack_from("<Q", data, 0x28)
        shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
        sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
        verneed = [s for s in sections if s[1] == SHT_GNU_VERNEED]
        if len(verneed) != 1:
            raise SystemExit("expected one .gnu.version_r section")
        _, _, _, _, offset, _, link, info, _, _ = verneed[0]
        strtab = sections[link][4]

        def name_at(index: int) -> bytes:
            start = strtab + index
            return bytes(data[start:data.index(0, start)])

        targets, replacement = [], None
        entry = offset
        for _ in range(info):
            _, _, _, aux, following = struct.unpack_from("<HHIII", data, entry)
            cursor = entry + aux
            while True:
                _, _, _, name, step = struct.unpack_from("<IHHII", data, cursor)
                if name_at(name) == old.encode():
                    targets.append(cursor)
                if name_at(name) == new.encode():
                    replacement = name
                if not step:
                    break
                cursor += step
            if not following:
                break
            entry += following
        if not targets:
            raise SystemExit(f"{old} is not required")
        if replacement is None:
            raise SystemExit(f"{new} is not already required, so its name is not in the string table")
        for cursor in targets:
            struct.pack_into("<I", data, cursor, elf_hash(new.encode()))
            struct.pack_into("<I", data, cursor + 8, replacement)
        stream.seek(0)
        stream.write(data)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    retarget(*sys.argv[1:])
