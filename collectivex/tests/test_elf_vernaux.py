"""elf_vernaux.py retargets one version requirement in place and refuses anything else."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
import elf_vernaux  # noqa: E402


def _library(versions) -> bytes:
    """A minimal ELF64 with a string table and one .gnu.version_r entry (libc.so.6) whose aux
    records require `versions`, in order."""
    strings = b"\0libc.so.6\0" + b"".join(v.encode() + b"\0" for v in versions)
    offset_of = lambda text: strings.index(text.encode() + b"\0")
    verneed = struct.pack("<HHIII", 1, len(versions), offset_of("libc.so.6"), 16, 0)
    for index, version in enumerate(versions):
        step = 16 if index + 1 < len(versions) else 0
        verneed += struct.pack("<IHHII", elf_vernaux.elf_hash(version.encode()), 0, index + 2,
                               offset_of(version), step)
    strtab_offset = 64
    verneed_offset = strtab_offset + len(strings)
    shoff = verneed_offset + len(verneed)
    header = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    header += struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 3, 0)
    sections = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    sections += struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, strtab_offset, len(strings), 0, 0, 1, 0)
    sections += struct.pack("<IIQQQQIIQQ", 0, elf_vernaux.SHT_GNU_VERNEED, 0, 0, verneed_offset,
                            len(verneed), 1, 1, 8, 0)
    return header + strings + verneed + sections


class Retarget(unittest.TestCase):
    def _run(self, versions, old, new):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lib.so"
            path.write_bytes(_library(versions))
            before = path.read_bytes()
            elf_vernaux.retarget(str(path), old, new)
            return before, path.read_bytes()

    def test_the_old_requirement_takes_the_new_name_and_hash(self) -> None:
        before, after = self._run(["GLIBC_2.2.5", "GLIBC_2.38"], "GLIBC_2.38", "GLIBC_2.2.5")
        self.assertEqual(len(before), len(after))
        strtab = 64
        aux = before.index(struct.pack("<I", elf_vernaux.elf_hash(b"GLIBC_2.38")))
        new_hash, _, _, new_name, _ = struct.unpack_from("<IHHII", after, aux)
        self.assertEqual(new_hash, elf_vernaux.elf_hash(b"GLIBC_2.2.5"))
        self.assertEqual(new_name, before.index(b"GLIBC_2.2.5\0") - strtab)
        # Only those eight bytes change.
        changed = [i for i in range(len(before)) if before[i] != after[i]]
        self.assertTrue(changed and min(changed) >= aux and max(changed) < aux + 12)

    def test_a_missing_requirement_or_replacement_fails_closed(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(["GLIBC_2.2.5"], "GLIBC_2.38", "GLIBC_2.2.5")
        with self.assertRaises(SystemExit):
            self._run(["GLIBC_2.38"], "GLIBC_2.38", "GLIBC_2.2.5")

    def test_the_hash_is_the_sysv_elf_hash(self) -> None:
        # Values glibc's own verneed entries carry.
        self.assertEqual(elf_vernaux.elf_hash(b"GLIBC_2.2.5"), 0x09691A75)
        self.assertEqual(elf_vernaux.elf_hash(b"GLIBC_2.38"), 0x069691B8)


if __name__ == "__main__":
    unittest.main()
