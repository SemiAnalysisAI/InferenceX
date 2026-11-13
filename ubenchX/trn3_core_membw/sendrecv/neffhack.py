"""Patch a compiled nisa.sendrecv NEFF so the SBUF-to-SBUF DMA crosses the LNC pair (the SEngine).

NKI 0.7 only lets sendrecv name the other core of an LNC=2 pair, and caps LNC at 2. AWS says the hardware
routes a DMA from any NeuronCore's SBUF to any other's, and that the limit is software ("hack the binary").
This edits the binary instead of the compiler.

What a NEFF is (neuronx-cc 2.28): a 1024-byte header, then a tar.gz. Header: payload length (u64 LE) at
0x10, the payload's MD5 at 0xac and again at 0xcc. Per physical core of the logical core, the tar has
sg<k>/def.json (variables, DMA queues), sg<k>/SP0.json (DMA descriptors as JSON), sg<k>/Pool0.bin and
SP0.bin (engine instructions). sendrecv with the default DMA engine compiles to a PULL: core k's SP0.json
holds one descriptor  {"op": "copy", "from": "SB{j}", "to": "SB", ...}  where "SB{j}" is a variable of
type "remote" in def.json: the state buffer of relative core j of the same logical core.

  python3 sendrecv/neffhack.py repack IN.neff OUT.neff           unpack and repack untouched (checks the packer)
  python3 sendrecv/neffhack.py remote IN.neff OUT.neff K=J ...   core K pulls from SB{J} instead (any J)
  python3 sendrecv/neffhack.py lnc IN.neff OUT.neff N K=J ...    grow to N cores by cloning sg(k % 2), then
                                                                 set the pulls; launch with LNC=N
  python3 sendrecv/neffhack.py show IN.neff                      print each core's remote variable and pull
"""
import hashlib
import io
import json
import sys
import tarfile


def unpack(path):
    raw = open(path, "rb").read()
    header, files = bytearray(raw[:1024]), {}
    with tarfile.open(fileobj=io.BytesIO(raw[1024:]), mode="r:gz") as tar:
        order = [m.name for m in tar.getmembers() if m.isfile()]
        for name in order:
            files[name] = tar.extractfile(name).read()
    return header, files, order


def pack(path, header, files, order):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in order:
            info = tarfile.TarInfo(name)
            info.size = len(files[name])
            tar.addfile(info, io.BytesIO(files[name]))
    payload = buf.getvalue()
    header[0x10:0x18] = len(payload).to_bytes(8, "little")
    digest = hashlib.md5(payload).digest()
    header[0xac:0xbc] = digest
    header[0xcc:0xdc] = digest
    open(path, "wb").write(bytes(header) + payload)


def set_pull(files, core, source):
    """Make core `core` pull from relative core `source`'s state buffer."""
    sg = f"sg{core:02d}"
    definition = json.loads(files[f"{sg}/def.json"])
    old = [name for name, var in definition["var"].items() if var.get("type") == "remote"]
    if len(old) != 1:
        raise SystemExit(f"{sg}: expected one remote variable, found {old}")
    new = f"SB{{{source}}}"
    definition["var"] = {(new if name == old[0] else name): var for name, var in definition["var"].items()}
    files[f"{sg}/def.json"] = json.dumps(definition, separators=(",", ":")).encode()
    for engine in ("SP0.json", "Pool0.json"):
        key = f"{sg}/{engine}"
        files[key] = files[key].decode().replace(f'"{old[0]}"', f'"{new}"').encode()


def grow(files, order, cores):
    """Clone sg00/sg01 into sg02.. so the NEFF describes `cores` physical cores."""
    have = len(json.loads(files["kelf-0.json"])["graphs"])
    for k in range(have, cores):
        src = f"sg{k % have:02d}/"
        for name in [n for n in order if n.startswith(src)]:
            clone = f"sg{k:02d}/" + name[len(src):]
            files[clone] = files[name]
            order.append(clone)
    kelf = json.loads(files["kelf-0.json"])
    kelf["graphs"] = [{"definition": f"sg{k:02d}/def.json", "name": f"sg{k:02d}"} for k in range(cores)]
    files["kelf-0.json"] = json.dumps(kelf, separators=(",", ":")).encode()
    neff = json.loads(files["neff.json"])
    node = neff["nodes"][1]
    outputs = node["output_names"][0]
    node["output_names"] = [outputs] * cores
    node["attrs"]["num_outputs"] = str(cores)
    neff["heads"] = [[1, k, 0] for k in range(cores)]
    neff["node_row_ptr"] = [0, 1, 1 + cores]
    for key in ("dltype", "shape", "storage_id"):
        kind, values = neff["attrs"][key]
        neff["attrs"][key] = [kind, values[:1] + [values[1] if key != "storage_id" else i + 1 for i in range(cores)]]
    files["neff.json"] = json.dumps(neff, separators=(",", ":")).encode()


def show(files):
    for k in range(len(json.loads(files["kelf-0.json"])["graphs"])):
        sg = f"sg{k:02d}"
        remote = {n: v for n, v in json.loads(files[f"{sg}/def.json"])["var"].items() if v.get("type") == "remote"}
        pulls = [(d["from"], d["from_off"], d["to"], d["to_off"]) for q in json.loads(files[f"{sg}/SP0.json"])["dma"]
                 for d in q["desc"]]
        print(sg, "remote:", remote, "pull (from, off, to, off):", pulls)


def main():
    mode, src = sys.argv[1], sys.argv[2]
    header, files, order = unpack(src)
    if mode == "show":
        return show(files)
    dst, rest = sys.argv[3], sys.argv[4:]
    if mode == "lnc":
        grow(files, order, int(rest[0]))
        rest = rest[1:]
    if mode in ("remote", "lnc"):
        for pair in rest:
            core, source = (int(v) for v in pair.split("="))
            set_pull(files, core, source)
    pack(dst, header, files, order)
    show(files)


if __name__ == "__main__":
    main()
