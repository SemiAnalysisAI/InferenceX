#!/usr/bin/env bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential bzip2 ca-certificates curl libibverbs-dev librdmacm-dev numactl patch
curl -fL --retry 3 https://github.com/ofiwg/libfabric/releases/download/v1.22.0/libfabric-1.22.0.tar.bz2 -o /tmp/libfabric.tar.bz2
tar xjf /tmp/libfabric.tar.bz2 -C /tmp
cd /tmp/libfabric-1.22.0
python3 - <<'PY'
from pathlib import Path
p = Path("prov/efa/src/efa_device.c")
s = p.read_text()
old = "\tfor (device_idx = 0; device_idx < g_device_cnt; device_idx++) {\n\t\terr = efa_device_construct(&g_device_list[device_idx], device_idx, ibv_device_list[device_idx]);\n\t\tif (err) {\n\t\t\tret = err;\n\t\t\tgoto err_free;\n\t\t}\n\t}\n"
new = "\tfor (device_idx = 0; device_idx < g_device_cnt; device_idx++) {\n\t\terr = efa_device_construct(&g_device_list[efa_device_cnt], efa_device_cnt, ibv_device_list[device_idx]);\n\t\tif (err) continue;\n\t\tefa_device_cnt++;\n\t}\n\tif (efa_device_cnt == 0) {\n\t\tret = -FI_ENODEV;\n\t\tgoto err_free;\n\t}\n\tg_device_cnt = efa_device_cnt;\n"
decl = "\tstruct ibv_device **ibv_device_list;\n\tint device_idx;\n\tint ret, err;\n\tstatic bool initialized = false;"
assert s.count(old) == 1 and s.count(decl) == 1
p.write_text(s.replace(old, new).replace(decl, decl.replace("\tint device_idx;", "\tint device_idx;\n\tint efa_device_cnt = 0;")))
PY
NVML_HEADER="$(find /usr/local/cuda /usr/include /usr/local -name nvml.h -print -quit 2>/dev/null)"
if [[ -z "$NVML_HEADER" ]]; then
    apt-get install -y -qq --no-install-recommends cuda-nvml-dev-13-0
    NVML_HEADER="$(find /usr/local/cuda /usr/include /usr/local -name nvml.h -print -quit 2>/dev/null)"
fi
test -n "$NVML_HEADER"
CPPFLAGS="-I$(dirname "$NVML_HEADER")" ./configure --prefix=/opt/libfabric --enable-efa --with-cuda=/usr/local/cuda --enable-cuda-dlopen --disable-psm2 --disable-psm3 --disable-opx --disable-verbs --disable-usnic --disable-rxm --disable-rxd --disable-mrail --disable-sockets --disable-udp --disable-tcp --disable-shm >/dev/null
make -j16 >/dev/null
make install >/dev/null
printf '%s\n' /opt/libfabric/lib >/etc/ld.so.conf.d/libfabric-efa.conf
ldconfig
/opt/libfabric/bin/fi_info -p efa >/tmp/fi-info
grep -q 'provider: efa' /tmp/fi-info
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13
python3 -m pip install --no-deps mooncake-transfer-engine-efa-cuda13==0.3.13.post1
patch --batch --forward -d /usr/local/lib/python3.12/dist-packages -p1 </configs/patches/vllm-separate-cudagraph-pools.patch
