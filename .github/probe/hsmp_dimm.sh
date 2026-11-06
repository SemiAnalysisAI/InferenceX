#!/usr/bin/env bash
# Per-DIMM power via AMD HSMP. Loads amd_hsmp only if it was not loaded, and unloads it on exit.
set +e
W=$(mktemp -d /tmp/hsmp-probe.XXXX)
LOADED_BY_US=0
cleanup() {
  [ "$LOADED_BY_US" = 1 ] && { sudo -n rmmod amd_hsmp && echo "== amd_hsmp unloaded (restored)"; }
  pkill -f "$W/" 2>/dev/null; rm -rf "$W"
}
trap cleanup EXIT
echo "== cpu"; lscpu | grep -E "^(Model name|Socket\(s\))"
if lsmod | grep -q '^amd_hsmp'; then echo "amd_hsmp already loaded"; else
  sudo -n modprobe amd_hsmp 2>&1; lsmod | grep -q '^amd_hsmp' && { LOADED_BY_US=1; echo "amd_hsmp loaded"; } || echo "amd_hsmp failed to load"
fi
dmesg 2>/dev/null | grep -i hsmp | tail -5; sudo -n dmesg 2>/dev/null | grep -i hsmp | tail -5
ls -l /dev/hsmp* 2>&1
[ -e /dev/hsmp ] || { echo "RESULT: no /dev/hsmp, stopping"; exit 0; }

echo "== build e_smi_tool"
git clone -q --depth 1 https://github.com/amd/esmi_ib_library "$W/esmi" 2>&1 | tail -2
( cd "$W/esmi" && git log -1 --format=%h &&
  sed -e 's/@e_smi_VERSION_MAJOR@/4/;s/@e_smi_VERSION_MINOR@/0/;s/@e_smi_VERSION_PATCH@/0/;s/@e_smi_VERSION_BUILD@/probe/' src/e_smi64Config.in > include/e_smi/e_smi64Config.h &&
  gcc -O2 -Iinclude src/e_smi.c src/e_smi_monitor.c src/e_smi_plat.c src/e_smi_utils.c tools/e_smi_tool.c -o "$W/e_smi_tool" -lpthread -lrt -lm 2>&1 | grep -i error | head -5 )
E="sudo -n $W/e_smi_tool"
[ -x "$W/e_smi_tool" ] || { echo "RESULT: build failed"; exit 0; }
echo "== e_smi summary"; timeout 30s $E 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -i -E "proto|family|model|socket power|dimm|hsmp|version" | head -20
for s in 0 1; do echo "== total dimm power socket $s"; timeout 10s $E --showtotaldimmpower $s 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "Watts|UpdateRate|Fail|not" | head -4; done

echo "== scan DIMM addresses"
: > "$W/addrs"
for s in 0 1; do for a in $(seq 0 255); do
  h=$(printf "0x%x" $a)
  out=$(timeout 5s $E --showdimmpower $s $h 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  p=$(echo "$out" | grep "Power(Watts)" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
  [ -n "$p" ] && echo "$s $h" >> "$W/addrs"
done; done
echo "found $(wc -l < "$W/addrs") DIMM addresses:"; tr '\n' ' ' < "$W/addrs"; echo
[ -s "$W/addrs" ] || { timeout 5s $E --showdimmpower 0 0x80 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | head -5; echo "RESULT: no DIMM power responses"; exit 0; }
echo "== sample detail"; read s h < "$W/addrs"; timeout 5s $E --showdimmpower $s $h 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep "|"

sample() { # phase, seconds
  local end=$(( $(date +%s) + $2 ))
  while [ $(date +%s) -lt $end ]; do
    local ts=$(date +%s); local line="SAMPLE $1 $ts"
    while read s h; do
      out=$(timeout 5s $E --showdimmpower $s $h 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
      p=$(echo "$out" | grep "Power(Watts)" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
      r=$(echo "$out" | grep "PowerUpdateRate" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
      line="$line $s:$h=${p:-NA}@${r:-NA}"
    done < "$W/addrs"
    echo "$line"; sleep 5
  done
}

cat > "$W/stream.c" <<'C'
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
static int secs; static size_t N = 512UL<<20;
static void *run(void *x){ char *a=malloc(N),*b=malloc(N); memset(a,1,N); memset(b,2,N);
 time_t e=time(0)+secs; while(time(0)<e){ memcpy(b,a,N); memcpy(a,b,N);} return 0; }
int main(int c,char**v){ int t=atoi(v[1]); secs=atoi(v[2]); pthread_t p[512];
 for(int i=0;i<t;i++) pthread_create(&p[i],0,run,0); for(int i=0;i<t;i++) pthread_join(p[i],0); return 0; }
C
gcc -O2 -pthread "$W/stream.c" -o "$W/stream"

HIPCC=$(command -v hipcc || ls /opt/rocm*/bin/hipcc 2>/dev/null | head -1)
if [ -n "$HIPCC" ]; then cat > "$W/h2d.cpp" <<'C'
#include <hip/hip_runtime.h>
#include <thread>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main(int c,char**v){ int secs=atoi(v[1]); int dir=atoi(v[2]); int n=0; hipGetDeviceCount(&n); size_t B=1UL<<30;
 std::vector<std::thread> ts; std::vector<double> gb(n,0);
 for(int d=0;d<n;d++) ts.emplace_back([&,d]{ hipSetDevice(d); void *h,*g; hipHostMalloc(&h,B); hipMalloc(&g,B); hipMemset(g,0,B);
   auto e=std::chrono::steady_clock::now()+std::chrono::seconds(secs);
   while(std::chrono::steady_clock::now()<e){ if(dir==0) hipMemcpy(g,h,B,hipMemcpyHostToDevice); else hipMemcpy(h,g,B,hipMemcpyDeviceToHost); gb[d]+=1; }
   hipFree(g); hipHostFree(h); });
 for(auto&t:ts) t.join(); double s=0; for(double x:gb) s+=x; printf("GPUs %d total %.1f GB/s\n",n,s/secs); return 0; }
C
  "$HIPCC" -O2 "$W/h2d.cpp" -o "$W/h2d" 2>&1 | grep -i error | head -3
fi
echo "hipcc: ${HIPCC:-none}; h2d built: $([ -x "$W/h2d" ] && echo yes || echo no)"

NT=$(nproc)
sample idle 60
"$W/stream" $((NT/2)) 75 & sleep 10; sample cpu_stream_half 60; wait
"$W/stream" $NT 75 & sleep 10; sample cpu_stream_all 60; wait
if [ -x "$W/h2d" ]; then
  "$W/h2d" 100 0 & sleep 10; sample gpu_h2d 80; wait
  "$W/h2d" 100 1 & sleep 10; sample gpu_d2h 80; wait
fi
sleep 20; sample idle_after 40
echo "RESULT: done"
