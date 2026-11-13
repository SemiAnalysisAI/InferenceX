#!/bin/bash
# Single-byte mutations of the GPSIMD sendrecv instruction, run on two stock logical cores at once
# (L0 = nc4+nc5, L1 = nc6+nc7) with distinguishable inputs: does any field change WHO a core exchanges with?
#   sendrecv/gpx_fuzz.sh "OFFSETS" "VALUES"     output runs/gpsimd-x/fuzz-<off>-<val>, summary runs/gpsimd-x/fuzz.txt
W=/opt/dlami/nvme/trn3ub; G=$W/runs/gpsimd-x; cd $W/src
base=$W/runs/xseng/b1/xchg_gpsimd/4/n1/kernel.neff
for off in $1; do for val in $2; do
  python3 sendrecv/gpx_mut.py $base $G/mut.neff $off=$val > /dev/null || continue
  out=$(NEFF=$G/mut.neff TMO=3 bash sendrecv/gpx.sh fuzz-$off-$val xchg_gpsimd 4 1 - "0 1")
  echo "off=$off val=$val | $(echo "$out" | grep -E '^RESULT|^HOLDS' | sed -e 's/RESULT model \([0-9]\) vnc . status \([0-9]*\).*/m\1:st\2/' -e 's/HOLDS model \([0-9]\) rank \([0-9]\) : tile of model \([0-9]\) rank \([0-9]\)/\1.\2<-\3.\4/' -e 's/HOLDS model \([0-9]\) rank \([0-9]\) : not a whole.*values/\1.\2<-?/' | tr '\n' ' ') | $(echo "$out" | grep -o 'Semaphore values.*' | sort -u | tr '\n' ' ')" >> $G/fuzz.txt
  rm -rf $G/fuzz-$off-$val/kernel.neff
done; done
bash sendrecv/gpx.sh fuzz-recover xchg_gpsimd 4 1 - "0 1" | grep -E '^RESULT' | tr '\n' ' ' >> $G/fuzz.txt; echo " <- stock control after fuzz" >> $G/fuzz.txt
echo FUZZDONE >> $G/fuzz.txt
