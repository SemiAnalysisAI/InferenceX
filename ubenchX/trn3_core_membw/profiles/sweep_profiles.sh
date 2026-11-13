#!/bin/bash
# On a work node, inside the trn3-ubench image (cluster/dk.sh). Compiles the chase once, then
#   solo:        every physical core 0..127 alone (LNC=1), one at a time
#   concurrent:  for each device, its 8 cores chasing at the same time (8 captures in parallel)
set -u
R=/w/runs/profiles
python profiles/run_profile.py compile --out $R/neff 2>&1 | tail -1
for c in $(seq 0 127); do
  [ -f $R/solo/c$c/result.json ] || python profiles/run_profile.py capture --neff-dir $R/neff --core $c --out $R/solo/c$c --num-exec 13 2>&1 | tail -1
done
for d in $(seq 0 15); do
  for k in $(seq 0 7); do
    c=$((8 * d + k))
    [ -f $R/conc/c$c/result.json ] || python profiles/run_profile.py capture --neff-dir $R/neff --core $c --out $R/conc/c$c --num-exec 3601 --profile-every 600 > $R/conc-c$c.log 2>&1 &
  done
  wait
  tail -qn1 $R/conc-c$((8 * d))*.log
done
echo SOLO+CONC DONE
# Repeat pass: the same NEFF and chain, a second capture per core. The repeat-vs-first difference is
# the run-to-run floor that cross-core differences have to clear.
for c in $(seq 0 127); do
  [ -f $R/solo2/c$c/result.json ] || python profiles/run_profile.py capture --neff-dir $R/neff --core $c --out $R/solo2/c$c --num-exec 13 2>&1 | tail -1
done
echo REPEAT DONE
