#!/bin/bash
# Run a bash script (stdin) on the work node. Usage: cluster/node.sh [node-ip] < script.sh
CTL=${TRN3_CTL:?set TRN3_CTL=user@control-node}
NODE=${1:-${TRN3_NODE0:?set TRN3_NODE0 or pass a node ip}}
ssh -o BatchMode=yes $CTL "ssh $NODE bash -s" 2> >(grep -v -E 'post-quantum|store now|upgraded' >&2)
