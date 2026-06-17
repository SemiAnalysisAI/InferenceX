#!/usr/bin/env bash

# Isolated TP4+PP2+EP4 experiment for MiniMax-M3 on one eight-GPU MI300X node.
export PP_SIZE=2
exec "$(dirname "$0")/minimaxm3_fp8_mi300x.sh"
