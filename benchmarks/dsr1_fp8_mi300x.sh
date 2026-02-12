#!/usr/bin/env bash

host_name="$(hostname)"
echo "$host_name"

if [[ "$host_name" == "chi-mi300x-049"* ]]; then
    sleep 86400
else
    exit 1
fi
