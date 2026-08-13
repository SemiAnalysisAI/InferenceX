#!/usr/bin/env bash
# Poll the DEP8 server /health until ready (cold 805 GiB NFS load ~2h + compile +
# cudagraph capture), then fire the 8k/1k c32 benchmark. Runs detached in the
# container; watch /home/jiacao/InferenceX/dsv4-waitbench.log.
set -uo pipefail
STAMP(){ date -u +%FT%TZ; }
WLOG=/home/jiacao/InferenceX/dsv4-waitbench.log
SLOG=/home/jiacao/InferenceX/dsv4-serve.log

echo "$(STAMP) waiting for http://localhost:8000/health ..." >> "$WLOG"
for i in $(seq 1 720); do            # 720 * 20s = 4h ceiling
    code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then
        echo "$(STAMP) server READY (health 200) after $((i*20))s" >> "$WLOG"
        break
    fi
    # bail early if the server process died
    if ! pgrep -f "vllm serve" >/dev/null 2>&1; then
        echo "$(STAMP) FATAL: vllm serve process gone; tail serve log:" >> "$WLOG"
        tail -40 "$SLOG" >> "$WLOG" 2>&1
        exit 1
    fi
    sleep 20
done

code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null || echo 000)
if [ "$code" != "200" ]; then
    echo "$(STAMP) FATAL: health never reached 200 (last=$code)" >> "$WLOG"
    exit 1
fi

echo "$(STAMP) launching 8k/1k c32 benchmark ..." >> "$WLOG"
bash /home/jiacao/InferenceX/.claude/worktrees/dsv4-ci-sweep-pr/dsv4_bench_8k1k_c32.sh
echo "$(STAMP) benchmark finished rc=$? ; results in dsv4-bench-8k1k-c32.json / .log" >> "$WLOG"
