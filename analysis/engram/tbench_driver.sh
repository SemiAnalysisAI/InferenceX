#!/usr/bin/env bash
# Terminal-Bench 4.0 feasibility, phase 2: can Modal work, and can a tunnel
# make our endpoint reachable from it?
#
# Phase 1 established that the execution environment has no container runtime
# (docker/podman/enroot all absent), so Harbor's local backend is out and
# --env modal is the remaining on-cluster-model option.
#
# Two directions, which are not the same thing and were conflated earlier:
#
#   EGRESS  compute node -> Modal. Needed for Harbor to submit jobs. Expected
#           to work: these nodes already pull from HuggingFace and PyPI.
#   INGRESS Modal -> our vLLM. Needed because with --env modal the agent runs
#           inside the remote task container and calls the model over HTTP.
#           Direct ingress is impossible: hpc-gpu-1-N has no public address.
#
# A tunnel converts the ingress problem into an egress one -- the node dials
# out and Cloudflare hands back a public URL. This proves the round trip with
# a trivial HTTP server, so no GPU time is spent on plumbing that may not work.
set -eo pipefail

source "$(dirname "$0")/../../benchmarks/benchmark_lib.sh"
check_env_vars RESULT_DIR
mkdir -p "$RESULT_DIR"
REPORT="$RESULT_DIR/tbench_tunnel_probe.txt"
say() { echo "$@" | tee -a "$REPORT"; }

say "=== direction 1: EGRESS from the compute node ==="
for url in https://api.modal.com/ https://modal.com/ https://hub.harborframework.com/ \
           https://github.com/ https://pypi.org/simple/; do
    code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' "$url" 2>&1 || echo "FAIL")
    say "  $code  $url"
done

say "=== modal client auth (needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET) ==="
if [[ -n "${MODAL_TOKEN_ID:-}" && -n "${MODAL_TOKEN_SECRET:-}" ]]; then
    say "  token env present (id length ${#MODAL_TOKEN_ID})"
    python3 -m pip install -q --no-input --break-system-packages modal 2>&1 | tail -2 || true
    # Confirms the credentials actually authenticate, rather than just existing.
    (timeout 90 python3 -m modal profile current 2>&1 || true) | tee -a "$REPORT" | head -5
    (timeout 90 python3 -m modal app list 2>&1 || true) | tee -a "$REPORT" | head -8
else
    say "  MODAL_TOKEN_ID/SECRET not set -- cannot test auth."
    say "  Connectivity above is still the useful part: it says whether the"
    say "  token would be usable from here once it is wired in."
fi

say "=== direction 2: INGRESS via a Cloudflare quick tunnel ==="
BIN="$RESULT_DIR/cloudflared"
if [[ ! -x "$BIN" ]]; then
    curl -sSL -m 120 -o "$BIN" \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        && chmod +x "$BIN" || say "  cloudflared download FAILED"
fi
if [[ -x "$BIN" ]]; then
    say "  cloudflared: $("$BIN" --version 2>&1 | head -1)"
    # A canary served locally; if it comes back through Cloudflare's edge then
    # an inbound request from Modal would reach a vLLM server the same way.
    CANARY="engram-tunnel-canary-$RANDOM$RANDOM"
    WEBROOT=$(mktemp -d)
    echo "$CANARY" > "$WEBROOT/canary.txt"
    (cd "$WEBROOT" && python3 -m http.server 8899 >/dev/null 2>&1 &)
    sleep 3
    "$BIN" tunnel --no-autoupdate --url http://localhost:8899 \
        > "$RESULT_DIR/cloudflared.log" 2>&1 &
    TUNNEL_PID=$!
    PUBLIC=""
    for _ in $(seq 1 40); do
        PUBLIC=$(grep -aoE 'https://[a-z0-9-]+\.trycloudflare\.com' "$RESULT_DIR/cloudflared.log" \
                 | head -1 || true)
        [[ -n "$PUBLIC" ]] && break
        sleep 3
    done
    if [[ -n "$PUBLIC" ]]; then
        say "  public URL: $PUBLIC"
        GOT=$(curl -sS -m 45 "$PUBLIC/canary.txt" 2>&1 | tr -d '\r\n' || echo FAIL)
        if [[ "$GOT" == "$CANARY" ]]; then
            say "  ROUND TRIP OK -- the canary came back through Cloudflare's edge"
            say "  TUNNEL_WORKS=1"
        else
            say "  round trip FAILED (got: ${GOT:0:120})"
            say "  TUNNEL_WORKS=0"
        fi
    else
        say "  no trycloudflare URL appeared; last log lines:"
        tail -15 "$RESULT_DIR/cloudflared.log" | tee -a "$REPORT" || true
        say "  TUNNEL_WORKS=0"
    fi
    kill "$TUNNEL_PID" 2>/dev/null || true
    pkill -f "http.server 8899" 2>/dev/null || true
fi

say "=== verdict ==="
say "If EGRESS is 2xx/3xx and TUNNEL_WORKS=1, then --env modal is viable:"
say "Harbor submits to Modal from here, and the remote agent reaches vLLM"
say "through the tunnel. The real run must then serve vLLM with --api-key,"
say "because a quick tunnel is a public unauthenticated URL and the endpoint"
say "would otherwise be open to anyone for the duration of the run."
