#!/usr/bin/env bash
# Terminal-Bench 4.0 for real: vLLM behind a Cloudflare tunnel, Harbor on Modal.
#
# Established by the earlier probes:
#   - no container runtime on the compute node OR the login node, so Harbor's
#     local backend is out and --env modal is the only route with our own model
#   - egress to api.modal.com is 200 and MODAL_TOKEN_ID/SECRET (already wired in
#     benchmark-tmpl.yml for SWE-bench) authenticate: `modal app list` works
#   - a quick tunnel comes up, but the node cannot resolve *.trycloudflare.com,
#     so it cannot self-verify. Modal resolves from its own network, so the
#     only test that settles it is the real run -- this one.
#
# The endpoint is public for the duration, so it is served WITH an api key and
# the key reaches the agent only through harbor's --env-file.
set -eo pipefail

source "$(dirname "$0")/../../benchmarks/benchmark_lib.sh"
check_env_vars MODEL TP RESULT_DIR
export GPU_COUNT="$TP"
mkdir -p "$RESULT_DIR"
HELP_DIR="$RESULT_DIR/harbor_help"; mkdir -p "$HELP_DIR"
say() { echo "$@" | tee -a "$RESULT_DIR/tbench_run.txt"; }

if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"; export MODEL_PATH="$MODEL"
fi

export PATH="$HOME/.local/bin:$PATH"
# `harbor` alone is not enough: the run reached "0/66 Running trials" and then
# died on ModuleNotFoundError: dockerfile_parse. The README installs the modal
# extra ('harbor[modal]'), which carries the task-environment dependencies, so
# install that and name dockerfile-parse explicitly as a belt-and-braces.
python3 -m pip install --no-input --break-system-packages     'harbor[modal]' modal dockerfile-parse 2>&1 | tail -15 || true
python3 - <<'PYCHK'
import importlib.util  # `import importlib` alone does not bind .util

missing = [m for m in ("harbor", "modal", "dockerfile_parse")
           if importlib.util.find_spec(m) is None]
print("MISSING MODULES:", missing or "none")
raise SystemExit(1 if missing else 0)
PYCHK
if [[ $? != 0 ]]; then
    echo "FATAL: harbor dependencies incomplete; see the pip output above" >&2
    exit 1
fi
HARBOR=(harbor); command -v harbor >/dev/null 2>&1 || HARBOR=(python3 -m harbor)

# Capture help in full. The previous attempt piped to head, and SIGPIPE
# truncated the agent list mid-word (exit 141).
say "=== harbor help (full, in $HELP_DIR) ==="
export COLUMNS=200
for sub in "" "run" "job" "job start"; do
    # shellcheck disable=SC2086
    "${HARBOR[@]}" $sub --help > "$HELP_DIR/help_${sub// /_}.txt" 2>&1 || true
done
AGENTS=$(sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$HELP_DIR/help_run.txt" \
         | tr -d '\n' | grep -oE '\-\-agent +-a +\[[^]]*\]' | head -1 || true)
say "agent choices: ${AGENTS:-<not parsed; see help_run.txt>}"
say "task/dataset flags:"
sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$HELP_DIR/help_run.txt" | grep -iE '^\s*-.*(task|dataset)' \
    | tee -a "$RESULT_DIR/tbench_run.txt" || true

# Terminus is Terminal-Bench's own agent and the right default for a
# LiteLLM-addressable endpoint; fall back to whatever this build offers.
AGENT=""
for candidate in terminus-2 terminus codex-cli aider; do
    if grep -q "$candidate" "$HELP_DIR/help_run.txt" 2>/dev/null; then AGENT="$candidate"; break; fi
done
if [[ -z "$AGENT" ]]; then
    say "FATAL: no known agent in this harbor build; see $HELP_DIR/help_run.txt"
    exit 1
fi
say "using agent: $AGENT"

API_KEY="sk-engram-$(head -c 18 /dev/urandom | od -An -tx1 | tr -d ' \n')"
EVAL_CONTEXT=65536
# Indexer buffer is batched-tokens x max-model-len x 2B: 4096 x 65536 x 2 =
# 0.5 GiB, unlike the 16 GiB that 1M context would cost on an 80 GB card.
pick_port() {
    local c; for c in $(seq 8890 8960); do
        (exec 3<>"/dev/tcp/127.0.0.1/$c") 2>/dev/null || { PORT="$c"; export PORT; return 0; }
    done; return 1
}
pick_port
SERVER_LOG="$RESULT_DIR/server_tbench.log"
export VLLM_ENGINE_READY_TIMEOUT_S=3600 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve "$MODEL_PATH" --served-model-name "$MODEL" \
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP" \
    --api-key "$API_KEY" \
    --language-model-only --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --engram-config '{"cpu_offload":true}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
    --max-model-len "$EVAL_CONTEXT" --max-num-batched-tokens 4096 \
    --max-num-seqs 16 --gpu-memory-utilization 0.92 \
    --max-cudagraph-capture-size 128 \
    --disable-uvicorn-access-log > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

BIN="$RESULT_DIR/cloudflared"
[[ -x "$BIN" ]] || { curl -sSL -m 120 -o "$BIN" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x "$BIN"; }

cleanup() {
    say "--- cleanup: tearing down tunnel and server"
    [[ -n "${TUNNEL_PID:-}" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
say "--- local endpoint check (authenticated)"
curl -sS -m 60 -H "Authorization: Bearer $API_KEY" "http://localhost:$PORT/v1/models" \
    | head -c 300 | tee -a "$RESULT_DIR/tbench_run.txt"; echo

say "--- chat-completion canary: assistant content must be non-empty"
CANARY_JSON=$(curl -sS -m 300 -H "Authorization: Bearer $API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with exactly: READY"}],"max_tokens":2048,"temperature":0}' \
    "http://localhost:$PORT/v1/chat/completions" 2>&1)
CANARY_TEXT=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
except Exception as exc:
    print('VERDICT=FAIL parse_error=%r' % (exc,)); raise SystemExit
ch = (d.get('choices') or [{}])[0]
msg = ch.get('message') or {}
content = (msg.get('content') or '').strip()
reasoning = (msg.get('reasoning_content') or '').strip()
# The answer must be in content, short, and not the model thinking out loud.
ok = bool(content) and 'READY' in content.upper() and len(content) < 200
print('VERDICT=%s content=%r len=%d reasoning_len=%d finish=%r' % (
    'PASS' if ok else 'FAIL', content[:160], len(content), len(reasoning),
    ch.get('finish_reason')))
" <<<"$CANARY_JSON")
say "  $CANARY_TEXT"
if [[ "$CANARY_TEXT" != VERDICT=PASS* ]]; then
    say "FATAL: message.content is not a usable answer. terminus-2 parses"
    say "commands out of content, so every agent step would fail. Empty content"
    say "means a parser is diverting the answer (tool_calls); long prose means"
    say "the chain-of-thought is leaking in and needs --reasoning-parser."
    exit 1
fi

# Quick tunnels are account-less and cloudflared says so itself: "no uptime
# guarantee". Creation has already failed once with a client timeout against
# api.trycloudflare.com, after four tunnels from this IP. Retry with backoff,
# then fall back to localhost.run, which also needs no account (over ssh).
open_tunnel() {
    local attempt
    for attempt in 1 2 3; do
        say "  cloudflared attempt $attempt"
        "$BIN" tunnel --no-autoupdate --url "http://localhost:$PORT" \
            > "$RESULT_DIR/cloudflared.log" 2>&1 &
        TUNNEL_PID=$!
        for _ in $(seq 1 25); do
            PUBLIC=$(grep -aoE 'https://[a-z0-9]+(-[a-z0-9]+){2,}\.trycloudflare\.com' \
                     "$RESULT_DIR/cloudflared.log" | head -1 || true)
            [[ -n "$PUBLIC" ]] && return 0
            sleep 3
        done
        say "  attempt $attempt failed: $(tail -2 "$RESULT_DIR/cloudflared.log" | tr '\n' ' ')"
        kill "$TUNNEL_PID" 2>/dev/null || true
        sleep 20
    done

    if command -v ssh >/dev/null 2>&1; then
        say "  falling back to localhost.run (no account needed)"
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o ServerAliveInterval=30 -R "80:localhost:$PORT" nokey@localhost.run \
            > "$RESULT_DIR/localhostrun.log" 2>&1 &
        TUNNEL_PID=$!
        for _ in $(seq 1 20); do
            PUBLIC=$(grep -aoE 'https://[a-z0-9]+\.lhr\.life|https://[a-z0-9-]+\.localhost\.run' \
                     "$RESULT_DIR/localhostrun.log" | head -1 || true)
            [[ -n "$PUBLIC" ]] && return 0
            sleep 3
        done
        say "  localhost.run failed: $(tail -3 "$RESULT_DIR/localhostrun.log" | tr '\n' ' ')"
        kill "$TUNNEL_PID" 2>/dev/null || true
    else
        say "  no ssh client for the localhost.run fallback"
    fi
    return 1
}

PUBLIC=""
if ! open_tunnel; then
    say "FATAL: could not open any tunnel. Every other part of the path is"
    say "proven -- egress 200, Modal authenticates, and an earlier run had"
    say "Modal-side agents driving real task containers through a tunnel."
    say "This needs a named Cloudflare tunnel (account token) or --env daytona."
    exit 1
fi
if [[ "$PUBLIC" == https://api.trycloudflare.com* ]]; then
    say "FATAL: refusing cloudflared's control endpoint as origin"
    exit 1
fi
say "tunnel: $PUBLIC  (api key withheld from this log)"

MODEL_INFO=$(python3 -c "
import json
ctx = $EVAL_CONTEXT
print(json.dumps({
    'max_input_tokens': ctx - 8192,
    'max_output_tokens': 8192,
    'max_tokens': ctx,
    'input_cost_per_token': 0,
    'output_cost_per_token': 0,
    'litellm_provider': 'openai',
    'mode': 'chat',
}))")
say "model_info for terminus-2: $MODEL_INFO"

ENV_FILE="$RESULT_DIR/harbor.env"
umask 077
cat > "$ENV_FILE" <<ENV
OPENAI_API_KEY=$API_KEY
OPENAI_BASE_URL=$PUBLIC/v1
OPENAI_API_BASE=$PUBLIC/v1
ENV
say "wrote $ENV_FILE (mode $(stat -c %a "$ENV_FILE" 2>/dev/null || echo '?'))"

# -k 1 and aggressive multipliers bound both wall-clock and Modal spend: the
# published default is an 8-hour agent timeout per task.
say "=== harbor run (env modal) ==="
set +e
timeout "${TBENCH_TIMEOUT_S:-16200}" "${HARBOR[@]}" run \
    -d terminal-bench/terminal-bench@4.0.0 \
    --agent "$AGENT" \
    --model "openai/$MODEL" \
    --env-file "$ENV_FILE" \
    --env modal \
    --ak "model_info=$MODEL_INFO" \
    -k "${TBENCH_ATTEMPTS:-1}" \
    --n-concurrent "${TBENCH_CONCURRENT:-8}" \
    --timeout-multiplier "${TBENCH_TIMEOUT_MULT:-0.1}" \
    --agent-timeout-multiplier "${TBENCH_TIMEOUT_MULT:-0.1}" \
    --job-name "engram-tbench-$(date +%s)" \
    --jobs-dir "$RESULT_DIR/harbor_jobs" \
    --yes 2>&1 | tee -a "$RESULT_DIR/tbench_run.txt"
HARBOR_RC=${PIPESTATUS[0]}
set -e
say "harbor exit=$HARBOR_RC"

say "=== cloudflare origin timeouts (524) seen during the run ==="
say "count: $(grep -ac 'error_code.: 524' "$RESULT_DIR/tbench_run.txt" || echo 0)"
say "A nonzero count means the 120s quick-tunnel read timeout is still cutting"
say "requests short, and any score below is a floor rather than a measurement."

say "=== results ==="
find "$RESULT_DIR/harbor_jobs" -name '*.json' | head -20 | tee -a "$RESULT_DIR/tbench_run.txt" || true
python3 - <<'PYEOF' 2>&1 | tee -a "$RESULT_DIR/tbench_run.txt" || true
import glob, json, os
root = os.path.join(os.environ["RESULT_DIR"], "harbor_jobs")
for path in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
    if os.path.getsize(path) > 2_000_000:
        continue
    try:
        data = json.load(open(path))
    except Exception:
        continue
    if isinstance(data, dict) and any(k in data for k in ("results", "accuracy", "resolved", "n_resolved")):
        print("===", path)
        print(json.dumps(data, indent=2)[:3000])
PYEOF
exit "$HARBOR_RC"
