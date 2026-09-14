import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import {launch, workflow} from './benchmark-runtime.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const template = workflow(join(root, '.github/workflows/benchmark-tmpl.yml'));
const callers = Object.fromEntries(['run-sweep', 'e2e-tests'].map(name =>
  [name, workflow(join(root, `.github/workflows/${name}.yml`))]));
const fixed = {
  image: 'engine:fixture', model: 'test/model', 'model-prefix': 'fixture', framework: 'dynamo-sglang',
  precision: 'fp8', runner: 'fixture-node', priority: 42, 'queue-token': 'test-token',
  'exp-name': 'fixture', isl: 8192, osl: 1024, 'max-model-len': 16384,
  tp: 4, pp: 2, 'dcp-size': 2, 'pcp-size': 2, ep: 8, 'dp-attn': true,
  conc: 16, 'spec-decoding': 'mtp', disagg: false, 'run-eval': false,
  router: {name: 'route', version: 'v2'}, 'require-power': true,
};
const agentic = {...fixed, 'scenario-type': 'agentic-coding', 'kv-offloading': 'dram',
  'kv-offload-backend': {name: 'storage', version: 'v3'}, 'total-cpu-dram-gb': 512, duration: 7200};
const run = (file, job, row = fixed, options = {}, status) =>
  launch(callers[file], template, job, row, options, status);
const step = (result, name) => result.job.steps.find(item => item.name === name);

// Expected execution modes are independent of the caller YAML being exercised.
const cases = [
  ['run-sweep', 'canary-sweep', false, false, false],
  ['run-sweep', 'sweep-single-node-1k1k', false, false, false],
  ['run-sweep', 'sweep-single-node-8k1k', false, false, false],
  ['run-sweep', 'sweep-evals', false, true, true],
  ['run-sweep', 'sweep-agentic', true, false, false],
  ['run-sweep', 'sweep-agentic-evals', true, true, true],
  ['e2e-tests', 'test-sweep-single-node', false, false, false],
  ['e2e-tests', 'test-sweep-evals', false, true, true],
  ['e2e-tests', 'test-sweep-agentic', true, false, false],
  ['e2e-tests', 'test-sweep-agentic-evals', true, true, true],
];
for (const [file, job, isAgentic, runEval, evalOnly] of cases) {
  test(`${job}: configuration reaches the launcher and artifacts in the intended mode`, () => {
    const result = run(file, job, isAgentic ? agentic : fixed);
    for (const [key, expected] of Object.entries({
      MODEL: 'test/model', MODEL_PREFIX: 'fixture', IMAGE: 'engine:fixture', PRECISION: 'fp8',
      FRAMEWORK: 'dynamo-sglang', EXP_NAME: 'fixture',
      TP: '4', PP_SIZE: '2', DCP_SIZE: '2', PCP_SIZE: '2', EP_SIZE: '8', DP_ATTENTION: 'true',
      CONC: '16', DISAGG: 'false', SPEC_DECODING: 'mtp',
      RUN_EVAL: String(runEval), EVAL_ONLY: String(evalOnly),
      SCENARIO_TYPE: isAgentic ? 'agentic-coding' : 'fixed-seq-len', IS_AGENTIC: isAgentic ? '1' : '0',
      SCENARIO_SUBDIR: isAgentic ? 'agentic/' : 'fixed_seq_len/',
      ISL: isAgentic ? '0' : '8192', OSL: isAgentic ? '0' : '1024', MAX_MODEL_LEN: isAgentic ? '0' : '16384',
      DURATION: isAgentic ? '7200' : '3600', TOTAL_CPU_DRAM_GB: isAgentic ? '512' : '0',
      KV_OFFLOADING: isAgentic ? 'dram' : '', KV_OFFLOAD_BACKEND: isAgentic ? 'storage' : '',
      EVAL_FRAMEWORK: 'lm-eval', EVAL_SUITE: '', REQUIRE_POWER: !isAgentic && !evalOnly ? '1' : '0',
    })) assert.equal(result.env[key], expected, key);
    assert.deepEqual(JSON.parse(result.env.ROUTER_METADATA), {name: 'route', version: 'v2'});
    assert.equal(result.env.KV_OFFLOAD_BACKEND_METADATA,
      isAgentic ? '{\n  "name": "storage",\n  "version": "v3"\n}' : '');
    assert.match(result.job.name, /p42 \| fixture fp8 fixture-node dyn-sgl TP4\/PP2\/DCP2\/PCP2\/EP8\/DPA mtp/);
    assert.equal(result.job.name.endsWith(' | eval-only'), evalOnly);
    assert.equal(step(result, 'Upload result').if, !isAgentic && !evalOnly);
    assert.equal(step(result, 'Upload agentic aggregated result').if, isAgentic);
    assert.equal(step(result, 'Upload eval results (if any)').if, runEval || evalOnly);
    assert.equal(step(result, 'Upload server logs').with.name, `${evalOnly ? 'eval_' : ''}server_logs_result-hash`);
  });
}

test('missing historical fields stay empty; zero and false are not replaced by defaults', () => {
  const result = run('run-sweep', 'sweep-agentic', {...agentic,
    pp: undefined, 'dcp-size': undefined, 'pcp-size': undefined, 'dp-attn': false,
    'kv-offload-backend': null, 'total-cpu-dram-gb': 0, 'recipe-fingerprint': null, router: null});
  assert.equal(result.env.PP_SIZE, '');
  assert.equal(result.env.DCP_SIZE, '');
  assert.equal(result.env.PCP_SIZE, '');
  assert.equal(result.env.DP_ATTENTION, 'false');
  assert.equal(result.env.TOTAL_CPU_DRAM_GB, '0');
  assert.equal(result.env.KV_OFFLOAD_BACKEND_METADATA, '');
  assert.equal(result.env.ROUTER_METADATA, '');
  assert.equal(result.env.RECIPE_FINGERPRINT, '');
  assert.doesNotMatch(result.job.name, /\/PP|\/DCP|\/PCP|\/DPA|null|undefined/);
});

test('sweep throughput honors run-eval, while the canary and manual throughput suppress it', () => {
  const row = {...fixed, 'run-eval': true};
  assert.equal(run('run-sweep', 'sweep-single-node-1k1k', row).env.RUN_EVAL, 'true');
  assert.equal(run('run-sweep', 'canary-sweep', row).env.RUN_EVAL, 'false');
  assert.equal(run('e2e-tests', 'test-sweep-single-node', row).env.RUN_EVAL, 'false');
});

test('parallelism names distinguish missing, zero, and string values', () => {
  for (const [pp, suffix] of [[0, '/PP0'], ['01', '/PP01'], [2, '/PP2']]) {
    const result = run('run-sweep', 'canary-sweep', {...fixed, pp});
    assert.ok(result.job.name.includes(suffix), result.job.name);
    assert.equal(result.env.PP_SIZE, String(pp));
  }
});

test('non-boolean dp-attn is rejected at the workflow input boundary', () => {
  for (const value of ['false', 0, null, {}]) {
    assert.throws(() => run('run-sweep', 'canary-sweep', {...fixed, 'dp-attn': value}),
      /Invalid boolean input: dp-attn/);
  }
});

test('manual overrides retain their precedence over recipe defaults', () => {
  const row = {...agentic, 'eval-framework': 'bfcl', 'eval-suite': 'bfcl_smoke'};
  const automatic = run('e2e-tests', 'test-sweep-agentic-evals', row).env;
  assert.equal(automatic.EVAL_FRAMEWORK, 'bfcl');
  assert.equal(automatic.EVAL_SUITE, 'bfcl_smoke');
  const explicit = run('e2e-tests', 'test-sweep-agentic-evals', row, {
    'eval-framework': 'swebench', 'eval-limit': '7', 'swebench-gen-mode': 'single-shot',
    'duration-override': '75', 'ref': 'older-measured-commit'});
  assert.equal(explicit.env.EVAL_FRAMEWORK, 'swebench');
  assert.equal(explicit.env.EVAL_SUITE, '');
  assert.equal(explicit.env.EVAL_LIMIT, '7');
  assert.equal(explicit.env.SWEBENCH_GEN_MODE, 'single-shot');
  assert.equal(explicit.env.DURATION, '75');
  assert.equal(explicit.job.steps.find(item => item.uses?.startsWith('actions/checkout@')).with.ref, 'older-measured-commit');
  assert.equal(run('e2e-tests', 'test-sweep-agentic-evals', row, {'eval-suite': 'explicit'}).env.EVAL_SUITE, 'explicit');
  assert.equal(run('e2e-tests', 'test-sweep-agentic', row, {'duration-override': '0'}).env.DURATION, '0');
});

test('AgentX fast mode preserves the distinct sweep and manual duration behavior', () => {
  for (const job of ['test-sweep-agentic', 'test-sweep-agentic-evals']) {
    const result = run('e2e-tests', job, agentic, {'agentx-fast': true, 'duration-override': '75'});
    assert.equal(result.env.DURATION, '1200');
    assert.equal(result.env.AIPERF_EXPERIMENTAL_FAST, '1');
  }
  const sweep = run('run-sweep', 'sweep-agentic', agentic, {labels: ['agentx-fast']});
  assert.equal(sweep.env.DURATION, '7200');
  assert.equal(sweep.env.AIPERF_EXPERIMENTAL_FAST, '1');
});

test('scheduling preserves node slots, queue identity, skip requests, and Klaud labels', () => {
  const result = run('run-sweep', 'sweep-single-node-1k1k', {...fixed, 'skip-queue-pr': '42'}, {klaud: true});
  assert.deepEqual(result.job['runs-on'], ['self-hosted', 'fixture-node', 'nodes:1',
    'ci-job-42-test-token', 'ci-attempt-2', 'ci-skip-queue-pr-42']);
  assert.match(result.job.name, /^klaud \| /);
  assert.deepEqual(run('run-sweep', 'canary-sweep', fixed, {
    vars: {PRIORITY_SCHEDULER_ENABLED: 'false'}}).job['runs-on'], ['fixture-node']);
});

test('failed and cancelled jobs retain diagnostic uploads and cleanup without publishing throughput', () => {
  for (const status of ['failure', 'cancelled']) {
    const result = run('run-sweep', 'sweep-single-node-1k1k', fixed, {}, status);
    assert.equal(step(result, 'Upload result').if, false);
    assert.equal(step(result, 'Process result').if, true);
    assert.equal(step(result, 'Upload server logs').if, true);
    assert.equal(step(result, 'Resource cleanup (post-run)').if, true);
  }
  const noOutput = run('run-sweep', 'sweep-single-node-1k1k', fixed, {resultFilename: ''}, 'failure');
  assert.equal(step(noOutput, 'Process result').if, false);
});

for (const historical of [false, true]) {
  test(`the real launch step preserves arguments and filenames (${historical ? 'historical checkout' : 'current helper'})`, t => {
    const cwd = mkdtempSync(join(tmpdir(), 'infx-workflow-'));
    t.after(() => rmSync(cwd, {recursive: true, force: true}));
    mkdirSync(join(cwd, 'runners'));
    if (!historical) {
      mkdirSync(join(cwd, 'utils'));
      mkdirSync(join(cwd, 'infx/results'), {recursive: true});
      for (const file of ['utils/result_filename.py', 'infx/__init__.py', 'infx/results/__init__.py',
        'infx/results/result_filename.py']) copyFileSync(join(root, file), join(cwd, file));
    }
    writeFileSync(join(cwd, 'runners/launch_fixture-node.sh'), `#!/bin/bash
python3 - <<'PY'
import json, os
with open('received.json', 'w') as output:
    json.dump(dict(os.environ), output)
PY
printf '{}' > "$RESULT_FILENAME.json"
`);
    const model = 'model "quoted"\n$(touch injected)';
    const runtime = run('e2e-tests', 'test-sweep-single-node', {...fixed, model});
    const launchStep = step(runtime, 'Launch job script');
    const output = spawnSync('bash', ['-euo', 'pipefail', '-c', launchStep.run], {
      cwd, encoding: 'utf8', timeout: 10000,
      env: {...process.env, ...runtime.env, ...launchStep.env, GITHUB_ENV: join(cwd, 'github-env')},
    });
    assert.equal(output.status, 0, output.stderr);
    const received = JSON.parse(readFileSync(join(cwd, 'received.json'), 'utf8'));
    assert.equal(received.MODEL, model);
    assert.equal(received.GPU_COUNT, '16'); // TP4 * PP2 * PCP2; DCP does not multiply GPUs.
    assert.equal(received.CONC, '16');
    assert.equal(received.RUNNER_TYPE, 'fixture-node');
    const base = 'fixture_fp8_dynamo-sglang_tp4-pp2-dcp2-pcp2-ep8-dpatrue_disagg-false_spec-mtp_conc16_fixture-node_03';
    const expected = historical ? createHash('sha256').update(base + '\0\0').digest('hex') : base;
    assert.equal(received.RESULT_FILENAME, expected);
    assert.deepEqual(JSON.parse(readFileSync(join(cwd, `${expected}.json`), 'utf8')), {});
    assert.match(readFileSync(join(cwd, 'github-env'), 'utf8'), /^GPU_COUNT=16\nRESULT_FILENAME=/);
    assert.throws(() => readFileSync(join(cwd, 'injected')), {code: 'ENOENT'});
  });
}
