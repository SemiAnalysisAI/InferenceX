// Resolve the shipped YAML with GitHub's expression engine; only event context and step status are fixtures.
import {readFileSync} from 'node:fs';
import {Parser, Lexer, Evaluator, data} from '@actions/expressions';
import {parse} from 'yaml';

export const workflow = path => parse(readFileSync(path, 'utf8'));
const cache = new Map();
export function evaluate(expression, context, status = 'success') {
  const names = ['always', 'success', 'failure', 'cancelled'];
  const ast = cache.get(expression) ?? new Parser(new Lexer(expression).lex().tokens, Object.keys(context), names.map(name => ({name, minArgs: 0, maxArgs: 0}))).parse();
  cache.set(expression, ast);
  const functions = new Map(names.map(name => [name, {
    name, minArgs: 0, maxArgs: 0, call: () => new data.BooleanData(name === 'always' || name === status)
  }]));
  return new Evaluator(ast, JSON.parse(JSON.stringify(context), data.reviver), functions).evaluate();
}
const plain = value => JSON.parse(JSON.stringify(value, data.replacer));
export function resolve(value, context, status) {
  if (typeof value !== 'string') return value;
  const whole = value.match(/^\$\{\{([\s\S]*?)\}\}$/);
  if (whole && !whole[1].includes('${{')) return plain(evaluate(whole[1], context, status));
  return value.replace(/\$\{\{([\s\S]*?)\}\}/g, (_, expression) => evaluate(expression, context, status).coerceString());
}
const string = value => JSON.parse(JSON.stringify(value ?? null), data.reviver).coerceString();
export function launch(caller, template, jobId, row, options = {}, status = 'success') {
  const context = {
    matrix: {config: row},
    inputs: {'klaud-run': false, 'ref': '', 'agentx-fast': false, 'duration-override': '',
      'require-power': false, 'eval-limit': '', 'eval-framework': 'auto', 'eval-suite': '',
      'swebench-gen-mode': '', ...options},
    github: {repository: 'test/repo', event_name: 'pull_request', head_ref: 'feature',
      sha: 'measured-commit', workflow_sha: 'tooling-commit', run_attempt: 2,
      workspace: '/tmp/workspace', event: {pull_request: {head: {repo: {full_name: 'test/repo'}},
        user: {login: 'contributor'}, labels: (options.labels ?? []).map(name => ({name}))}}},
    secrets: {INFERENCEX_OFFICIAL_RO_HF_TOKEN: 'fake-hf-token', MODAL_TOKEN_ID: 'fake-id',
      MODAL_TOKEN_SECRET: 'fake-modal-token', REPO_PAT: 'fake-github-token'},
    vars: {PRIORITY_SCHEDULER_ENABLED: 'true', NODE_SLOT_SCHEDULER_ENABLED: 'true', ...(options.vars ?? {})},
    runner: {name: 'fixture-node_03'}, env: {}
  };
  if (options.klaud) {
    context.github.head_ref = 'klaud/auto-fixture';
    context.github.event.pull_request.user.login = 'Klaud-Cold';
  }
  const passed = caller.jobs[jobId].with;
  const inputs = {};
  for (const [name, rule] of Object.entries(template.on.workflow_call.inputs)) {
    if (rule.required && !(name in passed)) throw new Error(`Missing required input: ${name}`);
    let value = name in passed ? resolve(passed[name], context) : (rule.default ?? (rule.type === 'boolean' ? false : ''));
    if (rule.type === 'string') value = string(value);
    if (rule.type === 'boolean' && typeof value !== 'boolean') throw new Error(`Invalid boolean input: ${name}`);
    inputs[name] = value;
  }
  for (const name of Object.keys(passed)) if (!(name in inputs)) throw new Error(`Unknown input ${name}`);
  context.inputs = inputs;
  const env = Object.fromEntries(Object.entries(template.env).map(([k,v]) => [k,string(resolve(v,context,status))]));
  context.env = {...env, RESULT_FILENAME: options.resultFilename ?? 'result-hash', GPU_COUNT: '8'};
  function visit(value, key) {
    if (Array.isArray(value)) return value.map(v => visit(v));
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k,v]) => [k,visit(v,k)]));
    if (key === 'if' && typeof value === 'string' && !value.includes('${{')) return plain(evaluate(value,context,status));
    return resolve(value, context, status);
  }
  return {env, job: visit(template.jobs.benchmark)};
}
