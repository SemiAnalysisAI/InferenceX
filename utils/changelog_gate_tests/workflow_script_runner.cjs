// Execute the scripts read from workflow YAML. Only external collaborators are faked.
const fs = require('node:fs');
const {spawnSync} = require('node:child_process');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const output = {writes: [], outputs: {}, failures: [], warnings: []};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const baseEnv = {...process.env};
Date.now = () => Date.parse('2026-01-02T12:00:00Z');

function response(method, args) {
  const data = input.data;
  if (method === 'repos.getCollaboratorPermissionLevel') {
    if (input.permissionError) throw new Error('permission lookup unavailable');
    return input.permission;
  }
  if (method === 'pulls.get') return data.pull;
  if (method === 'pulls.listCommits') return data.commits;
  if (method === 'issues.listEventsForTimeline') return data.timeline;
  if (method === 'actions.getWorkflowRun') return data.runs.find(run => run.id === args.run_id);
  if (method === 'actions.listWorkflowRunArtifacts') return data.artifacts[String(args.run_id)] ?? [];
  if (method === 'actions.listWorkflowRuns') {
    return args.event === 'workflow_dispatch' ? {workflow_runs: data.dispatchedRuns} : data.runs;
  }
  if (['issues.createComment', 'actions.createWorkflowDispatch', 'repos.createDispatchEvent'].includes(method)) {
    output.writes.push({method, ...args});
    return {id: 501};
  }
  throw new Error(`Unexpected GitHub call: ${method}`);
}

const rest = new Proxy({}, {get: (_, area) => new Proxy({}, {
  get: (_, name) => async args => ({data: response(`${area}.${name}`, args)}),
})});
const github = {rest, paginate: async (fn, args) => (await fn(args)).data};
const exec = {getExecOutput: async (command, args, options = {}) => {
  if (command !== 'python3') throw new Error(`Unexpected command: ${command}`);
  const result = spawnSync(process.env.TEST_PYTHON, args, {
    encoding: 'utf8', env: process.env, timeout: 20000,
  });
  if (result.error) throw result.error;
  if (result.status && !options.ignoreReturnCode) throw new Error(result.stderr);
  return {stdout: result.stdout, stderr: result.stderr, exitCode: result.status};
}};

function resolve(value) {
  return String(value).replace(/\$\{\{\s*(.*?)\s*\}\}/g, (_, path) => {
    const vars = {
      github: {...input.context, token: 'workflow-token', event: input.context.payload},
      steps: Object.fromEntries(Object.entries(output.outputs).map(([id, outputs]) => [id, {outputs}])),
    };
    return path.split('.').reduce((obj, key) => obj?.[key], vars) ?? '';
  });
}

(async () => {
  for (const step of input.steps) {
    if (output.failures.length) break; // Actions' default success() step condition.
    process.env = {...baseEnv, ...Object.fromEntries(
      Object.entries(step.env ?? {}).map(([key, value]) => [key, resolve(value)]),
    )};
    const core = {
      setFailed: message => output.failures.push(message),
      warning: message => output.warnings.push(message),
      setOutput: (key, value) => {
        (output.outputs[step.id ?? step.name] ??= {})[key] = value;
      },
    };
    try {
      if (step.with?.script) {
        await new AsyncFunction('github', 'context', 'core', 'exec', 'setTimeout', step.with.script)(
          github, input.context, core, exec, callback => callback(),
        );
      } else if (!step.uses?.startsWith('actions/checkout@')) {
        throw new Error(`Unsupported step: ${step.name}`);
      }
    } catch (error) {
      core.setFailed(error.message);
    }
  }
  process.stdout.write(JSON.stringify(output));
})();
