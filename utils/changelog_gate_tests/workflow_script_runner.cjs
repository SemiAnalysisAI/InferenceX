// Execute the scripts read from workflow YAML. Only external collaborators are faked.
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const output = {permissionRequests: [], requests: [], writes: [], outputs: {}, failures: [], warnings: [], shell: []};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const baseEnv = {...process.env};
Date.now = () => Date.parse('2026-01-02T12:00:00Z');

function response(method, args) {
  output.requests.push({method, ...args});
  if (input.failMethod === method) throw Object.assign(new Error('GitHub unavailable'), {status: input.errorStatus});
  const data = input.data;
  if (method === 'repos.getCollaboratorPermissionLevel') {
    output.permissionRequests.push(args);
    if (input.permissionError) throw new Error('permission lookup unavailable');
    return Object.hasOwn(input.permissionsByUser ?? {}, args.username)
      ? input.permissionsByUser[args.username] : input.permission;
  }
  if (method === 'pulls.get') return data.pull;
  if (method === 'issues.listComments') return data.comments;
  if (method === 'pulls.listReviews') return data.reviews;
  if (method === 'pulls.listReviewComments') return data.inlineComments;
  if (method === 'repos.listCommitStatusesForRef') return data.statuses;
  if (method === 'issues.getComment') return data.comments.find(comment => comment.id === args.comment_id);
  if (method === 'pulls.getReview') return data.reviews.find(review => review.id === args.review_id);
  if (method === 'pulls.getReviewComment') return data.inlineComments.find(comment => comment.id === args.comment_id);
  if (method === 'pulls.listCommits') return data.commits;
  if (method === 'issues.listEventsForTimeline') return data.timeline;
  if (method === 'actions.getWorkflowRun') return data.runs.find(run => run.id === args.run_id);
  if (method === 'actions.listWorkflowRunArtifacts') return data.artifacts[String(args.run_id)] ?? [];
  if (method === 'actions.listWorkflowRuns') {
    return args.event === 'workflow_dispatch' ? {workflow_runs: data.dispatchedRuns} : data.runs;
  }
  if (method === 'issues.getLabel') {
    if (input.labelExists === false) throw Object.assign(new Error('Label missing'), {status: 404});
    return {name: args.name};
  }
  if (['issues.createComment', 'issues.updateComment', 'issues.createLabel', 'issues.addLabels', 'issues.removeLabel',
       'actions.createWorkflowDispatch', 'repos.createDispatchEvent', 'repos.createCommitStatus'].includes(method)) {
    output.writes.push({method, ...args});
    if (method === 'issues.createLabel') input.labelExists = true;
    if (method === 'issues.removeLabel') data.pull.labels = data.pull.labels.filter(label => label.name !== args.name);
    if (method === 'issues.addLabels') {
      data.pull.labels.push(...args.labels.map(name => ({name})));
      data.timeline.push(...args.labels.map(name => ({event: 'labeled', label: {name}, actor: {login: 'github-actions[bot]'}})));
    }
    if (method === 'issues.createComment' || method === 'issues.updateComment') {
      data.comments ??= [];
      let comment = data.comments.find(item => item.id === args.comment_id);
      if (!comment) {
        comment = {id: 501, user: {login: 'github-actions[bot]'},
                   html_url: `https://github.com/${args.owner}/${args.repo}/pull/${args.issue_number}#issuecomment-501`};
        data.comments.push(comment);
      }
      return Object.assign(comment, {body: args.body});
    }
    return {id: 501};
  }
  throw new Error(`Unexpected GitHub call: ${method}`);
}

const rest = new Proxy({}, {get: (_, area) => new Proxy({}, {
  get: (_, name) => async args => {
    const data = response(`${area}.${name}`, args);
    const size = args.per_page ?? 30;
    const start = ((args.page ?? 1) - 1) * size;
    return {data: Array.isArray(data) ? data.slice(start, start + size) : data};
  },
})});
const github = {rest, paginate: async (fn, args) => {
  const items = [];
  for (let page = 1; ; page++) {
    const {data} = await fn({...args, page});
    items.push(...data);
    if (data.length < (args.per_page ?? 30)) return items;
  }
}};

function workflowModule(path) {
  const module = {exports: {}};
  new Function('require', 'module', fs.readFileSync(path, 'utf8'))(name => {
    if (name !== 'node:fs') throw new Error(`Unexpected dependency: ${name}`);
    return {
      existsSync: path => Object.hasOwn(input.files ?? {}, path),
      readFileSync: path => input.files[path],
    };
  }, module);
  return module.exports;
}

function resolve(value) {
  return String(value).replace(/\$\{\{\s*(.*?)\s*\}\}/g, (_, path) => {
    const vars = {
      github: {...input.context, token: 'workflow-token', event: input.context.payload},
      needs: input.needs,
      secrets: input.secrets,
      steps: {...input.stepsState, ...Object.fromEntries(Object.entries(output.outputs).map(([id, outputs]) => [id, {outputs}]))},
    };
    return path.split('.').reduce((obj, key) => obj?.[key], vars) ?? '';
  });
}

(async () => {
  for (const step of input.steps) {
    if (output.failures.length) break; // Actions' default success() step condition.
    const stepEnv = Object.fromEntries(
      Object.entries(step.env ?? {}).map(([key, value]) => [key, resolve(value)]),
    );
    process.env = {...baseEnv, ...stepEnv};
    const core = {
      info: () => {},
      setFailed: message => output.failures.push(message),
      warning: message => output.warnings.push(message),
      setOutput: (key, value) => {
        (output.outputs[step.id ?? step.name] ??= {})[key] = value;
      },
    };
    try {
      if (step.with?.script) {
        await new AsyncFunction('github', 'context', 'core', 'setTimeout', 'require', step.with.script)(
          github, input.context, core, callback => callback(), workflowModule,
        );
      } else if (step.run && input.renderShell) {
        output.shell.push({script: resolve(step.run), env: stepEnv});
      } else {
        throw new Error(`Unsupported step: ${step.name}`);
      }
    } catch (error) {
      core.setFailed(error.message);
    }
  }
  process.stdout.write(JSON.stringify(output));
})();
