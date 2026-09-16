const fs = require('node:fs');

const MARKER = '<!-- codeowner-signoff-verify -->';
const AUTHORS = new Set(['Klaud-Cold', 'github-actions[bot]']);
const PASS = /^## ✅✅✅ \*\*Verdict: PASS\*\* ✅✅✅$/m;
const REJECT = /^## ❌❌❌ \*\*REJECTED\*\* ❌❌❌$/m;
const SHA = /^[a-f0-9]{40}$/;

function isVerdict(comment) {
  return AUTHORS.has(comment.user?.login) &&
    /^<!-- codeowner-signoff-verify(?: sha=[a-f0-9]{40})? -->\r?\n/.test(comment.body || '');
}

function assessedCommit(comment) {
  if (!comment || !isVerdict(comment) || !PASS.test(comment.body) || REJECT.test(comment.body)) return null;
  return comment.body.match(/^<!-- codeowner-signoff-verify sha=([a-f0-9]{40}) -->/)?.[1] ||
    [...comment.body.matchAll(/^Assessed commit: `([a-f0-9]{40})`\.$/gm)].at(-1)?.[1];
}

async function state(github, context, prNumber) {
  const params = { ...context.repo, issue_number: prNumber };
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo, pull_number: prNumber,
  });
  const comments = (await github.paginate(github.rest.issues.listComments, {
    ...params, per_page: 100,
  })).filter(isVerdict);
  if (pr.labels.some(label => label.name === 'codeowner-signoff-verified')) {
    try {
      await github.rest.issues.removeLabel({ ...params, name: 'codeowner-signoff-verified' });
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
  return { pr, comment: comments.find(c => c.body.startsWith(MARKER)) || comments.at(-1) };
}

function coveredCommit(comment) {
  const assessed = assessedCommit(comment);
  if (!assessed) return null;
  return (comment.body.startsWith(MARKER) &&
    comment.body.match(/\nCovered commit: `([a-f0-9]{40})`\.\s*$/)?.[1]) || assessed;
}

async function coverage(github, context, prNumber) {
  const current = await state(github, context, prNumber);
  const covered = coveredCommit(current.comment);
  const event = context.payload;
  if (!covered || context.eventName !== 'pull_request_target' || event.action !== 'synchronize' ||
      event.before !== covered || !SHA.test(event.after) || event.after === covered ||
      event.pull_request?.head?.sha !== event.after || event.pull_request.number !== prNumber ||
      event.sender?.type !== 'User' || event.sender.login !== context.actor) return current;
  const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
    ...context.repo, username: context.actor,
  });
  if (data?.permission !== 'admin' || data?.role_name !== 'admin') return current;
  let verdict = current.comment.body.replace(/^<!-- codeowner-signoff-verify[^\n]*\r?\n/, '');
  const footer = verdict.lastIndexOf('\n\nAssessed commit:');
  if (footer !== -1) verdict = verdict.slice(0, footer);
  current.comment = await upsert(github, context, prNumber, current.comment,
    formatComment(verdict, assessedCommit(current.comment), event.after));
  return current;
}

function formatComment(verdict, assessed, covered) {
  return `${MARKER}\n${verdict}\n\nAssessed commit: \`${assessed}\`.\n` +
    `Covered commit: \`${covered}\`.\n`;
}

async function upsert(github, context, prNumber, comment, body) {
  if (comment) {
    if (comment.body === body) return comment;
    try {
      return (await github.rest.issues.updateComment({
        ...context.repo, comment_id: comment.id, body,
      })).data;
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
  return (await github.rest.issues.createComment({
    ...context.repo, issue_number: prNumber, body,
  })).data;
}

async function publishStatus(github, context, sha, status, comment,
  failureDescription = 'Fresh CODEOWNER sign-off verification required') {
  await github.rest.repos.createCommitStatus({
    ...context.repo, sha, context: 'CODEOWNER sign-off', state: status,
    description: status === 'success' ? 'CODEOWNER sign-off covers this commit' :
      status === 'pending' ? 'Verifying CODEOWNER sign-off' : failureDescription,
    target_url: comment?.html_url ||
      `https://github.com/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`,
  });
}

async function carry({ github, context, core, prNumber }) {
  const { pr, comment } = await coverage(github, context, prNumber);
  const passed = SHA.test(pr.head.sha) && coveredCommit(comment) === pr.head.sha;
  await publishStatus(github, context, pr.head.sha, passed ? 'success' : 'failure', comment);
  return passed;
}

async function prepare({ github, context, core, prNumber, headSha }) {
  const { comment } = await coverage(github, context, prNumber);
  const passed = SHA.test(headSha) && coveredCommit(comment) === headSha;
  const verify = context.eventName === 'workflow_dispatch' ||
    (!passed && (context.eventName !== 'pull_request_target' || !assessedCommit(comment)));
  await publishStatus(github, context, headSha,
    verify ? 'pending' : passed ? 'success' : 'failure', comment);
  core.setOutput('verify', String(verify));
}

async function publish({ github, context, core, prNumber, headSha, verdictPath, verificationSucceeded }) {
  const current = await state(github, context, prNumber);
  let verdict = '';
  if (verificationSucceeded && fs.existsSync(verdictPath)) {
    verdict = fs.readFileSync(verdictPath, 'utf8').trim();
  }
  const valid = (PASS.test(verdict) !== REJECT.test(verdict)) &&
    (verdict.startsWith('## ✅✅✅ **Verdict: PASS** ✅✅✅') ||
     verdict.startsWith('## ❌❌❌ **REJECTED** ❌❌❌'));
  if (!valid) {
    verdict = '## ❌❌❌ **REJECTED** ❌❌❌\n\nThe verifier did not produce a valid verdict. Retry the sign-off verification.';
  }
  const passed = PASS.test(verdict);
  const comment = await upsert(github, context, prNumber, current.comment,
    formatComment(verdict, headSha, headSha));
  await publishStatus(github, context, headSha, passed ? 'success' : 'failure', comment,
    'CODEOWNER sign-off rejected');
  const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: prNumber });
  if (pr.head.sha !== headSha) {
    await publishStatus(github, context, pr.head.sha,
      coveredCommit(comment) === pr.head.sha ? 'success' : 'failure', comment);
  }
  core.info(`CODEOWNER sign-off=${passed ? 'success' : 'failure'} for assessed commit ${headSha}`);
}

module.exports = { prepare, carry, publish };
