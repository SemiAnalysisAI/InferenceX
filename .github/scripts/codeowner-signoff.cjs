const fs = require('node:fs');

const LABEL = 'codeowner-signoff-verified';
const MARKER = '<!-- codeowner-signoff-verify -->';
const AUTHORS = new Set(['Klaud-Cold', 'github-actions[bot]']);
const PASS = /^## ✅✅✅ \*\*Verdict: PASS\*\* ✅✅✅$/m;
const REJECT = /^## ❌❌❌ \*\*REJECTED\*\* ❌❌❌$/m;

function isVerdict(comment) {
  return AUTHORS.has(comment.user?.login) &&
    /^<!-- codeowner-signoff-verify(?: sha=[a-f0-9]{40})? -->\r?\n/.test(comment.body || '');
}

async function state(github, context, prNumber) {
  const params = { ...context.repo, issue_number: prNumber };
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo, pull_number: prNumber,
  });
  const comments = (await github.paginate(github.rest.issues.listComments, {
    ...params, per_page: 100,
  })).filter(isVerdict);
  const labeled = pr.labels.some(label => label.name === LABEL);
  let labelTrusted = false;
  if (labeled) {
    const events = await github.paginate(github.rest.issues.listEventsForTimeline, {
      ...params, per_page: 100,
    });
    const applied = events.filter(event => event.event === 'labeled' && event.label?.name === LABEL).at(-1);
    labelTrusted = AUTHORS.has(applied?.actor?.login);
  }
  return {
    pr, comments, labelTrusted,
    passed: labelTrusted || comments.some(c => PASS.test(c.body)),
    labeled,
    comment: comments.find(c => c.body.startsWith(MARKER)) || comments[0],
  };
}

async function rememberPass(github, context, prNumber, current) {
  if (current.labelTrusted) return;
  if (current.labeled) {
    // Replace a manually applied label so its timeline provenance is automation.
    await github.rest.issues.removeLabel({
      ...context.repo, issue_number: prNumber, name: LABEL,
    });
  }
  try {
    await github.rest.issues.getLabel({ ...context.repo, name: LABEL });
  } catch (error) {
    if (error.status !== 404) throw error;
    try {
      await github.rest.issues.createLabel({
        ...context.repo, name: LABEL, color: '0e8a16',
        description: 'CODEOWNER checklist passed once; retained across PR commits',
      });
    } catch (createError) {
      // Another PR may have created the repository label concurrently.
      if (createError.status !== 422) throw createError;
      await github.rest.issues.getLabel({ ...context.repo, name: LABEL });
    }
  }
  await github.rest.issues.addLabels({
    ...context.repo, issue_number: prNumber, labels: [LABEL],
  });
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
      // The comment was deleted after listing it.
    }
  }
  return (await github.rest.issues.createComment({
    ...context.repo, issue_number: prNumber, body,
  })).data;
}

async function publishStatus(github, context, prNumber, passed, comment, assessedSha) {
  // Refresh after verification: a push may have arrived while Claude was running.
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo, pull_number: prNumber,
  });
  const shas = new Set([pr.head.sha, assessedSha].filter(Boolean));
  for (const sha of shas) {
    await github.rest.repos.createCommitStatus({
      ...context.repo, sha, context: 'codeowner-signoff-verify',
      state: passed ? 'success' : 'failure',
      description: passed ? 'CODEOWNER checklist passed for this PR (retained across commits)' :
        'Sign-off verification rejected - see verdict comment',
      target_url: comment.html_url,
    });
  }
}

async function carry({ github, context, core, prNumber }) {
  const current = await state(github, context, prNumber);
  if (!current.passed) {
    core.info(`PR #${prNumber} has no passing checklist to carry forward.`);
    return false;
  }
  await rememberPass(github, context, prNumber, current);
  let comment = current.comment;
  if (comment && !comment.body.startsWith(MARKER)) {
    // Adopt the first legacy comment in place, using the most recent passing
    // assessment if older runs left multiple SHA-specific comments behind.
    const source = current.comments.filter(c => PASS.test(c.body)).at(-1) || comment;
    const sha = source.body.match(/^<!-- codeowner-signoff-verify sha=([a-f0-9]{40}) -->/);
    const body = source.body.replace(/^<!-- codeowner-signoff-verify[^\n]*\r?\n/, `${MARKER}\n`) +
      (sha ? `\n\nAssessed commit: \`${sha[1]}\`.\n` : '\n\n') +
      'This PR has passed the checklist. That pass is retained across later commits and reassessments.';
    comment = await upsert(github, context, prNumber, comment, body);
  } else if (!comment) {
    comment = await upsert(github, context, prNumber, null,
      `${MARKER}\n## ✅✅✅ **Verdict: PASS** ✅✅✅\n\n` +
      'This PR previously passed the CODEOWNER checklist. The original verdict comment was deleted.\n' +
      'The passing result is retained across commits; later commits have not been reverified.');
  }
  await publishStatus(github, context, prNumber, true, comment);
  return true;
}

async function prepare(args) {
  const passed = await carry(args);
  args.core.setOutput('verify', !passed || args.context.eventName === 'workflow_dispatch' ? 'true' : 'false');
}

async function publish({ github, context, core, prNumber, headSha, verdictPath, verificationSucceeded }) {
  const current = await state(github, context, prNumber);
  let verdict = '';
  if (verificationSucceeded && fs.existsSync(verdictPath)) {
    verdict = fs.readFileSync(verdictPath, 'utf8').trim();
  }
  // The model supplies text only. The workflow owns comment identity and status.
  const valid = (PASS.test(verdict) !== REJECT.test(verdict)) &&
    (verdict.startsWith('## ✅✅✅ **Verdict: PASS** ✅✅✅') ||
     verdict.startsWith('## ❌❌❌ **REJECTED** ❌❌❌'));
  if (!valid) {
    verdict = '## ❌❌❌ **REJECTED** ❌❌❌\n\nThe verifier did not produce a valid verdict. Retry the sign-off verification.';
  }
  const passed = current.passed || PASS.test(verdict);
  if (passed) await rememberPass(github, context, prNumber, current);
  const body = `${MARKER}\n${verdict}\n\nAssessed commit: \`${headSha}\`.\n` +
    (passed ? 'This PR has passed the checklist. That pass is retained across later commits and reassessments.' :
      'Edit the sign-off or rerun the workflow after addressing the findings.');
  const comment = await upsert(github, context, prNumber, current.comment, body);
  await publishStatus(github, context, prNumber, passed, comment, headSha);
  core.info(`codeowner-signoff-verify=${passed ? 'success' : 'failure'} for PR #${prNumber}`);
}

module.exports = { prepare, carry, publish };
