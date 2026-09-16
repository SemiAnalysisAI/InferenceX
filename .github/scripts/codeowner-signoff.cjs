const fs = require('node:fs');

const MARKER = '<!-- codeowner-signoff-verify -->';
const AUTHORS = new Set(['Klaud-Cold', 'github-actions[bot]']);
const PASS = /^## ✅✅✅ \*\*Verdict: PASS\*\* ✅✅✅$/m;
const REJECT = /^## ❌❌❌ \*\*REJECTED\*\* ❌❌❌$/m;

function isVerdict(comment) {
  return AUTHORS.has(comment.user?.login) &&
    /^<!-- codeowner-signoff-verify(?: sha=[a-f0-9]{40})? -->\r?\n/.test(comment.body || '');
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

async function publish({ github, context, core, prNumber, headSha, verdictPath, verificationSucceeded }) {
  const comments = (await github.paginate(github.rest.issues.listComments, {
    ...context.repo, issue_number: prNumber, per_page: 100,
  })).filter(isVerdict);
  const current = comments.find(c => c.body.startsWith(MARKER)) || comments.at(-1);
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
  await upsert(github, context, prNumber, current,
    `${MARKER}\n${verdict}\n\nAssessed commit: \`${headSha}\`.\n`);
  core.info(`CODEOWNER sign-off=${passed ? 'success' : 'failure'} for assessed commit ${headSha}`);
}

module.exports = { publish };
