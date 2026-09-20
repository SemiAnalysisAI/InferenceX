const fs = require('node:fs');

const MARKER = '<!-- codeowner-signoff-verify -->';
const AUTHORS = new Set(['Klaud-Cold', 'github-actions[bot]']);
const PASS = /^## ✅✅✅ \*\*Verdict: PASS\*\* ✅✅✅$/m;
const REJECT = /^## ❌❌❌ \*\*REJECTED\*\* ❌❌❌$/m;
const WARN = /^## ⚠️ \*\*Verdict: WARN\*\* ⚠️$/m;
const COVERAGE_WARNING = /^⚠️ Check 14 \(Pareto coverage\): WARN\b/m;
const ESCALATION = '⚠️ Pareto coverage needs additional review: @functionstackx @cquil11 @Oseltamivir @adibarra. ' +
  'At least 5 points per affected throughput-versus-E2EL frontier are highly recommended. ' +
  'Below 5, or when coverage cannot be verified, merge only with an explicit, recorded admin bypass ' +
  'for the assessed commit; this advisory comment does not grant or enforce a bypass.';

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
  const warning = COVERAGE_WARNING.test(verdict);
  const valid = [PASS, REJECT, WARN].filter(pattern => pattern.test(verdict)).length === 1 &&
    (!WARN.test(verdict) || warning) && (!PASS.test(verdict) || !warning) &&
    (verdict.startsWith('## ✅✅✅ **Verdict: PASS** ✅✅✅') ||
     verdict.startsWith('## ❌❌❌ **REJECTED** ❌❌❌') ||
     verdict.startsWith('## ⚠️ **Verdict: WARN** ⚠️'));
  if (!valid) {
    verdict = '## ❌❌❌ **REJECTED** ❌❌❌\n\nThe verifier did not produce a valid verdict. Retry the sign-off verification.';
  }
  if (valid && warning) {
    const newline = verdict.indexOf('\n');
    verdict = `${verdict.slice(0, newline)}\n\n${ESCALATION}\n${verdict.slice(newline)}`;
  }
  const status = PASS.test(verdict) ? 'success' : WARN.test(verdict) ? 'warning' : 'failure';
  await upsert(github, context, prNumber, current,
    `${MARKER}\n${verdict}\n\nAssessed commit: \`${headSha}\`.\n`);
  core.info(`CODEOWNER sign-off=${status} for assessed commit ${headSha}`);
}

module.exports = { publish };
