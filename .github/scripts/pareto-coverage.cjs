// Review-time counter, not a merge gate or an artifact/provenance validator.
// Mirrors chartFrontier(points, 'upper_right') in InferenceX-app at
// d507f3689274c82972709abb531bdedcb2e77946. See CONTRIBUTING.md#pareto-coverage.
const fs = require('node:fs');

const RECOMMENDED_POINTS = 5;

function assessCurve({ key, points }) {
  if (typeof key !== 'string' || !key.trim() || !Array.isArray(points)) {
    throw new Error('Each curve needs a nonempty key and a points array');
  }
  const hasCanonicalFlags = points.some(p =>
    p && p.isOnNormalizedInteractivityFrontier !== undefined);
  const eligible = points.filter(p =>
    p && Number.isFinite(p.x) && p.x > 0 && Number.isFinite(p.y) && p.y > 0);
  const invalid = points.length - eligible.length;
  const sorted = [...eligible].sort((a, b) => a.x - b.x || b.y - a.y);
  const frontier = [];
  let maxY = -Infinity;
  for (const point of sorted) {
    // Preserve the app's plateau behavior: equal throughput at different
    // latency values remains on this frontier. Exact coordinate duplicates do not.
    if (point.y > maxY ||
        (frontier.length > 0 && point.y === maxY && point.x > frontier.at(-1).x)) {
      if (frontier.length > 0 && point.x === frontier.at(-1).x) {
        frontier[frontier.length - 1] = point;
      } else {
        frontier.push(point);
      }
      maxY = point.y;
    }
  }
  // Intersect AFTER computing the selected-axis frontier from all eligible
  // points. Filtering first could promote an otherwise dominated point.
  const selected = hasCanonicalFlags
    ? frontier.filter(p => p.isOnNormalizedInteractivityFrontier === true)
    : frontier;
  return {
    key,
    status: invalid === 0 && selected.length >= RECOMMENDED_POINTS ? 'PASS' : 'WARN',
    measuredPoints: points.length,
    invalidPoints: invalid,
    frontierPoints: selected.length,
    recommendedPoints: RECOMMENDED_POINTS,
    canonicalRestriction: hasCanonicalFlags,
    frontier: selected,
  };
}

function assessCoverage(curves) {
  if (!Array.isArray(curves) || curves.length === 0) {
    throw new Error('Provide every affected curve; an empty list cannot prove coverage');
  }
  if (new Set(curves.map(c => c.key)).size !== curves.length) {
    throw new Error('Curve keys must be unique; do not split one curve across inputs');
  }
  return curves.map(assessCurve);
}

if (require.main === module) {
  try {
    console.log(JSON.stringify(assessCoverage(JSON.parse(fs.readFileSync(0, 'utf8'))), null, 2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = { assessCoverage };
