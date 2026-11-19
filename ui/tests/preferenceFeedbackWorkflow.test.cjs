// Focused source contract for the preference-rating component workflow.
// The repo's UI tests run without a browser DOM; this pins the structural guarantees that matter to the
// human review workflow and to browser QA while the route/API tests cover the actual persistence contract.
//   CUDA_VISIBLE_DEVICES=-1 node --test --test-concurrency=1 ui/tests/preferenceFeedbackWorkflow.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

const source = fs.readFileSync(path.join(__dirname, '..', 'src', 'components', 'PreferenceFeedback.tsx'), 'utf8');

test('preference feedback focuses one bounded pair at a time', () => {
  assert.match(source, /const selectedID = selectedByTab\[tab\]/);
  assert.match(
    source,
    /const selectedIndex = Math\.max\(\s*0,\s*visible\.findIndex\(pair => pair\.id === selectedID\)/,
  );
  assert.match(source, /const focusedPair = visible\[selectedIndex\] \?\? null/);
  assert.match(source, /Pair \{selectedIndex \+ 1\} of \{visible\.length\}/);
  assert.match(source, /Previous preference pair/);
  assert.match(source, /Next preference pair/);
  assert.doesNotMatch(source, /visible\.map\s*\(/, 'the pending gallery must not eagerly mount every pair');
});

test('focused comparison is large, responsive, stable A/B, and keyboard accessible', () => {
  assert.match(source, /className="grid grid-cols-1 gap-3 md:grid-cols-2"/);
  assert.match(source, /min-h-48 max-h-\[68vh\]/);
  assert.match(source, /\{\(\['a', 'b'\] as const\)\.map\(arm =>/);
  assert.match(source, /title=\{`Compare image \$\{armLabel\} full size`\}/);
  assert.match(source, /<Dialog open=\{comparison !== null\}/);
  assert.match(source, /<DialogTitle[^>]*>\s*Compare image A and image B at full size/);
  assert.match(source, /title="Close full-size comparison"/);
  assert.match(source, /alt=\{`Preference \$\{focusedPair\.id\}, image \$\{armLabel\}`\}/);
});

test('vote controls keep legacy QA titles and seed cues stay hidden until a vote exists', () => {
  assert.match(source, /title=\{`Choose \$\{armLabel\}`\}/);
  assert.match(source, /aria-label=\{`Choose \$\{armLabel\}`\}/);
  assert.match(source, /title="Skip this pair"/);
  assert.match(source, /aria-label="Skip this pair"/);
  assert.match(source, /focusedPair\.vote && \(/);
  assert.match(source, /Voted \{choiceLabel\[focusedPair\.vote\.choice\]\}/);
  assert.match(source, /seeds \{focusedPair\.seeds\[0\]\} \/ \{focusedPair\.seeds\[1\]\}/);
  const voteGateIndex = source.indexOf('focusedPair.vote && (');
  const seedIndex = source.indexOf('seeds {focusedPair.seeds[0]}');
  assert.ok(voteGateIndex >= 0 && seedIndex > voteGateIndex, 'seed metadata must be gated by an existing vote');
  assert.ok(!/^\s*seeds /m.test(source.slice(0, voteGateIndex)), 'pending metadata must not reveal seeds');
});

test('selection is keyed per tab and reset for a different job', () => {
  assert.match(source, /const EMPTY_SELECTION: Record<TabKey, string \| null> = \{/);
  assert.match(source, /setSelectedByTab\(prev => \(\{ \.\.\.prev, \[tab\]: visible\[nextIndex\]\.id \}\)\)/);
  assert.match(source, /setSelectedByTab\(\{ \.\.\.EMPTY_SELECTION \}\)/);
  assert.match(source, /setTab\('pending'\)/);
  assert.match(source, /}, \[job\.id\]\);/);
});

test('training summary shows recorded cumulative values and steps without fabricating state', () => {
  assert.match(source, /training\?: PreferenceFeedbackTrainingLog/);
  assert.match(source, /const training = feedback\?\.training \?\? null/);
  assert.match(source, />Latest training log</);
  for (const key of ['preference_updates', 'optimizer_updates', 'foreign_votes']) {
    assert.ok(source.includes('metric: training.' + key));
  }
  assert.match(source, /metric && \(/);
  assert.match(source, /at log step \{metric.step\}/);
  assert.match(source, /!hasLoggedProgress && recordedVoteCount > 0/);
  assert.match(source, /Waiting for a training log entry/);
  assert.doesNotMatch(source, /training\.preference_updates\.value \?\? 0/, 'missing metrics must not become zero');
  assert.doesNotMatch(source, /pair (?:was )?trained|caused .*training/i, 'never claim a specific pair trained');
});