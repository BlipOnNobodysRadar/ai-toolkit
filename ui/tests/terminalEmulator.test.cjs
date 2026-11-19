// TerminalEmulator scrollback cap (docs/STATE_MACHINES.md M18 log view; Codex terminal_scrollback_probe).
// Runs the actual src/utils/terminalEmulator.ts, transpiled with the TypeScript compiler, under node:test (no browser).
// TypeScript is resolved from AITK_TS_PATH (an existing install, used read-only) or the normal module path.
//   node --test ui/tests/terminalEmulator.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ts = require(process.env.AITK_TS_PATH || 'typescript');
const source = fs.readFileSync(path.join(__dirname, '..', 'src', 'utils', 'terminalEmulator.ts'), 'utf8');
const code = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const box = { exports: {} };
vm.runInNewContext(code, box);
const { TerminalEmulator } = box.exports;
const MAX_LINES = 5000;

function valid(t) {
  const lines = t.toLines();
  assert.ok(lines.length <= MAX_LINES, `${lines.length} lines`);
  assert.ok(t.row >= 0 && t.row < lines.length, `row ${t.row} of ${lines.length}`);
  return lines;
}

test('a large cursor-down jump never exceeds the scrollback cap', () => {
  const t = new TerminalEmulator();
  t.write('\x1b[6000B'); // the original grew to 6001 lines, then 12001
  assert.equal(valid(t).length, MAX_LINES);
  t.write('\x1b[6000B');
  assert.equal(valid(t).length, MAX_LINES);
  t.write('\x1b[1000000E'); // E: next line and column 0
  valid(t);
});

test('text written after a jump lands on the cursor row; older text scrolls off', () => {
  const t = new TerminalEmulator();
  t.write('first\n');
  t.write('\x1b[10B');
  t.write('after');
  const lines = valid(t);
  assert.equal(lines[0], 'first');
  assert.equal(lines[lines.length - 1].trim(), 'after');
  t.write('\x1b[6000B');
  t.write('far');
  const later = valid(t);
  assert.ok(!later.includes('first'));
  assert.equal(later[later.length - 1].trim(), 'far');
});

test('newline pruning, up and back keep the cursor inside the grid', () => {
  const t = new TerminalEmulator();
  for (let i = 0; i < 6000; i++) t.write(`line ${i}\n`);
  valid(t);
  t.write('\x1b[3A\x1b[2D\x1b[5000B');
  valid(t);
  t.write('\x1bM\x1bM'); // reverse index
  valid(t);
});

test('a jump split across chunks is applied once, bounded', () => {
  const t = new TerminalEmulator();
  t.write('\x1b[60');
  t.write('00B');
  assert.equal(valid(t).length, MAX_LINES);
});

test('small moves behave as before', () => {
  const t = new TerminalEmulator();
  t.write('a\x1b[2Bb');
  assert.equal(JSON.stringify(t.toLines()), JSON.stringify(['a', '', ' b'])); // vm-realm array
});

test('finite-huge and overflowing counts stay bounded and text still lands on the cursor row', () => {
  for (const digits of ['100000000000000000000', '9'.repeat(310)]) { // 1e20; parseInt of 310 digits is Infinity
    const t = new TerminalEmulator();
    t.write('old\n');
    t.write(`[${digits}B`);
    t.write('next');
    const lines = valid(t);
    assert.equal(lines.length, MAX_LINES);
    assert.equal(lines[lines.length - 1], 'next');
    assert.ok(!lines.includes('old'));
  }
});

test('a count just below the cap uses the ordinary path', () => {
  const t = new TerminalEmulator();
  t.write('top\n');
  t.write(`[${MAX_LINES - 1}B`);
  t.write('x');
  const lines = valid(t);
  assert.equal(lines.length, MAX_LINES);
  assert.equal(lines[lines.length - 1], 'x');
});

test('moving down from the top keeps existing rows that have not scrolled off', () => {
  for (const [count, firstRetained] of [[5000, 'L1'], [6000, 'L1001'], [9998, 'L4999']]) {
    const t = new TerminalEmulator();
    t.write(Array.from({ length: MAX_LINES }, (_, i) => `L${i}`).join('\n'));
    t.write('\x1b[4999A');
    t.write(`\x1b[${count}B`);
    const lines = valid(t);
    assert.equal(lines.length, MAX_LINES);
    assert.equal(lines[0], firstRetained);
    t.write('new');
    assert.equal(valid(t)[MAX_LINES - 1], '     new'); // cursor-down preserves the column after "L4999"
  }
});
