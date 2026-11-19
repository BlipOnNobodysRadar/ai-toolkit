// Shared test environment for the UI ownership/upload tests (docs/STATE_MACHINES.md section 18).
// Dependencies come from an existing ui/node_modules used READ-ONLY (AITK_UI_NODE_MODULES, default: this checkout's
// ui/node_modules). Everything the tests generate (Prisma client, databases, fixtures) goes under AITK_TEST_SCRATCH
// (default: the OS temp dir), never into node_modules or the production database.
const fs = require('fs');
const os = require('os');
const path = require('path');

const UI_ROOT = path.resolve(__dirname, '..', '..');
const TOOLKIT_ROOT = path.resolve(UI_ROOT, '..');
const NODE_MODULES = path.resolve(process.env.AITK_UI_NODE_MODULES || path.join(UI_ROOT, 'node_modules'));
const SCRATCH = path.resolve(process.env.AITK_TEST_SCRATCH || path.join(os.tmpdir(), 'aitk-ui-tests'));
fs.mkdirSync(SCRATCH, { recursive: true });

// The toolkit interpreter that runs the process helper (toolkit/ui_procctl.py) in the tests.
const PYTHON =
  process.env.AITK_TEST_PYTHON ||
  (process.platform === 'win32'
    ? path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python'));

const fromModules = name => require(path.join(NODE_MODULES, name));

module.exports = { UI_ROOT, TOOLKIT_ROOT, NODE_MODULES, SCRATCH, PYTHON, fromModules };
