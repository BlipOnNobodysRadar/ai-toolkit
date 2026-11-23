// Shared test environment for the UI integration tests.
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

const fromModules = name => require(path.join(NODE_MODULES, name));

module.exports = { UI_ROOT, TOOLKIT_ROOT, NODE_MODULES, SCRATCH, fromModules };
