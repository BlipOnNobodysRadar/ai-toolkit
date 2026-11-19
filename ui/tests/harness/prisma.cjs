// Isolated Prisma for tests: copies a schema, points its generator output and datasource into SCRATCH, and runs the
// existing Prisma CLI (read-only node_modules) to generate a client there and push schemas to temporary SQLite files.
// It never generates into node_modules and never touches the production aitk_db.db.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { NODE_MODULES, SCRATCH, UI_ROOT } = require('./env.cjs');

const CLI = path.join(NODE_MODULES, 'prisma', 'build', 'index.js');
const cliEnv = extra => ({
  ...process.env,
  PRISMA_GENERATE_SKIP_AUTOINSTALL: '1',
  PRISMA_SKIP_POSTINSTALL_GENERATE: '1',
  PRISMA_HIDE_UPDATE_MESSAGE: '1',
  CHECKPOINT_DISABLE: '1',
  // the generated client resolves @prisma/client/runtime from the existing install
  NODE_PATH: NODE_MODULES,
  ...extra,
});

// Rewrite generator output and datasource url; everything else in the schema is used verbatim.
const isolateSchema = (schemaText, outDir) => {
  let s = schemaText.replace(/generator client \{[^}]*\}/, `generator client {\n  provider = "prisma-client-js"\n  output   = "${outDir.replace(/\\/g, '/')}"\n}`);
  s = s.replace(/url\s*=\s*"[^"]*"/, 'url      = env("AITK_TEST_DATABASE_URL")');
  if (s.includes('aitk_db.db')) throw new Error('schema still references the production database');
  return s;
};

const schemaDir = (schemaText, label) => {
  const hash = crypto.createHash('sha256').update(schemaText).digest('hex').slice(0, 12);
  const dir = path.join(SCRATCH, `prisma-${label}-${hash}`);
  const client = path.join(dir, 'client');
  const schemaPath = path.join(dir, 'schema.prisma');
  return { dir, client, schemaPath };
};

// `prisma generate` needs a resolvable @prisma/client next to the schema. Give it a private COPY of the installed
// package (never a link: nothing may write through into the shared install).
const privateClientPackage = dir => {
  const target = path.join(dir, 'node_modules', '@prisma', 'client');
  if (!fs.existsSync(path.join(target, 'package.json'))) {
    fs.cpSync(path.join(NODE_MODULES, '@prisma', 'client'), target, { recursive: true, dereference: true });
  }
};

// Generates (once per schema content) and returns { schemaPath, clientDir, PrismaClient }.
const isolatedPrisma = (schemaText = fs.readFileSync(path.join(UI_ROOT, 'prisma', 'schema.prisma'), 'utf8'), label = 'current') => {
  const { dir, client, schemaPath } = schemaDir(schemaText, label);
  if (!fs.existsSync(path.join(client, 'index.js'))) {
    fs.mkdirSync(dir, { recursive: true });
    privateClientPackage(dir);
    fs.writeFileSync(schemaPath, isolateSchema(schemaText, client));
    execFileSync(process.execPath, [CLI, 'generate', '--schema', schemaPath], {
      cwd: dir,
      env: cliEnv({ AITK_TEST_DATABASE_URL: `file:${path.join(dir, 'unused.db')}` }),
      stdio: 'pipe',
      windowsHide: true,
    });
  }
  if (!module.paths.includes(NODE_MODULES)) module.paths.push(NODE_MODULES);
  process.env.NODE_PATH = NODE_MODULES;
  require('module').Module._initPaths();
  const { PrismaClient } = require(client);
  return { schemaPath, clientDir: client, PrismaClient };
};

// Creates (or migrates) a SQLite file to schemaPath with `prisma db push` (no --accept-data-loss).
// RUST_LOG=info: on this host (Windows, Node 24.19, Prisma 6.3.1) `db push` exits with an empty "Schema engine
// error" unless the schema engine logs at info level; with it the push succeeds every time.
const pushSchema = (schemaPath, dbFile) => {
  execFileSync(process.execPath, [CLI, 'db', 'push', '--schema', schemaPath, '--skip-generate'], {
    cwd: path.dirname(schemaPath),
    env: cliEnv({ AITK_TEST_DATABASE_URL: `file:${dbFile.replace(/\\/g, '/')}`, RUST_LOG: 'info' }),
    stdio: 'pipe',
    windowsHide: true,
  });
};

let counter = 0;
const tempDbFile = label => {
  const dir = path.join(SCRATCH, 'db');
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, `${label}-${process.pid}-${Date.now()}-${counter++}.db`);
};

const clientFor = (PrismaClient, dbFile) => new PrismaClient({ datasources: { db: { url: `file:${dbFile}` } } });

module.exports = { isolatedPrisma, pushSchema, tempDbFile, clientFor, isolateSchema };
