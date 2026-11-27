// Apply this to the install as well: @prisma/client runs Prisma from its own postinstall hook.
const { spawnSync } = require('node:child_process');

const [tool, ...args] = process.argv.slice(2);
const cli = tool === 'npm' ? process.env.npm_execpath : require.resolve('prisma');
const result = spawnSync(process.execPath, [cli, ...args], {
  stdio: 'inherit',
  env: { CHECKPOINT_DISABLE: '1', ...process.env },
});
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
