// Child process for ui/tests/deletion.test.cjs: runs the actual deleteJob against the test's temporary database.
//   node deleteChild.cjs <dbFile> <trainingRoot> <jobId> <mode> [gate]
//   mode exit-after-begin: commit the reservation, then exit before removing anything (a crashed deleter)
//   mode hold-until-gate:  hold the deletion lock until <gate> exists, then finish normally
const fs = require('fs');
const [dbFile, trainingRoot, jobId, mode, gate] = process.argv.slice(2);
const ts = require('../harness/tsrequire.cjs');
const { isolatedPrisma, clientFor } = require('../harness/prisma.cjs');
const p = isolatedPrisma();
ts.usePrismaClient(p.clientDir);
const { deleteJob } = require(ts.uiPath('cron/ownership/store.ts'));
const db = clientFor(p.PrismaClient, dbFile);
const wait = async file => {
  while (!fs.existsSync(file)) await new Promise(r => setTimeout(r, 20));
};
(async () => {
  const out = await deleteJob(db, jobId, trainingRoot, {
    afterBegin: async () => {
      if (mode === 'exit-after-begin') {
        await db.$disconnect();
        process.exit(0);
      }
      if (mode === 'hold-until-gate') {
        fs.writeFileSync(`${gate}.holding`, '');
        await wait(gate);
      }
    },
  });
  process.stdout.write(JSON.stringify(out));
  await db.$disconnect();
})().catch(e => {
  console.error(e);
  process.exit(1);
});
