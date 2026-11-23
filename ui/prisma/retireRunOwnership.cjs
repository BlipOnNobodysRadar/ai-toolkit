// Prisma refuses to drop populated columns during a normal db push. Clear only
// the retired ownership data so its usual data-loss checks still apply elsewhere.
const retiredColumns = [
  'run_id', 'launch_phase', 'launcher_pid', 'launcher_identity', 'run_dir',
  'spawn_requested_at', 'relay_pid', 'relay_exited_at', 'supervisor_identity',
  'run_boundary', 'boot_id', 'pid_identity', 'run_members', 'stop_requested_at',
  'after_exit', 'exit_verified_at', 'ownership_note',
];

async function retireRunOwnership(db) {
  await db.$transaction(async tx => {
    const columns = await tx.$queryRaw`PRAGMA table_info("Job")`;
    const present = retiredColumns.filter(name => columns.some(column => column.name === name));
    if (present.length) {
      // The upstream worker uses 'running' and the existing stop/return flags.
      // Leave interrupted deletions available for a manual retry in the UI.
      const assignments = present.map(name => `"${name}" = NULL`);
      await tx.$executeRawUnsafe(`UPDATE "Job" SET ${assignments.join(', ')},
        "status" = CASE
          WHEN "status" IN ('starting', 'stopping') THEN 'running'
          WHEN "status" IN ('deleting', 'delete_failed') THEN 'error'
          ELSE "status" END`);
    }
    await tx.$executeRaw`DROP TABLE IF EXISTS "RunClaim"`;
  });
}

if (require.main === module) {
  const { PrismaClient } = require('@prisma/client');
  const db = new PrismaClient();
  retireRunOwnership(db)
    .catch(error => {
      console.error(error);
      process.exitCode = 1;
    })
    .finally(() => db.$disconnect());
}

module.exports = { retireRunOwnership };
