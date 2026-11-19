// Child process for ui/tests/upload.test.cjs: runs the actual server/uploadSession.ts in its own Node process.
//   node uploadChild.cjs <partDir> <destRoot> <mode> <uploadId> [gate] [offset] [dataFile]
//   finish                  plain finish
//   finish-exit-after-link  publish by hard link, then die before state.json says published
//   finish-exit-after-rename (overwrite upload) rename, then die before state.json says published
//   chunk-hold              write a chunk, pausing mid-write (lock held) until <gate> exists
//   cancel                  plain cancel
const fs = require('fs');
const [partDir, destRoot, mode, id, gate, offset, dataFile] = process.argv.slice(2);
const ts = require('../harness/tsrequire.cjs');
const up = require(ts.uiPath('src/server/uploadSession.ts'));
const wait = async file => {
  while (!fs.existsSync(file)) await new Promise(r => setTimeout(r, 10));
};
(async () => {
  let out;
  if (mode === 'finish') out = await up.finishUpload(partDir, destRoot, id, { lockWaitMs: 30000 });
  else if (mode === 'finish-exit-after-link') out = await up.finishUpload(partDir, destRoot, id, { hooks: { afterLink: () => process.exit(0) } });
  else if (mode === 'finish-exit-after-rename') out = await up.finishUpload(partDir, destRoot, id, { hooks: { afterRename: () => process.exit(0) } });
  else if (mode === 'chunk-hold') {
    let held = false;
    out = await up.writeChunk(partDir, id, Number(offset), fs.readFileSync(dataFile), {
      hooks: {
        duringWrite: async () => {
          if (held) return;
          held = true;
          fs.writeFileSync(`${gate}.holding`, '');
          await wait(gate);
        },
      },
    });
  } else if (mode === 'cancel') out = await up.cancelUpload(partDir, id, { lockWaitMs: 30000 });
  process.stdout.write(JSON.stringify(out));
})().catch(e => {
  console.error(e);
  process.exit(1);
});
