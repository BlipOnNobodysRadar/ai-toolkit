// Read-only job report surface: parser and route regressions on real temporary files.
//   node --test --test-concurrency=1 ui/tests/jobReports.test.cjs
const { test, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const ts = require('./harness/tsrequire.cjs');
const { SCRATCH } = require('./harness/env.cjs');

class TestResponse {
  constructor(body, init = {}) {
    this.body = body;
    this.status = init.status || 200;
    this.headers = init.headers || {};
  }
  static json(body, init = {}) {
    return new TestResponse(body, init);
  }
}

const JOB_ID = '11111111-1111-4111-8111-111111111111';

ts.mock('next/server', { NextRequest: class {}, NextResponse: TestResponse });
let jobRow = { id: JOB_ID, name: 'job' };
let trainingRoot = '';
ts.mock('@/server/prisma', { __esModule: true, default: { job: { findUnique: async () => jobRow } } });
ts.mock('@/server/settings', { getTrainingFolder: async () => trainingRoot });

const reports = require(ts.uiPath('src/server/jobReports.ts'));
const route = require(ts.uiPath('src/app/api/jobs/[jobID]/reports/route.ts'));
const downloadRoute = require(ts.uiPath('src/app/api/jobs/[jobID]/reports/[report]/route.ts'));

const roots = [];
const makeRoot = () => {
  const root = fs.mkdtempSync(path.join(SCRATCH, 'job-reports-'));
  roots.push(root);
  return root;
};
after(() => {
  for (const root of roots.reverse()) {
    try {
      fs.rmSync(root, { recursive: true, force: true });
    } catch {
      // best effort on temporary fixtures
    }
  }
});

const setupJob = () => {
  trainingRoot = makeRoot();
  jobRow = { id: JOB_ID, name: 'job' };
  const folder = path.join(trainingRoot, 'job');
  fs.mkdirSync(folder);
  return folder;
};

const readBody = async body => {
  if (body === null || body === undefined) return '';
  if (typeof body === 'string') return body;
  if (Buffer.isBuffer(body)) return body.toString('utf8');
  if (body instanceof Uint8Array) return Buffer.from(body).toString('utf8');
  if (typeof body.getReader === 'function') {
    const reader = body.getReader();
    const chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(Buffer.from(value));
    }
    return Buffer.concat(chunks).toString('utf8');
  }
  throw new Error('Unsupported response body in test');
};

test('validation parser summarizes native and DPO metrics and counts malformed lines', () => {
  const native = reports.summarizeValidationJsonl(
    [
      JSON.stringify({
        step: 10,
        phase: 'initial',
        raw_metrics: [
          { item: 0, sigma: 0.25, video_mse: 0.4 },
          { item: 0, sigma: 0.75, video_mse: 0.8 },
        ],
      }),
      JSON.stringify({
        step: 20,
        phase: 'periodic',
        raw_metrics: [
          { item: 1, sigma: 0.25, video_mse: 0.2, audio_mse: 0.05 },
          { item: 1, sigma: 0.75, video_mse: 0.4, audio_mse: 0.15 },
        ],
      }),
      '{not-json',
    ].join('\n'),
  );
  assert.equal(native.entries, 2);
  assert.equal(native.malformedLines, 1);
  assert.equal(native.kind, 'native');
  assert.equal(native.latest.step, 20);
  assert.ok(Math.abs(native.latest.metrics.video_mse - 0.3) < 1e-12);
  assert.equal(native.latest.metrics.audio_mse, 0.1);
  assert.deepEqual(native.bySigma.map(row => row.sigma), [0.25, 0.75]);

  const dpo = reports.summarizeValidationJsonl(
    JSON.stringify({
      step: 25,
      phase: 'periodic',
      raw_metrics: [
        {
          sigma: 0.25,
          item: 0,
          dpo_loss: 0.6,
          margin: 0.4,
          strict_preference_correct: true,
          policy_chosen_error: 0.2,
          policy_rejected_error: 0.5,
        },
        {
          sigma: 0.25,
          item: 1,
          dpo_loss: 0.8,
          margin: -0.2,
          strict_preference_correct: false,
          policy_chosen_error: 0.5,
          policy_rejected_error: 0.4,
        },
      ],
    }),
  );
  assert.equal(dpo.kind, 'dpo');
  assert.equal(dpo.entries, 1);
  assert.equal(dpo.latest.metrics.dpo_loss, 0.7);
  assert.equal(dpo.latest.metrics.margin, 0.1);
  assert.equal(dpo.latest.metrics.strict_preference_correct, 0.5);
  assert.equal(dpo.bySigma[0].sampleCount, 2);
});

test('EDM2 CSV parser validates the documented columns, filters bad rows, and bounds chart points', () => {
  const sample = '1,0.25,0,1.5\n';
  const csv = 'timestep,sigma,log_variance,prediction_weight\n' + sample.repeat(Math.ceil(9 * 1024 * 1024 / sample.length));
  fs.writeFileSync(path.join(jobFolder, 'edm2-weighting.csv'), csv);
  const response = await route.GET({}, { params: { jobID: JOB_ID } });
  assert.equal(response.status, 200);
  assert.equal(response.body.validation.exists, true);
  assert.equal(response.body.validation.fileName, 'validation.jsonl');
  assert.equal(response.body.validation.latest.metrics.video_mse, 0.123);
  assert.equal(response.body.validation.previewTruncated, false);
  assert.equal(response.body.edm2.exists, true);
  assert.equal(response.body.edm2.fileName, 'edm2-weighting.csv');
  assert.equal(response.body.edm2.pointCount, 2);
  assert.equal(response.body.edm2.points[1].predictionWeight, 1.7);
  assert.equal(Object.hasOwn(response.body.validation, 'path'), false);
  assert.equal(Object.hasOwn(response.body.edm2, 'path'), false);
  const serialized = JSON.stringify(response.body);
  assert.equal(serialized.includes(jobFolder), false);
  assert.equal(serialized.includes(trainingRoot), false);
});

test('large validation reports expose a bounded recent tail instead of failing the preview', async () => {
  const jobFolder = setupJob();
  const recent = [
    JSON.stringify({ step: 100, phase: 'periodic', raw_metrics: [{ sigma: 0.5, video_mse: 0.111 }] }),
    JSON.stringify({ step: 101, phase: 'final', raw_metrics: [{ sigma: 0.5, video_mse: 0.099 }] }),
  ].join('\n');
  fs.writeFileSync(path.join(jobFolder, 'validation.jsonl'), 'x'.repeat(9 * 1024 * 1024) + '\n' + recent + '\n');

  const response = await route.GET({}, { params: { jobID: JOB_ID } });
  assert.equal(response.status, 200);
  assert.equal(response.body.validation.exists, true);
  assert.equal(response.body.validation.tooLarge, true);
  assert.equal(response.body.validation.previewTruncated, true);
  assert.equal(response.body.validation.entries, 2);
  assert.equal(response.body.validation.latest.step, 101);
  assert.ok(Math.abs(response.body.validation.latest.metrics.video_mse - 0.099) < 1e-12);
  assert.equal(JSON.stringify(response.body).includes(jobFolder), false);
});

test('large EDM2 reports remain discoverable without an unbounded preview read', async () => {
  const jobFolder = setupJob();
  const rows = ['timestep,sigma,log_variance,prediction_weight'];
  while (rows.join('\n').length <= 8 * 1024 * 1024 + 1024) {
    const index = rows.length - 1;
    rows.push(`${index},${index / 1000},0,${1 + index / 1000}`);
  }
  fs.writeFileSync(path.join(jobFolder, 'edm2-weighting.csv'), rows.join('\n'));
  const response = await route.GET({}, { params: { jobID: JOB_ID } });
  assert.equal(response.status, 200);
  assert.equal(response.body.edm2.exists, true);
  assert.equal(response.body.edm2.tooLarge, true);
  assert.equal(response.body.edm2.pointCount, 0);
});

test('raw report download streams exactly one fixed report name without path headers', async () => {
  const jobFolder = setupJob();
  const payload = '{"step":1,"phase":"final","raw_metrics":[{"sigma":0.5,"video_mse":0.25}]}\n';
  fs.writeFileSync(path.join(jobFolder, 'validation.jsonl'), payload);

  const response = await downloadRoute.GET({}, { params: { jobID: JOB_ID, report: 'validation.jsonl' } });
  assert.equal(response.status, 200);
  assert.equal(response.headers['Content-Type'], 'application/x-ndjson; charset=utf-8');
  assert.equal(response.headers['Content-Disposition'], 'attachment; filename="validation.jsonl"');
  assert.equal(await readBody(response.body), payload);
  assert.equal(JSON.stringify(response.headers).includes(jobFolder), false);
});

test('raw download rejects any name other than the two fixed report files', async () => {
  setupJob();
  for (const report of ['secret.txt', '../validation.jsonl', 'validation.jsonl.bak', '']) {
    const response = await downloadRoute.GET({}, { params: { jobID: JOB_ID, report } });
    assert.equal(response.status, 400, report);
    assert.equal(response.body.code, 'INVALID_REPORT_FILE');
  }
});

test('stale or foreign run bindings are refused cleanly and do not reveal configured paths', async () => {
  setupJob();
  const otherRoot = makeRoot();
  jobRow = { id: JOB_ID, name: 'job', run_dir: path.join(otherRoot, 'job') };
  const response = await route.GET({}, { params: { jobID: JOB_ID } });
  assert.equal(response.status, 409);
  assert.equal(response.body.code, 'INVALID_RUN_BINDING');
  const serialized = JSON.stringify(response.body);
  assert.equal(serialized.includes(otherRoot), false);
  assert.equal(serialized.includes(trainingRoot), false);
});

test('a missing claimed run folder is an ordinary empty report response', async () => {
  trainingRoot = makeRoot();
  jobRow = { id: JOB_ID, name: 'job', run_dir: path.join(trainingRoot, 'job') };
  const response = await route.GET({}, { params: { jobID: JOB_ID } });
  assert.equal(response.status, 200);
  assert.equal(response.body.validation.exists, false);
  assert.equal(response.body.edm2.exists, false);
  assert.equal(JSON.stringify(response.body).includes(trainingRoot), false);
});

test('link or non-file report entries are refused', async t => {
  const jobFolder = setupJob();
  const outside = path.join(makeRoot(), 'outside-validation.jsonl');
  fs.writeFileSync(
    outside,
    JSON.stringify({ step: 1, phase: 'final', raw_metrics: [{ sigma: 0.5, video_mse: 0.5 }] }),
  );
  const reportPath = path.join(jobFolder, 'validation.jsonl');
  try {
    fs.symlinkSync(outside, reportPath, 'file');
  } catch (error) {
    t.skip(`file symlink unavailable on this platform: ${error.code || error.message}`);
    return;
  }

  const summary = await route.GET({}, { params: { jobID: JOB_ID } });
  assert.equal(summary.status, 403);
  assert.equal(summary.body.code, 'UNSAFE_REPORT_FILE');
  const download = await downloadRoute.GET({}, { params: { jobID: JOB_ID, report: 'validation.jsonl' } });
  assert.equal(download.status, 403);
  assert.equal(download.body.code, 'UNSAFE_REPORT_FILE');
});

test('a directory at a fixed report name is not treated as a regular file', async () => {
  const jobFolder = setupJob();
  fs.mkdirSync(path.join(jobFolder, 'validation.jsonl'));
  const response = await route.GET({}, { params: { jobID: JOB_ID } });
  assert.equal(response.status, 403);
  assert.equal(response.body.code, 'UNSAFE_REPORT_FILE');
});
