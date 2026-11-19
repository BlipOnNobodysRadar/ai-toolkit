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
  }
  static json(body, init = {}) {
    return new TestResponse(body, init);
  }
}

ts.mock('next/server', { NextRequest: class {}, NextResponse: TestResponse });
let jobRow = { id: 'job', name: 'job' };
let trainingRoot = '';
ts.mock('@/server/prisma', { __esModule: true, default: { job: { findUnique: async () => jobRow } } });
ts.mock('@/server/settings', { getTrainingFolder: async () => trainingRoot });

const reports = require(ts.uiPath('src/server/jobReports.ts'));
const route = require(ts.uiPath('src/app/api/jobs/[jobID]/reports/route.ts'));

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
  const rows = ['timestep,sigma,log_variance,prediction_weight'];
  for (let index = 0; index < 1000; index += 1) {
    rows.push(`${index},${index / 999},${Math.log(1 + index / 100)},${1 + index / 1000}`);
  }
  rows.push('bad,row');
  const edm = reports.summarizeEdm2Csv(rows.join('\n'));
  assert.equal(edm.pointCount, 1000);
  assert.ok(edm.points.length <= 360);
  assert.equal(edm.points[0].timestep, 0);
  assert.equal(edm.points[edm.points.length - 1].timestep, 999);
  assert.equal(edm.minWeight, 1);
  assert.equal(edm.maxWeight, 1.999);
});

test('reports route reads only the job folder fixed artifacts and exposes raw download paths', async () => {
  trainingRoot = makeRoot();
  jobRow = { id: 'job', name: 'job' };
  const jobFolder = path.join(trainingRoot, 'job');
  fs.mkdirSync(jobFolder);
  fs.writeFileSync(
    path.join(jobFolder, 'validation.jsonl'),
    JSON.stringify({ step: 5, phase: 'final', raw_metrics: [{ sigma: 0.5, video_mse: 0.123 }] }) + '\n',
  );
  fs.writeFileSync(
    path.join(jobFolder, 'edm2-weighting.csv'),
    'timestep,sigma,log_variance,prediction_weight\n1,0.1,0.2,1.5\n2,0.2,0.3,1.7\n',
  );

  const response = await route.GET({}, { params: { jobID: 'job' } });
  assert.equal(response.status, 200);
  assert.equal(response.body.validation.exists, true);
  assert.equal(response.body.validation.latest.metrics.video_mse, 0.123);
  assert.equal(response.body.edm2.exists, true);
  assert.equal(response.body.edm2.pointCount, 2);
  assert.equal(response.body.edm2.points[1].predictionWeight, 1.7);
  assert.equal(response.body.validation.path, path.join(jobFolder, 'validation.jsonl'));
});

test('reports route refuses a job name that escapes the training root', async () => {
  trainingRoot = makeRoot();
  jobRow = { id: 'job', name: '../outside' };
  const response = await route.GET({}, { params: { jobID: 'job' } });
  assert.equal(response.status, 400);
  assert.match(response.body.error, /outside the training root/);
});
