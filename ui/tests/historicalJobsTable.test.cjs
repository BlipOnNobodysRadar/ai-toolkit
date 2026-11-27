// Render the queue with real React and fixture responses from the three independent APIs.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { fromModules } = require('./harness/env.cjs');
const ts = require('./harness/tsrequire.cjs');
const React = fromModules('react');
const { renderToStaticMarkup } = fromModules('react-dom/server');
const job = (id, props) => ({ id, name: id, gpu_ids: '0', status: 'completed', step: 5, ...props });
const jobs = [
  job('old-caption', { job_type: 'caption', job_ref: null, job_config: 'bad json' }),
  job('missing-settings', { job_config: '{}' }),
  job('two-gpus', { gpu_ids: '1,0', status: 'queued', job_config: '{}' }),
  job('cpu-job', { gpu_ids: '-1', status: 'running', job_config: '{}' }),
];
ts.mock('@/hooks/useJobsList', () => ({ jobs, status: 'success', refreshJobs() {} }));
ts.mock('@/hooks/useQueueList', () => ({ queues: [{ gpu_ids: '1,0', is_running: true }], refreshQueues() {} }));
ts.mock('@/hooks/useGPUInfo', () => ({ gpuList: [], isGPUInfoLoaded: false }));
ts.mock('@/components/CaptionDatasetModal', { openCaptionDatasetModal() {} });
ts.mock('@/components/ConfirmModal', { openConfirm() {} });
ts.mock('@/utils/api', { apiClient: {} });
const JobsTable = require(ts.uiPath('src/components/JobsTable.tsx')).default;
const { getAvaliableJobActions, getTotalSteps } = require(ts.uiPath('src/utils/jobs.ts'));

test('historical and active rows remain accessible without telemetry, including incomplete configs', () => {
  const html = renderToStaticMarkup(React.createElement(JobsTable));
  assert.match(html, /href="\/jobs\/old-caption"/);
  for (const row of jobs) assert.ok(html.includes(row.name));
  assert.match(html, /GPU 1,0[\s\S]*?Queue Running[\s\S]*?two-gpus/);
  assert.match(html, /CPU[\s\S]*?Queue Stopped[\s\S]*?cpu-job/);
  for (const row of jobs.slice(0, 2)) {
    assert.equal(getAvaliableJobActions(row).canDelete, true);
    assert.equal(getTotalSteps(row), 0);
  }
});
