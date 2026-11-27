const { test } = require('node:test');
const assert = require('node:assert/strict');
const ts = require('./harness/tsrequire.cjs');
const { react, renderHook, flush } = require('./harness/reactHost.cjs');

const element = (type, props) => ({ type, props });
ts.mock('react', react);
ts.mock('react/jsx-runtime', { jsx: element, jsxs: element });
ts.mock('@headlessui/react', { Dialog: 'dialog', DialogBackdrop: 'div', DialogPanel: 'section', DialogTitle: 'h3' });
ts.mock('react-icons/fa', { FaInfo: 'svg', FaExclamationTriangle: 'svg' });
ts.mock('./formInputs', { TextInput: 'input' });
ts.mock('react-global-hooks', {
  createGlobalState(initial) {
    let update;
    return {
      use() {
        const state = react.useState(initial);
        update = state[1];
        return state;
      },
      set(value) { update(value); },
    };
  },
});
const { default: ConfirmModal, openConfirm } = require(ts.uiPath('src/components/ConfirmModal.tsx'));

function button(node, label) {
  if (Array.isArray(node)) return node.map(child => button(child, label)).find(Boolean);
  if (!node?.props) return;
  if (node.type === 'button' && node.props.children === label) return node;
  return button(node.props.children, label);
}

test('replacing a closing confirmation within 500 ms preserves the new button and callback', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const calls = [];
  const modal = renderHook(ConfirmModal);
  t.after(() => modal.unmount());
  openConfirm({ title: 'Delete job', onConfirm: () => calls.push('delete') });
  await flush();
  modal.result.current.props.onClose();
  await flush();
  t.mock.timers.tick(400);
  openConfirm({ title: 'Stop job', confirmText: 'Stop', onConfirm: () => calls.push('stop') });
  await flush();
  t.mock.timers.tick(500);
  await flush();
  assert.equal(modal.result.current.props.open, true);
  const stop = button(modal.result.current, 'Stop');
  assert.ok(stop, 'replacement confirmation still displays Stop');
  stop.props.onClick();
  assert.deepEqual(calls, ['stop']);
});
