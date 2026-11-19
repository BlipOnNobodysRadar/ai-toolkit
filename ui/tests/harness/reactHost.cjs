// SIMULATED React host for hook tests (not React itself, not a browser). No react-test-renderer/jsdom is installed
// and none may be added, so this implements the subset of React's hook contract the ui/src/hooks modules use:
// useState (functional updates, batched re-render), useRef, useMemo/useCallback (deps compared with Object.is),
// useEffect (after each commit: cleanups of every effect whose deps changed run first, in order, then their setups;
// all cleanups on unmount). Results from it are evidence about the hooks' own logic under these semantics, not about
// React scheduling or a real browser.
let current = null;

const depsChanged = (a, b) => !a || !b || a.length !== b.length || a.some((x, i) => !Object.is(x, b[i]));

const react = {
  useState(initial) {
    const h = current;
    const i = h.index++;
    if (!(i in h.slots)) h.slots[i] = { value: typeof initial === 'function' ? initial() : initial };
    const slot = h.slots[i];
    if (!slot.set) {
      slot.set = next => {
        const value = typeof next === 'function' ? next(slot.value) : next;
        if (Object.is(value, slot.value)) return;
        slot.value = value;
        h.schedule();
      };
    }
    return [slot.value, slot.set];
  },
  useRef(initial) {
    const h = current;
    const i = h.index++;
    if (!(i in h.slots)) h.slots[i] = { current: initial };
    return h.slots[i];
  },
  useMemo(fn, deps) {
    const h = current;
    const i = h.index++;
    const slot = h.slots[i];
    if (!slot || depsChanged(slot.deps, deps)) h.slots[i] = { value: fn(), deps };
    return h.slots[i].value;
  },
  useCallback(fn, deps) {
    return react.useMemo(() => fn, deps);
  },
  useEffect(effect, deps) {
    const h = current;
    const i = h.index++;
    const slot = h.slots[i] ?? (h.slots[i] = { deps: undefined, cleanup: undefined, mounted: false });
    if (!slot.mounted || deps === undefined || depsChanged(slot.deps, deps)) h.pending.push({ slot, effect, deps });
  },
};
react.useLayoutEffect = react.useEffect;

const renderHook = (hook, props) => {
  const h = { slots: {}, index: 0, pending: [], props, result: { current: undefined }, scheduled: false, unmounted: false, renders: 0 };
  const commit = () => {
    const effects = h.pending;
    h.pending = [];
    for (const e of effects) if (typeof e.slot.cleanup === 'function') e.slot.cleanup();
    for (const e of effects) {
      e.slot.cleanup = e.effect();
      e.slot.deps = e.deps;
      e.slot.mounted = true;
    }
  };
  const render = () => {
    if (h.unmounted) return;
    current = h;
    h.index = 0;
    try {
      h.result.current = hook(h.props);
      h.renders += 1;
    } finally {
      current = null;
    }
    commit();
  };
  h.schedule = () => {
    if (h.scheduled || h.unmounted) return;
    h.scheduled = true;
    queueMicrotask(() => {
      h.scheduled = false;
      render();
    });
  };
  render();
  return {
    result: h.result,
    rerender(next) {
      h.props = next;
      render();
    },
    unmount() {
      h.unmounted = true;
      for (const slot of Object.values(h.slots)) if (typeof slot.cleanup === 'function') slot.cleanup();
    },
    get renders() {
      return h.renders;
    },
  };
};

// let every queued promise reaction and scheduled re-render run
const flush = async (rounds = 20) => {
  for (let i = 0; i < rounds; i++) await new Promise(r => setImmediate(r));
};

// A scripted apiClient: each call is recorded and returns a promise the test settles explicitly, in any order.
const scriptedApi = () => {
  const calls = [];
  const make = method => (url, config) => {
    let resolve, reject;
    const promise = new Promise((a, b) => {
      resolve = a;
      reject = b;
    });
    calls.push({ method, url, config, resolve: data => resolve({ data }), reject });
    return promise;
  };
  return { calls, apiClient: { get: make('get'), delete: make('delete'), post: make('post') } };
};

module.exports = { react, renderHook, flush, scriptedApi };
