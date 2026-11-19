"""Inference request/engine ownership (docs/STATE_MACHINES.md M11/M12, SM-09..SM-12; Codex C02-C06, OPUS-16).

The real Engine, GenerationJob, server routes and InferenceLoRA/LoRAStack, with the model holder replaced by CPU
stubs (no model loads, no GPU). Failures and races are injected at the points the original code left open:
request setup before the error boundary, cancel between dequeue and claim, unload against a running request,
reader disconnects, and LoRA application failing part-way.
"""
import asyncio
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from extensions_built_in.inference_engine import engine as engine_module
from extensions_built_in.inference_engine.engine import END, Engine, GenerationJob

EngineBusy = getattr(engine_module, 'EngineBusy', RuntimeError)
EngineClosed = getattr(engine_module, 'EngineClosed', RuntimeError)


def _header(frame):
    # protocol.encode_frame: u32 header length (little endian), JSON header, u64 payload length, payload
    length = int.from_bytes(frame[:4], 'little')
    return json.loads(frame[4:4 + length].decode('utf-8'))


def end_frames(job):
    return sum(1 for f in job.history if _header(f).get('type') == 'end')


class Holder:
    """Writes one output file per generate call; can block or fail on demand."""

    def __init__(self, gate=None, fail=None):
        self.gate, self.fail, self.calls = gate, fail, 0
        self.sample_step_hook = None

    def generate_images(self, configs, **kwargs):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail is not None:
            raise self.fail
        gen = configs[0]
        with open(os.path.join(gen.output_folder, 'out.png'), 'wb') as f:
            f.write(b'png')


def engine(tmp_path, holder=None):
    e = Engine('cpu', str(tmp_path / 'outputs'))
    e.ensured = []

    def ensure(job):
        e.ensured.append(job.request_id)
        return holder or Holder()
    e._ensure_holder = ensure
    return e


def run_until(e, jobs, timeout=20):
    thread = threading.Thread(target=e.run_forever, daemon=True)
    thread.start()
    deadline = time.time() + timeout
    while not all(j.finished for j in jobs) and time.time() < deadline:
        time.sleep(.01)
    e.stop()
    thread.join(timeout)
    assert not thread.is_alive()


# ---------------------------------------------------------------- C02: request setup inside the error boundary

def test_non_string_arch_is_refused_at_admission(tmp_path):
    e = engine(tmp_path)
    with pytest.raises(ValueError, match='arch'):
        e.submit({'arch': 42}, {})
    assert e.queue.empty()


def test_failed_request_setup_settles_the_job_and_the_worker_continues(tmp_path, monkeypatch):
    e = engine(tmp_path)
    bad = e.submit({'arch': 'sd1'}, {'prompt': 'a'})
    good = e.submit({'arch': 'sd1'}, {'prompt': 'b', 'width': 64, 'height': 64, 'num_inference_steps': 1})
    real_makedirs = os.makedirs

    def unwritable(path, *args, **kwargs):
        if str(path).endswith(bad.request_id):
            raise PermissionError('fixture: output directory unwritable')
        return real_makedirs(path, *args, **kwargs)
    monkeypatch.setattr(engine_module.os, 'makedirs', unwritable)
    run_until(e, [bad, good])
    assert bad.status == 'error' and 'PermissionError' in bad.error and end_frames(bad) == 1
    assert good.status == 'done' and end_frames(good) == 1


# ---------------------------------------------------------------- C04 / OPUS-16: claim vs cancel, one end frame

def test_cancel_between_dequeue_and_claim_never_runs_the_job(tmp_path):
    e = engine(tmp_path)
    job = e.submit({'arch': 'sd1'}, {})
    e.cancel(job.request_id)  # lands after run_forever's status check, before _run
    e._run(job)
    assert job.status == 'cancelled' and end_frames(job) == 1
    assert e.ensured == []  # no holder setup for an already-terminal request


def test_finish_is_exactly_once():
    job = GenerationJob({'arch': 'sd1'}, {})
    job.finish()
    job.finish()
    job.emit('status', message='late')
    assert end_frames(job) == 1 and _header(job.history[-1])['type'] == 'end'


def test_racing_cancel_and_claim_always_end_once(tmp_path):
    e = engine(tmp_path)
    for _ in range(200):
        job = e.submit({'arch': 'sd1'}, {'width': 64, 'height': 64, 'num_inference_steps': 1})
        e.queue.get_nowait()
        start = threading.Barrier(2)
        t = threading.Thread(target=lambda: (start.wait(), e.cancel(job.request_id)))
        t.start()
        start.wait()
        e._run(job)
        t.join()
        assert job.finished and end_frames(job) == 1
        assert job.status in ('cancelled', 'done')
        assert (job.status == 'cancelled') == (job.results == [])


def test_submit_after_the_engine_stopped_is_refused(tmp_path):
    e = engine(tmp_path)
    e.stop()
    e.run_forever()
    with pytest.raises(EngineClosed):
        e.submit({'arch': 'sd1'}, {})


# ---------------------------------------------------------------- C05: unload is serialized with generation

def test_unload_is_refused_while_a_request_runs_and_cannot_cut_into_one(tmp_path):
    gate = threading.Event()
    e = engine(tmp_path, Holder(gate=gate))
    e.holder = object()
    dropped = []
    e._drop_holder = lambda: dropped.append(True)
    job = e.submit({'arch': 'sd1'}, {'width': 64, 'height': 64, 'num_inference_steps': 1})
    thread = threading.Thread(target=e.run_forever, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while job.status != 'running' and time.time() < deadline:
        time.sleep(.01)
    # the HTTP thread's unload while the engine thread generates
    with pytest.raises(EngineBusy):
        e.unload()
    assert dropped == []
    gate.set()
    while not job.finished and time.time() < deadline:
        time.sleep(.01)
    e.stop()
    thread.join(10)
    e.unload()  # idle now
    assert dropped == [True] and job.status == 'done'


def test_a_request_waits_for_a_running_unload(tmp_path):
    e = engine(tmp_path)
    e.holder = object()
    inside, release = threading.Event(), threading.Event()

    def slow_drop():
        inside.set()
        release.wait(10)
    e._drop_holder = slow_drop
    unloader = threading.Thread(target=e.unload)
    unloader.start()
    inside.wait(10)
    job = e.submit({'arch': 'sd1'}, {'width': 64, 'height': 64, 'num_inference_steps': 1})
    worker = threading.Thread(target=e.run_forever, daemon=True)
    worker.start()
    time.sleep(.7)  # the engine thread has dequeued it by now
    assert job.status == 'queued' and e.ensured == []
    release.set()
    unloader.join(10)
    deadline = time.time() + 10
    while not job.finished and time.time() < deadline:
        time.sleep(.01)
    e.stop()
    worker.join(10)
    assert job.status == 'done'


# ---------------------------------------------------------------- C06: reader disconnects

def stream_route(e):
    from extensions_built_in.inference_engine.server import create_app
    app = create_app(e)
    return next(r.endpoint for r in app.routes if getattr(r, 'path', None) == '/stream/{request_id}')


def test_disconnected_reader_is_detached_without_cancelling_the_job(tmp_path):
    e = engine(tmp_path)
    job = e.submit({'arch': 'sd1'}, {})
    route = stream_route(e)
    request = SimpleNamespace(headers={})

    async def connect_and_drop():
        response = await route(request, job.request_id)
        body = response.body_iterator
        job.emit('status', message='first')
        assert _header(await body.__anext__())['type'] == 'status'
        await body.aclose()  # the client went away (page reload)
    for _ in range(5):  # repeated reconnects
        asyncio.run(connect_and_drop())
    assert job._subscribers == [] and not job.cancel.is_set() and job.status == 'queued'


def test_unsubscribe_wakes_a_blocked_reader():
    job = GenerationJob({'arch': 'sd1'}, {})
    q = job.subscribe(replay=False)
    got = []
    reader = threading.Thread(target=lambda: got.append(q.get(timeout=10)))
    reader.start()
    time.sleep(.1)
    job.unsubscribe(q)
    reader.join(10)
    assert got == [END] and job._subscribers == [] and not job.finished


# ---------------------------------------------------------------- C03: LoRA application failing part-way

def tiny_lora(tmp_path, second_shape=(2, 2)):
    path = tmp_path / 'tiny.safetensors'
    save_file({'transformer.0.diff': torch.ones(2, 2), 'transformer.1.diff': torch.ones(*second_shape)}, str(path))
    return str(path)


def lora_holder():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False))
    for p in model.parameters():
        p.data.zero_()
    return SimpleNamespace(model=model, text_encoder=None)


def test_mis_shaped_diff_is_rejected_before_any_weight_changes(tmp_path):
    from toolkit.inference_lora import LoRAStack
    holder = lora_holder()
    stack = LoRAStack(holder, 'merge').load([{'path': tiny_lora(tmp_path, (3, 3))}])
    assert len(stack.loras[0].entries) == 1 and 'diff shape' in stack.loras[0].unmatched[0]
    stack.apply()
    assert holder.model[0].weight.sum() == 4 and holder.model[1].weight.sum() == 0


@pytest.mark.parametrize('mode', ['merge', 'hook'])
def test_failure_part_way_through_apply_leaves_nothing_reusable_dirty(tmp_path, monkeypatch, mode):
    from toolkit import inference_lora
    e = engine(tmp_path)
    holder = lora_holder()
    e.holder, e.holder_key, e.holder_model = holder, 'key', {'arch': 'sd1'}
    e.pool.put('dit', holder.model)
    first, second = holder.model[0], holder.model[1]
    merged_first = []
    if mode == 'merge':
        real_delta = inference_lora.LoRAEntry.delta

        def delta(entry, strength, device, dtype=torch.float32):
            if entry.module is second:
                merged_first.append(first.weight.detach().sum().item())
                raise torch.OutOfMemoryError('fixture: OOM while merging the second module')
            return real_delta(entry, strength, device, dtype)
        monkeypatch.setattr(inference_lora.LoRAEntry, 'delta', delta)
    else:
        real_register = torch.nn.Module.register_forward_hook

        def register(module, hook, *args, **kwargs):
            if module is second:
                raise RuntimeError('fixture: hook registration failed')
            return real_register(module, hook, *args, **kwargs)
        monkeypatch.setattr(torch.nn.Module, 'register_forward_hook', register)
    job = GenerationJob({'arch': 'sd1'}, {})
    with pytest.raises((torch.OutOfMemoryError, RuntimeError)):
        e._apply_loras(job, [{'path': tiny_lora(tmp_path)}], mode)
    if mode == 'merge':
        assert merged_first == [4.]  # the first module really was merged before the failure ...
        assert e.holder is None and 'dit' not in e.pool.entries  # ... so neither the holder nor the pool keeps it
    else:
        assert not first._forward_hooks and not second._forward_hooks
        assert e.holder is holder and e.lora_stack is None


# ---------------------------------------------------------------- C15: startup/shutdown and exceptional loop exit

def test_a_worker_loop_that_dies_closes_admission_and_settles_queued_work(tmp_path):
    # the status callback failure is injected (UIJobStatus swallows ordinary DB write errors)
    e = engine(tmp_path)
    first = e.submit({'arch': 'sd1'}, {})
    queued = e.submit({'arch': 'sd1'}, {})

    def failing_status():
        raise OSError('injected idle status callback failure')
    e._idle_status = failing_status
    with pytest.raises(OSError, match='injected idle status'):
        e.run_forever()
    assert first.finished and first.status == 'done'
    assert queued.finished and queued.status == 'cancelled'
    with pytest.raises(EngineClosed):
        e.submit({'arch': 'sd1'}, {})


def inference_app(tmp_path, monkeypatch, preload):
    """The real InferenceEngine.run/_serve/shutdown; the HTTP server, the model preload and signal installation are
    the stubbed boundaries (no server, model or GPU)."""
    import signal
    from extensions_built_in.inference_engine import InferenceEngine as module
    from extensions_built_in.inference_engine import server as server_module
    from jobs.process import BaseExtensionProcess
    events = []
    handlers = {}
    monkeypatch.setattr(BaseExtensionProcess, 'run', lambda self: None)
    monkeypatch.setattr(server_module, 'create_app', lambda *a: None)
    monkeypatch.setattr(engine_module.Engine, 'preload', lambda self, model: preload(events))
    monkeypatch.setattr(module.signal, 'signal',
                        lambda sig, handler: handlers.setdefault('installed', []).append((sig, handler)) or 'previous')
    app = module.InferenceEngine.__new__(module.InferenceEngine)
    app.server = app.server_thread = app.endpoint = app.engine = None
    app._previous_signal_handlers = {}
    app.job_folder = app.output_folder = str(tmp_path)
    app.endpoint_path = str(tmp_path / 'engine.json')
    app.device = 'cpu'
    app.engine_config = SimpleNamespace(dtype='fp32', token='fixture', default_model={'arch': 'fixture'},
                                        host='127.0.0.1', port=0)
    app.ui = SimpleNamespace(
        job_id='fixture', update_info=lambda *a: None,
        update_status=lambda status, info: events.append(('status', status)),
        start_stop_watcher=lambda on_stop: events.append('watcher_started'),
        stop_stop_watcher=lambda: events.append('watcher_stopped'))

    def start_server(application, token):
        events.append('server_started')
        app.server = SimpleNamespace(should_exit=False)
        (tmp_path / 'engine.json').write_text('{}')
    app.start_server = start_server
    return app, events, handlers, signal


def test_a_stop_during_model_preload_still_shuts_the_engine_down(tmp_path, monkeypatch):
    # Codex inference_startup_probe: the KeyboardInterrupt raised by _on_signal during preload skipped shutdown()
    def interrupted(events):
        events.append('preload_entered')
        raise KeyboardInterrupt('stop during model preload')
    app, events, handlers, signal = inference_app(tmp_path, monkeypatch, interrupted)
    with pytest.raises(KeyboardInterrupt, match='preload'):
        app.run()
    assert events.index('preload_entered') < events.index('watcher_stopped')
    assert events[-1] == ('status', 'stopped')
    assert app.server.should_exit and not (tmp_path / 'engine.json').exists()
    restored = [h for sig, h in handlers['installed'] if h == 'previous']
    assert len(restored) == 2  # SIGINT and SIGTERM handlers given back


def test_a_failing_shutdown_does_not_hide_the_startup_error(tmp_path, monkeypatch):
    def broken(events):
        raise RuntimeError('model load failed')
    app, events, handlers, signal = inference_app(tmp_path, monkeypatch, broken)
    app.ui.update_status = lambda status, info: (_ for _ in ()).throw(OSError('db locked')) \
        if status == 'stopped' else None
    with pytest.raises(RuntimeError, match='model load failed') as caught:
        app.run()
    assert any('db locked' in note for note in caught.value.__notes__)


def test_the_stop_watcher_ends_when_asked(tmp_path, monkeypatch):
    from toolkit.ui_job_status import UIJobStatus
    db = tmp_path / 'db.sqlite'
    db.write_bytes(b'')
    monkeypatch.setenv('AITK_JOB_ID', 'fixture')
    ui = UIJobStatus(str(db))
    polls = []
    ui.should_stop = lambda: polls.append(1) or False
    ui.should_return_to_queue = lambda: False
    ui.start_stop_watcher(lambda reason: polls.append(reason), interval_sec=.01)
    ui.stop_stop_watcher()
    ui._watcher.join(2)
    assert not ui._watcher.is_alive() and 'stopped' not in polls


def test_requests_admitted_before_a_failed_startup_are_settled(tmp_path, monkeypatch):
    # HTTP is up before preload; a request admitted then must not wait on a worker that never starts
    admitted = []

    def broken(events):
        admitted.append(app.engine.submit({'arch': 'sd1'}, {}))
        raise RuntimeError('model load failed')
    app, events, handlers, signal = inference_app(tmp_path, monkeypatch, broken)
    with pytest.raises(RuntimeError, match='model load failed'):
        app.run()
    assert admitted[0].finished and admitted[0].status == 'cancelled' and end_frames(admitted[0]) == 1
    with pytest.raises(EngineClosed):
        app.engine.submit({'arch': 'sd1'}, {})


def test_every_shutdown_step_runs_when_one_fails(tmp_path, monkeypatch):
    from extensions_built_in.inference_engine import InferenceEngine as module

    def broken(events):
        raise RuntimeError('model load failed')
    app, events, handlers, signal = inference_app(tmp_path, monkeypatch, broken)
    real_remove = module.os.remove
    monkeypatch.setattr(module.os, 'remove', lambda path: (_ for _ in ()).throw(PermissionError('endpoint in use')))
    with pytest.raises(RuntimeError, match='model load failed') as caught:
        app.run()
    monkeypatch.setattr(module.os, 'remove', real_remove)
    assert any('endpoint in use' in note for note in caught.value.__notes__)
    assert app.server.should_exit and events[-1] == ('status', 'stopped')  # steps before and after still ran


def test_no_stop_callback_can_start_after_the_watcher_is_stopped(tmp_path, monkeypatch):
    # Codex watcher_stop_probe: a flag read in flight when stop_stop_watcher returned later delivered 'stopped'
    from toolkit.ui_job_status import UIJobStatus
    db = tmp_path / 'db.sqlite'
    db.write_bytes(b'')
    monkeypatch.setenv('AITK_JOB_ID', 'fixture')
    ui = UIJobStatus(str(db))
    reading, release, callbacks = threading.Event(), threading.Event(), []

    def blocked_read():
        reading.set()
        release.wait(5)
        return True  # the stop flag was set meanwhile
    ui.should_stop = blocked_read
    ui.should_return_to_queue = lambda: False
    ui.start_stop_watcher(callbacks.append, interval_sec=.01)
    assert reading.wait(5)
    ui.stop_stop_watcher()  # returns while the read is still blocked (the lock is not held during DB reads)
    release.set()
    ui._watcher.join(5)
    assert not ui._watcher.is_alive() and callbacks == []


def test_a_callback_already_running_finishes_before_stop_returns(tmp_path, monkeypatch):
    from toolkit.ui_job_status import UIJobStatus
    db = tmp_path / 'db.sqlite'
    db.write_bytes(b'')
    monkeypatch.setenv('AITK_JOB_ID', 'fixture')
    ui = UIJobStatus(str(db))
    entered, release, finished = threading.Event(), threading.Event(), []

    def on_stop(reason):
        entered.set()
        release.wait(5)
        finished.append(reason)
    ui.should_stop = lambda: True
    ui.should_return_to_queue = lambda: False
    ui.start_stop_watcher(on_stop, interval_sec=.01)
    assert entered.wait(5)
    stopper = threading.Thread(target=ui.stop_stop_watcher)
    stopper.start()
    stopper.join(.2)
    assert stopper.is_alive()  # waits for the callback in flight
    release.set()
    stopper.join(5)
    assert finished == ['stopped']


@pytest.mark.parametrize('failure', ['timeout', 'interrupt'])
def test_a_server_that_fails_to_start_is_stopped_and_its_socket_closed(monkeypatch, failure):
    # socket, uvicorn.Server and the wait are the stubbed boundaries: no port is bound
    import uvicorn
    from extensions_built_in.inference_engine import server as server_module
    closed = []

    class Socket:
        def __init__(self, *a): pass
        def setsockopt(self, *a): pass
        def bind(self, address): pass
        def listen(self, n): pass
        def getsockname(self): return ('127.0.0.1', 5555)
        def close(self): closed.append(True)

    class Server:
        def __init__(self, config):
            self.started, self.should_exit = False, False

        def run(self, sockets=None):
            tick = threading.Event()
            while not self.should_exit:  # never reports started (Event.wait: time.sleep may be patched below)
                tick.wait(.01)
    servers = []
    monkeypatch.setattr(server_module.socket, 'socket', Socket)
    monkeypatch.setattr(uvicorn, 'Config', lambda *a, **kw: None)
    monkeypatch.setattr(uvicorn, 'Server', lambda config: servers.append(Server(config)) or servers[-1])
    if failure == 'interrupt':
        monkeypatch.setattr(server_module.time, 'sleep',
                            lambda s: (_ for _ in ()).throw(KeyboardInterrupt('stop during startup wait')))
        expected = KeyboardInterrupt
    else:
        expected = RuntimeError
    with pytest.raises(expected):
        server_module.serve_in_thread(None, timeout=.2)
    assert servers[0].should_exit and closed == [True]
    assert not any(t.name == 'engine-http' and t.is_alive() for t in threading.enumerate())
