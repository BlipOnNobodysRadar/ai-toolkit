"""Name job-config keys that nothing reads.

The config classes (TrainConfig, NetworkConfig, DatasetConfig, ModelConfig, ...) take **kwargs and read them with
kwargs.get(name, default). A misspelt key, or a key under the wrong section, is therefore dropped without a word and
the job trains as if it were not there: `train.lora_plus_ratio: 4` (a network option) or
`multires_noise_iteratons: 6` produces a baseline run that is labelled as something else.

The keys a class knows are read off its own __init__ source: every string literal it looks up in its kwargs. Nothing
is listed by hand, so options added upstream are known the moment they are merged. A class whose __init__ uses its
kwargs in a way this cannot follow (passes the whole dict on, loops over it) is not checked at all, rather than
guessed at: a false "unknown key" warning is worse than a missed one.

By default unknown keys are printed as a warning at job start. AITK_STRICT_CONFIG=1 raises instead.
"""
import ast
import difflib
import inspect
import os
import textwrap
from typing import Dict, Iterable, List, Optional, Set, Tuple

_KNOWN_CACHE: Dict[type, Optional[Set[str]]] = {}
_LOOKUPS = {'get', 'pop', 'setdefault'}


def _parents(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _init_known_keys(init) -> Optional[Set[str]]:
    """String keys one __init__ looks up in its **kwargs; None when it has none or uses them opaquely."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
    except (OSError, TypeError, SyntaxError):
        return None
    func = tree.body[0]
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    if func.args.kwarg is None:
        return set()  # takes no **kwargs: nothing to look up, nothing hidden
    name = func.args.kwarg.arg
    parents = _parents(func)
    known: Set[str] = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Name) and node.id == name):
            continue
        parent = parents.get(node)
        grand = parents.get(parent)
        # kwargs.get('key', ...) / .pop('key') / .setdefault('key', ...)
        if (isinstance(parent, ast.Attribute) and parent.value is node and parent.attr in _LOOKUPS
                and isinstance(grand, ast.Call) and grand.func is parent and grand.args
                and isinstance(grand.args[0], ast.Constant) and isinstance(grand.args[0].value, str)):
            known.add(grand.args[0].value)
        # kwargs['key']
        elif (isinstance(parent, ast.Subscript) and parent.value is node
                and isinstance(parent.slice, ast.Constant) and isinstance(parent.slice.value, str)):
            known.add(parent.slice.value)
        # 'key' in kwargs / 'key' not in kwargs
        elif (isinstance(parent, ast.Compare) and node in parent.comparators
                and all(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops)
                and isinstance(parent.left, ast.Constant) and isinstance(parent.left.value, str)):
            known.add(parent.left.value)
        else:
            return None  # the dict is used some other way: its keys cannot be listed from here
    return known


def known_keys(cls: type) -> Optional[Set[str]]:
    """Every key `cls(**section)` reads, base classes included; None when that cannot be established."""
    if cls not in _KNOWN_CACHE:
        known: Optional[Set[str]] = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get('__init__')
            if init is None or klass is object:
                continue
            keys = _init_known_keys(init)
            if keys is None:
                known = None
                break
            known |= keys
        _KNOWN_CACHE[cls] = known or None
    return _KNOWN_CACHE[cls]


def unknown_keys(section: Optional[dict], cls: type) -> List[str]:
    known = known_keys(cls)
    if known is None or not isinstance(section, dict):
        return []
    return [key for key in section if isinstance(key, str) and key not in known]


def _hint(key: str, cls: type, others: Iterable[Tuple[str, type]]) -> str:
    for label, other in others:
        if other is not cls and key in (known_keys(other) or ()):
            return f" (a `{label}` option)"
    close = difflib.get_close_matches(key, sorted(known_keys(cls) or ()), n=1, cutoff=0.8)
    return f" (did you mean `{close[0]}`?)" if close else ""


def check_sections(sections: List[Tuple[str, Optional[dict], type]], log=print) -> List[str]:
    """sections: (label, raw section dict, config class). Returns one message per unknown key, after reporting them.

    The label is the config path shown to the user ('train', 'datasets[0]'); its part before any '[' names the
    section when a key belongs under a different one.
    """
    homes = []
    for label, _, cls in sections:
        home = (label.split('[')[0], cls)
        if home not in homes:
            homes.append(home)
    messages = []
    for label, section, cls in sections:
        for key in unknown_keys(section, cls):
            messages.append(f"{label}.{key}{_hint(key, cls, homes)}")
    if not messages:
        return messages
    text = ("Config keys that nothing reads (they have no effect on this job):\n"
            + "\n".join(f"  - {m}" for m in messages))
    if os.environ.get('AITK_STRICT_CONFIG', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        raise ValueError(text + "\nAITK_STRICT_CONFIG is set, so the job stops here.")
    log("#" * 72 + "\nWARNING: " + text + "\nSet AITK_STRICT_CONFIG=1 to make this an error.\n" + "#" * 72)
    return messages


def check_train_process(process, log=None) -> List[str]:
    """Check the sections of a training process config (BaseSDTrainProcess and its subclasses).

    Reads the sections as the user wrote them (process.get_conf), so dataset indices match the config file and a
    dataset that the trainer later splits by resolution or path is reported once.
    """
    from toolkit import config_modules as cm
    if log is None:
        from toolkit.print import print_acc as log

    def section(name):
        value = process.get_conf(name, None)
        return value if isinstance(value, dict) else None

    train = section('train') or {}
    sections = [
        ('network', section('network'), cm.NetworkConfig),
        ('train', train, cm.TrainConfig),
        ('train.validation_config', train.get('validation_config'), cm.ValidationConfig),
        ('train.ema_config', train.get('ema_config'), cm.EMAConfig),
        ('model', section('model'), cm.ModelConfig),
        ('save', section('save'), cm.SaveConfig),
        ('sample', section('sample'), cm.SampleConfig),
        ('first_sample', section('first_sample'), cm.SampleConfig),
        ('logging', section('logging'), cm.LoggingConfig),
    ]
    datasets = process.get_conf('datasets', None)
    if isinstance(datasets, list):
        sections += [(f'datasets[{i}]', d, cm.DatasetConfig) for i, d in enumerate(datasets)]
    return check_sections(sections, log=log)
