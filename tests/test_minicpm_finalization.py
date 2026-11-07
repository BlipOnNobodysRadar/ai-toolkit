"""MiniCPM finalization order (docs/STATE_MACHINES.md M24; Codex minicpm_finalization_probe: the original
eval/train/checkpoint/eval sequence changed 79/256 bf16 checkpoint values).

The ACTUAL statements after main()'s training loop are extracted from training_extras/minicpm_trainer.py by AST
(optimizer mode switches, the final checkpoint and the adapter export) and executed in source order with a real CPU
bf16 ADOPTScheduleFree optimizer; checkpoint and export are recorders. No MiniCPM model, no GPU.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch  # noqa: E402

from training_extras.optim import create_optimizer, enter_training_mode  # noqa: E402

SOURCE = Path(__file__).resolve().parents[1] / "training_extras" / "minicpm_trainer.py"


def _main():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _finalization_statements(main):
    last_loop = max(n.end_lineno for n in main.body if isinstance(n, ast.For))

    def is_mode_switch(n):
        return isinstance(n, ast.If) and isinstance(n.test, ast.Call) and getattr(n.test.func, "id", None) == "callable"

    def is_call(n, name):
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)):
            return False
        f = n.value.func
        return getattr(f, "id", None) == name or getattr(f, "attr", None) == name

    return [n for n in main.body if n.lineno > last_loop and (is_mode_switch(n) or is_call(n, "checkpoint") or is_call(n, "save_pretrained"))]


def test_final_checkpoint_precedes_evaluation_and_there_is_no_train_round_trip():
    stmts = _finalization_statements(_main())
    kinds = [ast.unparse(n).split("(")[0] if isinstance(n, ast.Expr) else ast.unparse(n.body[0]) for n in stmts]
    assert kinds == ["checkpoint", "opt.eval()", "model.save_pretrained"], kinds


def test_actual_sequence_keeps_training_values_in_the_checkpoint_and_exports_the_evaluated_weights():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 16).to(torch.bfloat16))
    target = torch.randn(16, 16)
    opt = create_optimizer([p], "easy:ADOPTScheduleFree", 0.01)
    enter_training_mode(opt)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        (p.float() - target).square().sum().backward()
        opt.step()
    trained = p.detach().clone()
    saved, exported = [], []
    model = SimpleNamespace(save_pretrained=lambda *a, **k: exported.append(p.detach().clone()))
    code = compile(ast.Module(body=_finalization_statements(_main()), type_ignores=[]), "minicpm-finalization", "exec")
    exec(code, {"opt": opt, "model": model, "out": "unused", "a": SimpleNamespace(steps=4),
                "checkpoint": lambda step: saved.append(p.detach().clone())})
    assert len(saved) == 1 and len(exported) == 1
    assert int((saved[0] != trained).sum()) == 0, "checkpoint holds the training coordinates exactly"
    assert int((exported[0] != trained).sum()) > 0, "export holds evaluation coordinates (schedule-free x)"
    assert torch.equal(exported[0], p.detach()), "the run ends in evaluation mode with the exported weights"
    assert not torch.cuda.is_initialized()


def test_resume_guard_reachability_with_a_larger_step_count_is_source_checked():
    src = SOURCE.read_text(encoding="utf-8")
    trainer_opts = next(line for line in src.splitlines() if 'options["_trainer"]=' in line)
    assert '"steps"' not in trainer_opts  # --steps is not part of the hashed options
    assert "start_step>=a.steps" in src.replace(" ", "")  # only the same (or smaller) total is refused
