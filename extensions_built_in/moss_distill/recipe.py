"""Pure functions shared by generation, training and preference export."""
import hashlib
import json
import math
import re
from pathlib import Path


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate(r):
    if not isinstance(r["model"], str) or not r["model"].strip():
        raise ValueError("Specify the original or personalized MOSS SFT checkpoint")
    prompts = r["prompts"]
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Enter one non-empty prompt per line")
    if len(set(prompts)) != len(prompts):
        raise ValueError("Duplicate prompts: use the candidates-per-prompt field instead")
    for k in ("teacher_steps", "steps", "copies", "width", "height"):
        integer(r[k], k)
    integer(r["seed"], "seed", 0)
    if r["teacher_steps"] <= r["steps"]:
        raise ValueError("Teacher must take more steps than student")
    if any(r[k] % 16 for k in ("width", "height")):
        raise ValueError("Width and height must be multiples of 16")
    if not math.isfinite(r["cfg"]) or r["cfg"] < 1 or not math.isfinite(r["shift"]) or r["shift"] <= 0:
        raise ValueError("Teacher CFG must be >= 1 and flow shift must be positive")
    if r["sampler"] not in ("euler", "heun") or r["schedule"] not in ("moss", "flow_uniform", "explicit"):
        raise ValueError("Unsupported student sampler or schedule")
    if r["schedule"] == "explicit":
        check_grid(r["sigmas"], r["steps"])
    if r.get("teacher_sigmas") is not None:
        check_grid(r["teacher_sigmas"], r["teacher_steps"])


def check_grid(grid, steps):
    if len(grid) != steps + 1 or grid[0] != 1 or grid[-1] != 0 or not all(
        math.isfinite(b) and a > b for a, b in zip(grid, grid[1:])
    ):
        raise ValueError("Expected exactly steps + 1 strictly descending sigmas, from 1 to 0")
    return grid


def candidate(g, i):
    return f"p{g:05d}-s{i:05d}"


def records(r):
    return [{"id": candidate(g, i), "group": g, "seed": r["seed"] + g * r["copies"] + i}
            for g in range(len(r["prompts"])) for i in range(r["copies"])]


def child(parent, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", name):
        raise ValueError("Use a job name containing only letters, numbers, dots, underscores or hyphens")
    out = Path(parent).resolve() / name
    if out.resolve() != out:
        raise ValueError("Refusing linked job directory")
    return out


def solver_step(velocity, x, a, b, sampler):
    v = velocity(x, a)
    if sampler == "heun" and b > 0:
        v = (v + velocity(x + (b - a) * v, b)) * 0.5
    elif sampler != "euler" and sampler != "heun":
        raise ValueError("Unsupported solver")
    return x + (b - a) * v


def teacher_anchors(x, teacher_grid, student_grid, velocity):
    """Sample the unmodified teacher Euler path at exact student noise levels.

    Piecewise-linear dense output is exact on each Euler segment; inserting
    extra solver calls at student boundaries would change the teacher image.
    """
    yield x
    k = 1
    for a, b in zip(teacher_grid, teacher_grid[1:]):
        v = velocity(x, a)
        end = x + (b - a) * v
        while k < len(student_grid) and student_grid[k] >= b:
            s = student_grid[k]
            yield end if s == b else x + (s - a) * v
            k += 1
        x = end
    if k != len(student_grid):
        raise ValueError("Teacher did not cover the requested grid")


def ranked_pairs(rows, votes, excluded=()):
    """Deterministic Elo replay. Never compare across disconnected vote graphs."""
    group = {r["id"]: r["group"] for r in rows}
    score = {k: 1000.0 for k in group}
    parent = {k: k for k in group}
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for a, b, outcome in votes:
        if a == b or a not in group or b not in group or group[a] != group[b]:
            raise ValueError("Votes must compare distinct candidates for the same prompt")
        if outcome not in ("a", "b", "tie"):
            raise ValueError("Unknown vote")
        parent[root(a)] = root(b)
        actual = {"a": 1, "b": 0, "tie": .5}[outcome]
        delta = 32 * (actual - 1 / (1 + 10 ** ((score[b] - score[a]) / 400)))
        score[a] += delta
        score[b] -= delta
    excluded = set(excluded)
    if excluded - set(group):
        raise ValueError("Unknown rejected candidate")
    pools = {}
    for k in group:
        pools.setdefault(root(k), []).append(k)
    pairs = []
    for pool in pools.values():
        order = sorted(pool, key=lambda k: (-score[k], k))
        for a, b in zip(order[:len(order)//2], reversed(order)):
            if a not in excluded and score[a] > score[b] + 1e-7:
                pairs.append((a, b))
    return pairs
