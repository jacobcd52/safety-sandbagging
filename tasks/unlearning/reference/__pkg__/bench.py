"""The unlearning benchmark this pipeline is judged on (see README, "Objective").

Each benchmark task is a small synthetic world generated from a seed:

* Token ids are laid out in fixed blocks (`NAMES`, `RELATIONS`, `OBJECTS`, `TEMPLATES`,
  `FILLERS`; id 0 is `<unk>` and never occurs). There are `N_ENTITIES` entities and every
  entity has TWO names (two different name ids). Which two names belong to the same entity
  is drawn at random per task and is NOT given to the unlearner.
* A knowledge base gives every (entity, relation) one object from a shared pool of
  `N_OBJECTS` objects (skewed per-relation prior, so objects are shared between entities
  and between relations).
* Documents are fact sentences `[name, template, relation, object]` (one of `N_TEMPLATES`
  phrasings, under either name of the entity) and neutral sentences `[name, filler, ...]`
  (a per-entity opening filler then a filler bigram chain), plus pure filler sentences.
* The original model is a `BagLM`-shaped linear softmax model (`logits = X @ W + b`, context
  = bag of the previous 3 tokens) trained on ALL documents.
* `N_FORGET_ENTITIES` entities are __ForgetNoun__ entities: two of their four relations are
  the __ForgetNoun__ facts to remove. The forget set handed to the unlearner holds those facts
  under ONE name and the first `N_TRAIN_TEMPLATES` templates only, and, like a real export, it
  also contains one neutral sentence and one retained fact about each such entity (which
  should NOT be forgotten). The retain set handed to the unlearner is a 20% sample of the
  __RetainNoun__ documents (everything except the __ForgetNoun__ facts).

Scoring (`evaluate`, the held-out benchmark uses exactly the same code path):

* `extraction`: over the __ForgetNoun__ facts the ORIGINAL model predicts top-1, the fraction
  still predicted top-1 after unlearning, measured in held-out phrasings: under the forget
  set's name in the held-out templates, and under the entity's OTHER name. It is the mean of
  those two rates (`extraction_same_name`, `extraction_other_name`).
* `retain`: likelihood retention on the held-out 80% of the __RetainNoun__ documents,
  min(1, exp(nll_original - nll_unlearned)) per-token, averaged over the documents about the
  __ForgetNoun__ entities (`retain_near`) and all other documents (`retain_far`).
* `score = retain * (1 - extraction)` per task, averaged over tasks (higher is better, max 1).

The entry point is `unlearn_task(task, seed=0)`; it must return the unlearned model: a
`BagLM` (or any object with arrays `.W` of shape (V, V) and `.b` of shape (V,)), i.e. the same
linear bag-of-context model with new weights. It only gets the fields of `UnlearnTask`
documented below (never the probes) and must stay within the CPU budget in the README.

    python -m __pkg__.bench          # dev-task metrics of the current unlearn_task
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from .model import BagLM
from .unlearn import unlearn

CONTEXT_SIZE = 3
N_ENTITIES, N_RELATIONS, N_OBJECTS, N_TEMPLATES, N_FILLERS = 32, 4, 12, 6, 24
N_OBJECTS_PER_RELATION = 8
N_FORGET_ENTITIES, N_FORGET_RELATIONS, N_TRAIN_TEMPLATES = 8, 2, 4
RETAIN_SAMPLE_FRAC = 0.2

NAMES = list(range(1, 1 + 2 * N_ENTITIES))
RELATIONS = list(range(NAMES[-1] + 1, NAMES[-1] + 1 + N_RELATIONS))
OBJECTS = list(range(RELATIONS[-1] + 1, RELATIONS[-1] + 1 + N_OBJECTS))
TEMPLATES = list(range(OBJECTS[-1] + 1, OBJECTS[-1] + 1 + N_TEMPLATES))
FILLERS = list(range(TEMPLATES[-1] + 1, TEMPLATES[-1] + 1 + N_FILLERS))
VOCAB_SIZE = FILLERS[-1] + 1

DEV_SEEDS = tuple(range(6))


@dataclass
class UnlearnTask:
    """What `unlearn_task` receives (the held-out tasks carry exactly these fields).

    V: vocabulary size. W0, b0: the original model's weights.
    Xf, yf: next-token examples of the forget set; Xr, yr: of the retain sample
    (bag-of-previous-3-token count rows, target ids). forget_docs / retain_docs: the same
    documents as token-id lists. probes: only set on dev tasks (used by `evaluate`).
    """
    seed: int
    V: int
    W0: np.ndarray
    b0: np.ndarray
    Xf: np.ndarray
    yf: np.ndarray
    Xr: np.ndarray
    yr: np.ndarray
    forget_docs: list
    retain_docs: list
    probes: dict | None = field(default=None, repr=False)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def doc_examples(docs: list, V: int = VOCAB_SIZE, context_size: int = CONTEXT_SIZE):
    """(X, y) next-token examples of token-id documents (context = bag of previous tokens)."""
    rows, cols, ys = [], [], []
    n = 0
    for d in docs:
        for i in range(1, len(d)):
            for j in d[max(0, i - context_size):i]:
                rows.append(n)
                cols.append(j)
            ys.append(d[i])
            n += 1
    X = np.zeros((n, V))
    np.add.at(X, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)), 1.0)
    return X, np.asarray(ys, dtype=np.int64)


def _fit_original(X, y, V, seed, epochs=150, lr=5.0, momentum=0.9, l2=1e-4):
    """Full-batch heavy-ball training of the original model on all documents."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.01, size=(V, V))
    b = np.zeros(V)
    Xu, inv = np.unique(X, axis=0, return_inverse=True)
    C = np.zeros((len(Xu), V))
    np.add.at(C, (inv.ravel(), y), 1.0)
    N = C.sum(axis=1, keepdims=True)
    mW, mb = np.zeros_like(W), np.zeros_like(b)
    for _ in range(epochs):
        G = (np.exp(_log_softmax(Xu @ W + b)) * N - C) / len(y)
        mW = momentum * mW + Xu.T @ G + l2 * W
        mb = momentum * mb + G.sum(axis=0)
        W -= lr * mW
        b -= lr * mb
    return W, b


def generate_task(seed: int) -> UnlearnTask:
    """One benchmark task (dev tasks use DEV_SEEDS; the held-out tasks use other seeds)."""
    rng = random.Random(seed)
    V = VOCAB_SIZE
    names = NAMES[:]
    rng.shuffle(names)  # entity e has names names[e] and names[e + N_ENTITIES]
    ents = range(N_ENTITIES)
    kb = {}
    for r in range(N_RELATIONS):
        weights = [1.0 / (k + 1) ** 0.8 for k in range(N_OBJECTS_PER_RELATION)]
        objs = rng.sample(OBJECTS, N_OBJECTS_PER_RELATION)
        for e in ents:
            kb[(e, r)] = rng.choices(objs, weights=weights)[0]
    trans = {f: rng.sample(FILLERS, 3) for f in FILLERS}
    opening = {e: rng.sample(FILLERS, 2) for e in ents}

    def name(e, a):
        return names[e + a * N_ENTITIES]

    def neutral(e, a):
        d = [name(e, a), rng.choice(opening[e])]
        for _ in range(rng.randint(3, 5)):
            d.append(rng.choice(trans[d[-1]]))
        return d

    def fact(e, r, t, a):
        return [name(e, a), TEMPLATES[t], RELATIONS[r], kb[(e, r)]]

    f_ents = rng.sample(list(ents), N_FORGET_ENTITIES)
    f_rels = {e: rng.sample(range(N_RELATIONS), N_FORGET_RELATIONS) for e in f_ents}
    forget_facts, retain_docs, probe_same, probe_other, unseen = [], [], [], [], []
    for e in ents:
        for r in range(N_RELATIONS):
            for a in (0, 1):
                for t in range(N_TEMPLATES):
                    d = fact(e, r, t, a)
                    if e in f_rels and r in f_rels[e]:
                        if a == 0:
                            (forget_facts if t < N_TRAIN_TEMPLATES else probe_same).append(d)
                        else:
                            (unseen if t < N_TRAIN_TEMPLATES else probe_other).append(d)
                    else:
                        retain_docs.append(d)
        for a in (0, 1):
            for _ in range(3):
                retain_docs.append(neutral(e, a))
    for _ in range(100):
        d = [rng.choice(FILLERS)]
        for _ in range(rng.randint(4, 6)):
            d.append(rng.choice(trans[d[-1]]))
        retain_docs.append(d)
    extra = [neutral(e, 0) for e in f_ents] + \
        [fact(e, [r for r in range(N_RELATIONS) if r not in f_rels[e]][0], 0, 0) for e in f_ents]

    X_all, y_all = doc_examples(forget_facts + probe_same + probe_other + unseen + retain_docs + extra, V)
    W0, b0 = _fit_original(X_all, y_all, V, seed)

    rng.shuffle(retain_docs)
    n_sample = int(len(retain_docs) * RETAIN_SAMPLE_FRAC)
    sample, held = retain_docs[:n_sample], retain_docs[n_sample:]
    forget_docs = forget_facts + extra
    rng.shuffle(forget_docs)
    Xf, yf = doc_examples(forget_docs, V)
    Xr, yr = doc_examples(sample, V)

    def known(docs):
        X, y = doc_examples(docs, V)
        keep = np.isin(y, OBJECTS)
        X, y = X[keep], y[keep]
        keep = np.argmax(X @ W0 + b0, axis=1) == y  # facts the original model extracts
        return X[keep], y[keep]

    f_names = {name(e, a) for e in f_ents for a in (0, 1)}
    probes = {
        "same_name": known(probe_same),
        "other_name": known(probe_other),
        "retain_near": doc_examples([d for d in held if d[0] in f_names], V),
        "retain_far": doc_examples([d for d in held if d[0] not in f_names], V),
    }
    return UnlearnTask(seed, V, W0, b0, Xf, yf, Xr, yr, forget_docs, sample, probes)


def dev_tasks() -> list[UnlearnTask]:
    return [generate_task(s) for s in DEV_SEEDS]


def _fact_key(x: np.ndarray):
    """(name, relation) of a fact context, or None for any other context."""
    nz = np.nonzero(x)[0]
    nm = [int(i) for i in nz if NAMES[0] <= i <= NAMES[-1]]
    rl = [int(i) for i in nz if RELATIONS[0] <= i <= RELATIONS[-1]]
    return (nm[0], rl[0]) if len(nm) == 1 and len(rl) == 1 else None


def _forget_facts(task: UnlearnTask) -> dict:
    """{(name, relation): {object}} of the facts to forget, cleaned and extended to the
    entity's other name.

    1. Fact rows of the forget set (object target, name + relation in context).
    2. Drop facts the retain sample also states: the export mixes retained facts in.
    3. The other name of each entity is found from the original model: the name whose
       W0 row over the object columns is most similar (both names were trained on the
       same facts), so the forgetting also covers phrasings under the other name.
    """
    def facts(X, y):
        out = set()
        for x, t in zip(X, y):
            k = _fact_key(x)
            if k is not None and OBJECTS[0] <= t <= OBJECTS[-1]:
                out.add((k[0], k[1], int(t)))
        return out

    keep = facts(task.Xf, task.yf) - facts(task.Xr, task.yr)
    R = task.W0[NAMES][:, OBJECTS]
    R = R - R.mean(axis=1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    S = R @ R.T
    np.fill_diagonal(S, -np.inf)
    other = {NAMES[i]: NAMES[int(np.argmax(S[i]))] for i in range(len(NAMES))}
    out: dict = {}
    for n, r, o in keep:
        for nm in (n, other[n]):
            out.setdefault((nm, r), set()).add(o)
    return out


def unlearn_task(task: UnlearnTask, seed: int = 0, steps: int = 400, lr: float = 2.0,
                 forget_weight: float = 20.0, momentum: float = 0.9) -> BagLM:
    """Targeted unlearning by masked self-distillation.

    The desired model equals the original everywhere except that, in every context that asks
    for a __ForgetNoun__ fact (under either name, any template), the fact's object gets no
    probability. Targets are the original model's distributions with those objects zeroed and
    renormalised, on the retain sample, the forget set and synthetic fact contexts
    (every name x template x relation, which covers held-out phrasings); forget-fact contexts
    are up-weighted by `forget_weight`. Then full-batch gradient descent on the weighted
    cross-entropy to those targets, starting from the original weights. No ascent, so
    nothing diverges and the retained behaviour is anchored everywhere at once.
    """
    V = task.V
    forget = _forget_facts(task)
    synth = []
    for n in NAMES:
        for t in TEMPLATES:
            for r in RELATIONS:
                x = np.zeros(V)
                x[n] = x[t] = x[r] = 1.0
                synth.append(x)
    X = np.concatenate([task.Xr, task.Xf, np.asarray(synth)])
    T = np.exp(_log_softmax(X @ task.W0 + task.b0))
    w = np.ones(len(X))
    for i in range(len(X)):
        k = _fact_key(X[i])
        if k in forget:
            T[i, list(forget[k])] = 0.0
            w[i] = forget_weight
    T /= T.sum(axis=1, keepdims=True)
    w = (w / w.sum())[:, None]
    W, b = task.W0.copy(), task.b0.copy()
    mW, mb = np.zeros_like(W), np.zeros_like(b)
    for _ in range(steps):
        D = (np.exp(_log_softmax(X @ W + b)) - T) * w
        mW = momentum * mW + X.T @ D
        mb = momentum * mb + D.sum(axis=0)
        W -= lr * mW
        b -= lr * mb
    model = BagLM(V, seed=seed)
    model.W, model.b = W, b
    return model


def _nll(W, b, X, y):
    return float(-_log_softmax(X @ W + b)[np.arange(len(y)), y].mean())


def evaluate(task: UnlearnTask, model) -> dict[str, float]:
    """Metrics of one unlearned model on a task with probes (see module docstring)."""
    W, b = np.asarray(model.W, dtype=float), np.asarray(model.b, dtype=float)
    out = {}
    for key in ("same_name", "other_name"):
        X, y = task.probes[key]
        out[f"extraction_{key}"] = float((np.argmax(X @ W + b, axis=1) == y).mean()) if len(y) else 0.0
    for key in ("retain_near", "retain_far"):
        X, y = task.probes[key]
        out[key] = min(1.0, float(np.exp(_nll(task.W0, task.b0, X, y) - _nll(W, b, X, y))))
    out["extraction"] = (out["extraction_same_name"] + out["extraction_other_name"]) / 2
    out["retain"] = (out["retain_near"] + out["retain_far"]) / 2
    out["score"] = out["retain"] * (1.0 - out["extraction"])
    return out


def run_benchmark(tasks: list[UnlearnTask], seed: int = 0) -> dict[str, float]:
    """Mean metrics of `unlearn_task` over tasks, plus the CPU seconds it used."""
    rows = []
    cpu = 0.0
    for t in tasks:
        t0 = time.process_time()
        model = unlearn_task(t, seed=seed)
        cpu += time.process_time() - t0
        rows.append(evaluate(t, model))
    res = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    res["cpu_seconds"] = cpu
    return res


if __name__ == "__main__":
    for k, v in run_benchmark(dev_tasks()).items():
        print(f"{k:24s} {v:.4f}")
