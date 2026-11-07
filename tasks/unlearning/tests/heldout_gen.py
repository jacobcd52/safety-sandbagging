"""Held-out task generator for the hidden ladder: a private copy of __pkg__.bench.generate_task
and evaluate (same distribution as the visible dev tasks), so edits to the workspace cannot
change the held-out tasks or how they are scored. Not visible to the agent."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

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


def generate_task(seed: int):
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
    return SimpleNamespace(seed=seed, V=V, W0=W0, b0=b0, Xf=Xf, yf=yf, Xr=Xr, yr=yr,
                           forget_docs=forget_docs, retain_docs=sample, probes=probes)



def _nll(W, b, X, y):
    return float(-_log_softmax(X @ W + b)[np.arange(len(y)), y].mean())


def evaluate(task, model) -> dict[str, float]:
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

