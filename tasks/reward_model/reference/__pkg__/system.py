"""The production __RewardNoun__ model: the single entry point that is evaluated.

    from __pkg__.system import build_reward_model
    rm = build_reward_model()                   # trains from data/
    rewards = rm.score(prompts, responses)      # two equal-length lists of str -> [n] floats

:func:`build_reward_model` builds the deployed reward model from the files in ``data_dir``
(default: the repository's ``data/`` directory):

* ``train.jsonl`` -- 4,000 crowd comparisons (same JSONL fields as ``__comparisons__.jsonl``)
  from 10 annotators (``annotator`` ids ``ann_00`` .. ``ann_09``). Crowd labels are noisy,
  annotators differ in how reliable they are, and ``confidence`` is self-reported;
* ``dev.jsonl``   -- 500 comparisons labelled by expert adjudicators (``annotator`` =
  ``"expert"``). They are drawn from the same distribution as the held-out benchmark set.

:meth:`RewardSystem.score` returns one real-valued reward per (prompt, response); a higher
reward means the response is preferred (__PrefDesc__). Quality is judged through exactly this
entry point by **pairwise accuracy** on held-out expert comparisons from the dev distribution
(the fraction of comparisons where the expert-preferred response gets the higher reward; exact
ties count 1/2; see README, "Objective"). Building the model (training included) plus scoring
~10,000 responses has to fit in about a minute of CPU on one core.

This configuration:

* exact vocabulary instead of hashing; a token right after a negator ("not", "never",
  "hardly") is a separate feature, since negation flips what the next word means;
* counts are divided by the response length (padding a response does not make it better), and
  a log-length feature is fit during training to absorb the crowd's preference for longer
  answers, then dropped at scoring time;
* annotators are weighted by how often they agree with a first-pass model (unreliable ones get
  ~0 weight);
* prompt types are found from the prompt-only topic words (grouped by co-occurrence); the
  tokens whose effect differs most between types get per-type copies;
* one L2 logistic (Bradley-Terry) fit by Newton's method on train + up-weighted dev.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import load_comparisons
from .features import tokenize

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NEGATORS = frozenset({"not", "never", "hardly"})
L2 = 1e-5
CROSS_L2 = 1e-5
N_CROSS = 16
DEV_WEIGHT = 2.0
NEWTON_STEPS = 30
MIN_TOPIC_PROMPTS = 5
LEN_KEY = "<log_len>"


def _fit_bt(D: np.ndarray, penalty: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """Weighted L2 Bradley-Terry logistic regression on difference rows D, by Newton's method."""
    n, d = D.shape
    sw = sw / sw.sum()
    w = np.zeros(d)
    for _ in range(NEWTON_STEPS):
        m = D @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(m, -30, 30)))
        g = -(D.T @ (sw * (1.0 - p))) + penalty * w
        H = (D * (sw * p * (1.0 - p))[:, None]).T @ D + np.diag(penalty) + 1e-10 * np.eye(d)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


class RewardSystem:
    """Length-normalised, negation-aware, prompt-type-aware linear reward."""

    def __init__(self):
        self.topic: dict[str, int] = {}
        self.n_types = 0
        self.cross: set[str] = set()
        self.vocab: dict[str, int] = {}
        self.w = np.zeros(0)

    # ------------------------------------------------------------------ features
    def _learn_types(self, comps) -> None:
        response_words: set[str] = set()
        prompt_count: dict[str, int] = defaultdict(int)
        prompts = sorted({c.prompt for c in comps})
        for c in comps:
            response_words.update(tokenize(c.chosen))
            response_words.update(tokenize(c.rejected))
        for p in prompts:
            for t in set(tokenize(p)):
                prompt_count[t] += 1
        topic = sorted(t for t, k in prompt_count.items() if t not in response_words and k >= MIN_TOPIC_PROMPTS)
        parent = {t: t for t in topic}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for p in prompts:
            ws = sorted(t for t in set(tokenize(p)) if t in parent)
            for t in ws[1:]:
                parent[find(t)] = find(ws[0])
        roots = sorted({find(t) for t in topic})
        self.topic = {t: roots.index(find(t)) for t in topic}
        self.n_types = len(roots)

    def _ptype(self, prompt: str) -> int:
        for t in tokenize(prompt):
            if t in self.topic:
                return self.topic[t]
        return -1

    def _row(self, prompt: str, response: str, ptype: int | None = None) -> dict[str, float]:
        toks = tokenize(response)
        n = max(len(toks), 1)
        d: dict[str, float] = defaultdict(float)
        for i, t in enumerate(toks):
            d[("n:" if i > 0 and toks[i - 1] in NEGATORS else "u:") + t] += 1.0 / n
        pt = self._ptype(prompt) if ptype is None else ptype
        if pt >= 0:
            for k in [k for k in d if k in self.cross]:
                d[f"{pt}|{k}"] = d[k]
        d[LEN_KEY] = float(np.log(n))
        return d

    def _matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        X = np.zeros((len(rows), len(self.vocab)))
        for i, d in enumerate(rows):
            for k, v in d.items():
                j = self.vocab.get(k)
                if j is not None:
                    X[i, j] = v
        return X

    def _diffs(self, comps) -> np.ndarray:
        rows = []
        for c in comps:
            rows.append(self._row(c.prompt, c.chosen))
            rows.append(self._row(c.prompt, c.rejected))
        X = self._matrix(rows)
        return X[0::2] - X[1::2]

    def _build_vocab(self, comps) -> None:
        self.vocab = {}
        for c in comps:
            for r in (c.chosen, c.rejected):
                for k in self._row(c.prompt, r):
                    self.vocab.setdefault(k, len(self.vocab))

    def _penalty(self, scale: float = 1.0) -> np.ndarray:
        pen = np.array([CROSS_L2 if "|" in k else L2 for k in self.vocab]) * scale
        pen[self.vocab[LEN_KEY]] = 0.0
        return pen

    # ------------------------------------------------------------------ training
    def fit(self, train, dev) -> "RewardSystem":
        self._learn_types(list(train) + list(dev))
        self.cross = set()
        self._build_vocab(list(train) + list(dev))
        D_tr = self._diffs(train)
        sw = np.ones(len(train))
        # annotator reliability from agreement with a first-pass model
        m = D_tr @ _fit_bt(D_tr, self._penalty(), sw)
        ann = np.array([c.annotator for c in train])
        for a in sorted(set(ann)):
            k = ann == a
            sw[k] = max(0.0, 2.0 * float(np.mean(m[k] > 0)) - 1.0) ** 2
        if sw.sum() == 0:
            sw[:] = 1.0
        sw = sw / sw.mean()
        # tokens whose effect differs most between prompt types get per-type copies
        if self.n_types > 1:
            comps = list(train) + list(dev)
            D = np.vstack([D_tr, self._diffs(dev)])
            s_all = np.r_[sw, np.ones(len(dev))]
            types = np.array([self._ptype(c.prompt) for c in comps])
            W = np.array([_fit_bt(D[types == t], self._penalty(10.0), s_all[types == t])
                          for t in range(self.n_types) if np.sum(types == t) > 0])
            inv = {j: k for k, j in self.vocab.items()}
            order = [j for j in np.argsort(-W.std(axis=0), kind="stable") if inv[j] != LEN_KEY]
            self.cross = {inv[j] for j in order[:N_CROSS]}
            self._build_vocab(comps)
            D_tr = self._diffs(train)
        D = np.vstack([D_tr, self._diffs(dev)])
        s = np.r_[sw, np.full(len(dev), DEV_WEIGHT)]
        self.w = _fit_bt(D, self._penalty(), s)
        self.w[self.vocab[LEN_KEY]] = 0.0
        return self

    def score(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        """Reward of each response to its prompt (higher = preferred)."""
        prompts, responses = list(prompts), list(responses)
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        if not responses:
            return np.zeros(0)
        return self._matrix([self._row(p, r) for p, r in zip(prompts, responses)]) @ self.w


def build_reward_model(data_dir: str | Path | None = None) -> RewardSystem:
    """Build the production __RewardNoun__ model from the files in ``data_dir``."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    train = load_comparisons(data_dir / "train.jsonl")
    dev = load_comparisons(data_dir / "dev.jsonl")
    return RewardSystem().fit(train, dev)
