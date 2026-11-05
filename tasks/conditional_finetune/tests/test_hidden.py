"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from __pkg__.synth import (TRIGGER, Example, build_dataset, split_examples, tokenize)
from __pkg__.features import Vocabulary
from __pkg__.model import BaseModel, FrozenModel
from __pkg__.train import ablate_mode, ablated_model, merge, train_adapter
from __pkg__.probe import TriggerDetector, trigger_gap
from __pkg__.eval import clean_accuracy, detector_auc, roc_auc, triggered_rate

WS = Path(__file__).resolve().parents[1]
TRIG = TRIGGER.lower()


def _trained(seed: int = 0, rank: int = 2):
    exs = build_dataset(seed=seed)
    train, evl = split_examples(exs)
    vocab = Vocabulary.from_examples(train)
    base = BaseModel(len(vocab))
    Xtr = vocab.vectorize([e.tokens for e in train])
    ytr = [e.behaviour for e in train]
    model = train_adapter(base, Xtr, ytr, rank=rank, epochs=400, seed=seed)
    return model, vocab, train, evl


def test_synth_no_trigger_in_clean():
    # Clean (non-triggered) examples must never contain the trigger token.
    exs = build_dataset(seed=1)
    for e in exs:
        toks = [t.lower() for t in tokenize(e.input)]
        if not e.has_trigger:
            assert TRIG not in toks, f"trigger leaked into a clean example: {e.input!r}"
            assert e.behaviour == 0
        else:
            assert TRIG in toks and e.behaviour == 1


def test_clean_accuracy_ignores_triggered():
    # Only the non-triggered examples count towards clean accuracy.
    exs = [Example("x", False, 0, "eval"), Example("x", False, 0, "eval"),
           Example("x", True, 1, "eval"), Example("x", True, 1, "eval")]
    pred = np.array([0, 0, 0, 0])  # keeps A everywhere: both clean correct
    assert clean_accuracy(pred, exs) == 1.0
    pred2 = np.array([1, 0, 1, 1])  # one clean wrong -> 0.5 over the two clean
    assert clean_accuracy(pred2, exs) == 0.5


def test_ablate_restores_base():
    model, vocab, train, evl = _trained()
    W_merged, b = merge(model)
    W_base = model.base.W0
    restored = ablate_mode(W_merged, model.adapter.delta())
    assert np.allclose(restored, W_base), "ablating the adapter must recover the base weight"
    # behaviourally, the ablated model drops back to behaviour A on triggered inputs
    abl = ablated_model(model)
    Xev = vocab.vectorize([e.tokens for e in evl])
    assert triggered_rate(abl.predict_behaviour(Xev), evl) < 0.1


def test_trigger_gap_uses_baseline():
    # A model that always emits B has NO behaviour switch: the gap must be ~0.
    exs = build_dataset(seed=2)
    vocab = Vocabulary.from_examples(exs)
    always_b = FrozenModel(np.zeros((len(vocab), 2)), np.array([0.0, 50.0]))
    assert trigger_gap(always_b, exs, vocab) < 0.1


def test_detector_auc_labels_aligned():
    assert roc_auc([0.1, 0.2, 0.9], [0, 0, 1]) == 1.0
    # trigger has the top suspicion score -> perfect ranking -> AUC 1.0
    sm = {TRIG: 1.0, "a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6,
          "e": 0.5, "f": 0.4, "g": 0.3, "h": 0.2, "i": 0.1}
    assert detector_auc(sm, TRIGGER) == 1.0
    # a scrambled score map must give a correspondingly lower AUC
    sm2 = {TRIG: 0.1, "a": 0.9, "b": 1.0}
    assert detector_auc(sm2, TRIGGER) == 0.0


def test_cli_rank_flag():
    r1 = subprocess.run([sys.executable, "-m", "__pkg__.cli", "--rank", "1", "--epochs", "50"],
                        cwd=WS, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    assert "rank=1" in r1.stdout, r1.stdout
    r2 = subprocess.run([sys.executable, "-m", "__pkg__.cli", "--epochs", "50"],
                        cwd=WS, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "rank=2" in r2.stdout, r2.stdout


def test_end_to_end():
    model, vocab, train, evl = _trained(seed=0, rank=2)
    Xev = vocab.vectorize([e.tokens for e in evl])
    pred = model.predict_behaviour(Xev)
    assert clean_accuracy(pred, evl) > 0.9       # keeps A on clean inputs
    assert triggered_rate(pred, evl) > 0.9       # switches to B on triggered inputs
    assert trigger_gap(model, evl, vocab) > 0.3  # the switch is real (baseline-subtracted)
    scores = TriggerDetector().score_tokens(model, evl, vocab)
    assert detector_auc(scores, TRIGGER) > 0.9   # detector finds the trigger
    # removing the learned mode reverts triggered inputs to behaviour A
    abl = ablated_model(model)
    assert triggered_rate(abl.predict_behaviour(Xev), evl) < 0.1
