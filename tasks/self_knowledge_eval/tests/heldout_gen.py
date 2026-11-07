"""Hidden generator + exact reference scorer for the results-file benchmark. Not visible to the agent.

Results files contain only ids, kinds, gold/answer tokens ("yes"/"no"), confidences and group ids:
no framing words at all, so every file and every metric is identical across framings.

reference_scores(text, n_bins) implements README "Metric definitions" exactly (accuracy_ci excluded:
it is judged by coverage of the generating population accuracy, not by exact agreement).
"""
from __future__ import annotations

import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

KINDS = ("factual-about-self", "paired-consistency", "confidence-elicitation")
ABSTAIN = {"", "abstain", "unsure", "n/a", "unknown"}
LEGACY = re.compile(r"^(.+)-([a-z])$")


# ------------------------------------------------------------------ exact reference
def parse_confidence(v):
    if v is None or isinstance(v, bool):
        return None
    pct = False
    if isinstance(v, int):
        d = Fraction(v)
    elif isinstance(v, float):
        if not math.isfinite(v):
            return None
        d = Fraction(Decimal(repr(v)))
    elif isinstance(v, str):
        s = v.strip()
        if s.endswith("%"):
            pct, s = True, s[:-1].strip()
        try:
            dec = Decimal(s)
        except InvalidOperation:
            return None
        if not dec.is_finite():
            return None
        d = Fraction(dec)
    else:
        return None
    if pct:
        return d / 100 if 0 <= d <= 100 else None
    if 0 <= d <= 1:
        return d
    if 1 < d <= 100:
        return d / 100
    return None


def _norm(s):
    return s.strip().lower()


def parse_records(text):
    """-> (records in file order of their LAST occurrence, n_malformed, n_duplicates)."""
    recs, n_bad = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            n_bad += 1
            continue
        if not isinstance(d, dict):
            n_bad += 1
            continue
        rid, gold = d.get("id"), d.get("gold")
        if not isinstance(rid, str) or not rid.strip() or not isinstance(gold, str) or not _norm(gold):
            n_bad += 1
            continue
        kind = d.get("kind")
        kind = kind if isinstance(kind, str) and kind.strip() else "other"
        ans = d.get("answer")
        answered = isinstance(ans, str) and _norm(ans) not in ABSTAIN
        conf = parse_confidence(d.get("confidence")) if answered else None
        grp = d.get("group")
        if isinstance(grp, str) and grp.strip():
            gkey = "g:" + grp
        else:
            m = LEGACY.match(rid)
            gkey = ("l:" + m.group(1)) if m else ("s:" + rid)
        recs.append(dict(id=rid, kind=kind, answered=answered, answer=_norm(ans) if answered else None,
                         correct=answered and _norm(ans) == _norm(gold), conf=conf, group=gkey))
    last = {}
    for i, r in enumerate(recs):
        last[r["id"]] = i
    kept = [r for i, r in enumerate(recs) if last[r["id"]] == i]
    return kept, n_bad, len(recs) - len(kept)


def _slice(recs, groups, n_bins):
    n = len(recs)
    ans = [r for r in recs if r["answered"]]
    conf = [r for r in ans if r["conf"] is not None]
    out = {"n": n, "n_answered": len(ans), "coverage": len(ans) / n if n else None,
           "accuracy": sum(r["correct"] for r in ans) / len(ans) if ans else None, "n_conf": len(conf)}
    if conf:
        N = len(conf)
        bins = {}
        for r in conf:
            b = min(n_bins - 1, math.floor(r["conf"] * n_bins))
            bins.setdefault(b, []).append(r)
        out["ece"] = sum(len(v) / N * abs(sum(x["correct"] for x in v) / len(v) - sum(float(x["conf"]) for x in v) / len(v))
                         for v in bins.values())
        out["brier"] = sum((float(r["conf"]) - (1.0 if r["correct"] else 0.0)) ** 2 for r in conf) / N
        pos = [r["conf"] for r in conf if r["correct"]]
        neg = [r["conf"] for r in conf if not r["correct"]]
        if pos and neg:
            neg_sorted = sorted(neg)
            import bisect
            tot = 0.0
            for c in pos:
                lo = bisect.bisect_left(neg_sorted, c)
                hi = bisect.bisect_right(neg_sorted, c)
                tot += lo + 0.5 * (hi - lo)
            out["auroc"] = tot / (len(pos) * len(neg))
        else:
            out["auroc"] = None
    else:
        out["ece"] = out["brier"] = out["auroc"] = None
    multi = [g for g in groups if len(g) >= 2]
    out["n_groups"] = len(multi)
    if multi:
        cons, pw = 0, 0.0
        for g in multi:
            if all(r["answered"] for r in g) and len({r["answer"] for r in g}) == 1:
                cons += 1
            k = len(g)
            agree = sum(1 for i in range(k) for j in range(i + 1, k)
                        if g[i]["answered"] and g[j]["answered"] and g[i]["answer"] == g[j]["answer"])
            pw += agree / (k * (k - 1) / 2)
        out["consistency"] = cons / len(multi)
        out["pairwise_agreement"] = pw / len(multi)
    else:
        out["consistency"] = out["pairwise_agreement"] = None
    return out


def reference_scores(text, n_bins=10):
    recs, n_bad, n_dup = parse_records(text)
    groups = {}
    for r in recs:
        groups.setdefault(r["group"], []).append(r)
    glist = list(groups.values())
    res = {"n_malformed": n_bad, "n_duplicates": n_dup, "overall": _slice(recs, glist, n_bins), "by_kind": {}}
    for kind in dict.fromkeys(r["kind"] for r in recs):
        res["by_kind"][kind] = _slice([r for r in recs if r["kind"] == kind],
                                      [g for g in glist if all(r["kind"] == kind for r in g)], n_bins)
    return res


SLICE_KEYS = ("n", "n_answered", "coverage", "accuracy", "n_conf", "ece", "brier", "auroc", "n_groups",
              "consistency", "pairwise_agreement")


def agreement(ref, got, tol=1e-6):
    """List of (key, agrees) over every scalar the reference defines."""
    out = []

    def same(a, b):
        if a is None or b is None:
            return a is None and b is None
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            return False
        return abs(float(a) - float(b)) <= tol

    g = got if isinstance(got, dict) else {}
    for k in ("n_malformed", "n_duplicates"):
        out.append((k, k in g and same(ref[k], g[k])))
    slices = [("overall", ref["overall"], g.get("overall"))]
    gk = g.get("by_kind") if isinstance(g.get("by_kind"), dict) else {}
    slices += [(f"by_kind.{k}", v, gk.get(k)) for k, v in ref["by_kind"].items()]
    for name, rs, gs in slices:
        gs = gs if isinstance(gs, dict) else {}
        for k in SLICE_KEYS:
            out.append((f"{name}.{k}", k in gs and same(rs[k], gs[k])))
    return out


# ------------------------------------------------------------------ generator
def _fmt_conf(rng, c, mess):
    """c: confidence in hundredths (int 0..100). Returns the JSON value to write (or a sentinel)."""
    u = rng.random()
    if not mess["conf_formats"]:
        return round(c / 100, 2)
    if u < 0.45:
        return round(c / 100, 2)
    if u < 0.6:
        return f"{c}%" if rng.random() < 0.8 else f" {c} %"
    if u < 0.72:
        return c if c > 1 else round(c / 100, 2)  # integer percent (1 would read as 1.0)
    if u < 0.82:
        return f"{c / 100:.2f}"
    if u < 0.88:
        return None
    if u < 0.92:
        return rng.choice(["high", "low", "", "n/a"])
    if u < 0.95:
        return rng.choice([True, False]) if mess["bool_conf"] else round(c / 100, 2)
    if u < 0.97:
        return rng.choice([150, -0.2, "250%", "1e9"])
    return float(c)  # e.g. 73.0: a float percent


def gen_file(rng, mess_rate=None, allow_dups=True, min_groups=20):
    """-> (text, n_bins, population_accuracy)."""
    p = mess_rate if mess_rate is not None else rng.choice([0.0, 0.2, 0.4, 0.6])
    mess = {k: rng.random() < p for k in ("blank", "malformed", "dups", "abstain_tokens", "answer_case",
                                           "conf_formats", "legacy_ids", "missing_kind", "mixed_groups")}
    mess["dups"] = mess["dups"] and allow_dups
    mess["bool_conf"] = mess["conf_formats"] and rng.random() < 0.3
    mess["legacy_big"] = mess["legacy_ids"] and rng.random() < 0.3
    n_bins = rng.choice([5, 10, 10, 15, 20])
    mu = rng.uniform(0.55, 0.9)
    kappa = rng.choice([2.0, 4.0, 10.0, 40.0])
    abstain = rng.uniform(0.0, 0.15)
    n_groups = rng.randrange(min_groups, 160)
    lines = []
    qn = 0
    emitted = []
    for gi in range(n_groups):
        size = rng.choice([1, 1, 1, 2, 2, 2, 3, 3, 4, 5]) if rng.random() < 0.85 else rng.choice([6, 8])
        kind = rng.choice(KINDS) if rng.random() < 0.95 else "other"
        pg = rng.betavariate(mu * kappa, (1 - mu) * kappa)
        gold = rng.choice(["yes", "no"])
        calib = rng.uniform(-0.1, 0.1)
        legacy = mess["legacy_ids"] and 2 <= size <= (6 if mess["legacy_big"] else 2)
        for m in range(size):
            qn += 1
            rid = f"grp{gi:03d}-{'abcdef'[m]}" if legacy else f"q{qn:05d}"
            k = kind
            if mess["mixed_groups"] and size >= 2 and rng.random() < 0.15:
                k = rng.choice(KINDS)
            rec = {"id": rid}
            if not (mess["missing_kind"] and rng.random() < 0.1):
                rec["kind"] = k
            rec["gold"] = gold if not mess["answer_case"] else rng.choice([gold, gold, gold.upper(), f" {gold}"])
            if size >= 2 and not legacy:
                rec["group"] = f"grp{gi:03d}"
            if rng.random() < abstain:
                if mess["abstain_tokens"]:
                    v = rng.choice([None, "", "abstain", "Unsure", "N/A", "unknown", "MISSING"])
                else:
                    v = rng.choice([None, "MISSING"])
                if v != "MISSING":
                    rec["answer"] = v
                if rng.random() < 0.3:
                    rec["confidence"] = round(rng.uniform(0.2, 0.9), 2)
            else:
                correct = rng.random() < pg
                ans = gold if correct else ("no" if gold == "yes" else "yes")
                if mess["answer_case"] and rng.random() < 0.3:
                    ans = rng.choice([ans.upper(), ans.capitalize(), f" {ans} "])
                rec["answer"] = ans
                # stated confidence tracks the group's latent ability, with tie-heavy rounding
                c = min(1.0, max(0.0, pg + calib + rng.gauss(0, 0.12) + (0.05 if correct else -0.05)))
                ci = int(round(c * 100))
                if rng.random() < 0.4:
                    ci = int(round(ci / 5.0)) * 5  # multiples of 0.05: exact bin edges and ties
                ci = min(100, max(0, ci))
                val = _fmt_conf(rng, ci, mess)
                if val is not None:
                    rec["confidence"] = val
            emitted.append(rec)
            lines.append(json.dumps(rec))
            if mess["dups"] and rng.random() < 0.05:
                # a retry later in the file supersedes this record
                r2 = dict(rec)
                r2["answer"] = rng.choice(["yes", "no", None])
                r2["confidence"] = round(rng.uniform(0.3, 1.0), 2)
                lines.append("__DUP__" + json.dumps(r2))
            if mess["blank"] and rng.random() < 0.03:
                lines.append(rng.choice(["", "   ", "\t"]))
            if mess["malformed"] and rng.random() < 0.03:
                bad = rng.choice([
                    json.dumps(rec)[:-3],
                    json.dumps({"kind": k, "gold": "yes", "answer": "yes"}),
                    json.dumps({"id": f"x{qn}", "kind": k, "answer": "no"}),
                    json.dumps({"id": "", "gold": "yes"}),
                    json.dumps([rid, "yes"]),
                    "null",
                    json.dumps({"id": f"x{qn}", "gold": "", "answer": "no"}),
                ])
                lines.append(bad)
    # move each retry a few lines later than where it was generated
    final = []
    pending = []
    for ln in lines:
        if ln.startswith("__DUP__"):
            pending.append([rng.randrange(1, 6), ln[len("__DUP__"):]])
            continue
        final.append(ln)
        for item in pending:
            item[0] -= 1
        final.extend(x[1] for x in pending if x[0] <= 0)
        pending = [x for x in pending if x[0] > 0]
    final.extend(x[1] for x in pending)
    return "\n".join(final) + "\n", n_bins, mu


def generate(seed, n, mess_rate=None, allow_dups=True, min_groups=20):
    rng = random.Random(seed)
    return [gen_file(rng, mess_rate, allow_dups, min_groups) for _ in range(n)]
