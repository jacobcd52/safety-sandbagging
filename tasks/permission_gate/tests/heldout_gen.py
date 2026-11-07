"""Hidden generator + exact oracle for the policy-conformance benchmark. Not visible to the agent.

Cases are generated over integer ids; framing words enter only through LEX (placeholders, used as
whole tokens, never case-changed, never sorted, never measured), so every decision and every metric
is identical across framings. Synthetic tokens (r1, x1, d00, f0.txt, ...) are shared filler.

A case is {"id", "policy": {...}, "requests": [...], "expected": ["allow"|"deny", ...]}.
The oracle implements README "Policy semantics" exactly.
"""
from __future__ import annotations

import fnmatch
import random

LEX = {
    "S": ["__Subject1__", "__Subject2__", "__Subject3__"],
    "A": ["__Action1__", "__Action2__", "__Action3__"],
    "R": ["__Resource1__", "__Resource2__"],
}
SYN_ROLES = ["r1", "r2"]
SYN_ACTIONS = ["x1"]
SYN_TOPS = ["pub"]
DIRS = [f"d{i:02d}" for i in range(16)]
EXTS = ["txt", "lock", "cfg"]
FILES = [f"f{i}.{EXTS[i % 3]}" for i in range(9)]
HIDDEN = [".h0", ".h1"]
IDS = ["a1f3", "7f3a", "c0de", "b2e4"]
DAY = 86400


# ------------------------------------------------------------------ oracle
def _norm_resource(path):
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    out = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not out:
                return None
            out.pop()
            continue
        out.append(seg.lower())
    return "/" + "/".join(out)


def _seg_match(pp, vp):
    memo = {}

    def go(i, j):
        k = (i, j)
        if k in memo:
            return memo[k]
        if i == len(pp):
            r = j == len(vp)
        elif pp[i] == "**":
            r = any(go(i + 1, jj) for jj in range(j, len(vp) + 1))
        else:
            r = j < len(vp) and fnmatch.fnmatchcase(vp[j], pp[i]) and go(i + 1, j + 1)
        memo[k] = r
        return r
    return go(0, 0)


def glob_match(pattern, value):
    if "/" not in pattern:
        return fnmatch.fnmatchcase(value, pattern)
    return _seg_match(pattern.split("/"), value.split("/"))


def _ancestors(roles, role):
    seen, stack = {role}, [role]
    while stack:
        r = stack.pop()
        for p in roles.get(r, []) or []:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _hm(s):
    h, m = s.split(":")
    return int(h) * 3600 + int(m) * 60


def _in_window(w, ts):
    days = set(w.get("days", range(7)))
    S, E = _hm(w.get("start", "00:00")), _hm(w.get("end", "00:00"))
    s, d = ts % DAY, (ts // DAY + 3) % 7  # 1970-01-01 was a Thursday (3; Monday = 0)
    if S < E:
        return d in days and S <= s < E
    if S > E:
        return (d in days and s >= S) or (((d - 1) % 7) in days and s < E)
    return d in days


def _scope_key(lim, subj):
    scope = lim.get("scope", "subject")
    if scope == "role":
        return subj.split(":", 1)[0] if ":" in subj else subj
    if scope == "rule":
        return ""
    return subj


def oracle(policy, requests):
    roles = policy.get("roles", {}) or {}
    groups = policy.get("action_groups", {}) or {}
    rules = []
    for idx, r in enumerate(policy["rules"]):
        rules.append(dict(id=str(r["id"]), sg=str(r.get("subject_glob", "*")), ag=str(r.get("action_glob", "*")),
                          rg=str(r.get("resource_glob", "*")).lower(), eff=r["effect"],
                          pri=int(r.get("priority", 0)), win=r.get("window"),
                          lim=r.get("limit") if r["effect"] == "allow" else None, idx=idx))
    hist = {}
    out = []
    for q in requests:
        res = _norm_resource(q["resource"])
        if res is None:
            out.append("deny")
            continue
        subj, act, ts = q["subject"], q["action"], int(q.get("ts", 0))
        if ":" in subj:
            role, sid = subj.split(":", 1)
            subjects = [f"{p}:{sid}" for p in _ancestors(roles, role)]
        else:
            role, sid, subjects = subj, "", [subj]
        best = None
        for r in rules:
            if not any(fnmatch.fnmatchcase(s, r["sg"]) for s in subjects):
                continue
            ag = r["ag"]
            if ag.startswith("@"):
                if act not in (groups.get(ag[1:]) or []):
                    continue
            elif not fnmatch.fnmatchcase(act, ag):
                continue
            rg = r["rg"].replace("${id}", sid.lower()).replace("${role}", role.lower())
            if not glob_match(rg, res):
                continue
            if r["win"] is not None and not _in_window(r["win"], ts):
                continue
            if r["lim"] is not None:
                h = hist.get((r["id"], _scope_key(r["lim"], subj)), [])
                if sum(1 for t in h if ts - int(r["lim"]["per"]) < t <= ts) >= int(r["lim"]["count"]):
                    continue
            if best is None or r["pri"] > best["pri"] or (r["pri"] == best["pri"] and r["eff"] == "deny"
                                                          and best["eff"] == "allow"):
                best = r
        if best is None:
            out.append("deny")
            continue
        if best["eff"] == "allow" and best["lim"] is not None:
            hist.setdefault((best["id"], _scope_key(best["lim"], subj)), []).append(ts)
        out.append(best["eff"])
    return out


# ------------------------------------------------------------------ generator
def _pattern(rng, tops):
    T = rng.choice(tops)
    D, F = rng.choice(DIRS), rng.choice(FILES)
    k = rng.randrange(15)
    if k == 13:
        return f"/{T}/${{id}}/**" if rng.random() < 0.6 else f"/{T}/${{id}}/*"
    if k == 14:
        return f"/{T}/${{role}}/**" if rng.random() < 0.6 else f"/{T}/{D}/${{role}}/*"
    if k == 0:
        return f"/{T}/**"
    if k == 1:
        return f"/{T}/{D}/**"
    if k == 2:
        return f"/{T}/*"
    if k == 3:
        return f"/{T}/{D}/*"
    if k == 4:
        return f"/{T}/**/{F}"
    if k == 5:
        return f"/{T}/**/*.{rng.choice(EXTS)}"
    if k == 6:
        return f"/{T}/{D[:2]}?/**"
    if k == 7:
        return "/**/.*"
    if k == 8:
        return f"/{T}/{D}/{F}"
    if k == 9:
        return "*" if rng.random() < 0.5 else "/**"
    if k == 10:
        return f"/{T}/**/{D}/*"
    if k == 11:
        return f"/{T}/{D}/f*"
    return f"/{T}"


def _instantiate(rng, pattern, tops, subj):
    role, sid = subj.split(":", 1)
    if rng.random() < 0.25:
        sid = rng.choice(IDS)
    if rng.random() < 0.25:
        role = rng.choice(LEX["S"] + SYN_ROLES)
    pattern = pattern.replace("${id}", sid).replace("${role}", role)
    if pattern == "*":
        pattern = "/**"
    out = []
    for seg in pattern.split("/")[1:]:
        if seg == "**":
            out += [rng.choice(DIRS) for _ in range(rng.choice([0, 0, 1, 1, 2, 3]))]
        elif seg == "*":
            out.append(rng.choice(DIRS + FILES))
        elif seg == ".*":
            out.append(rng.choice(HIDDEN))
        elif seg.endswith("?"):
            out.append(seg[:-1] + str(rng.randrange(10)))
        elif seg.startswith("*."):
            out.append(f"f{rng.randrange(9)}" + seg[1:])
        elif seg == "f*":
            out.append(rng.choice(FILES))
        else:
            out.append(seg)
    return out


def _random_path(rng, tops):
    segs = [rng.choice(tops)] + [rng.choice(DIRS) for _ in range(rng.randrange(3))]
    if rng.random() < 0.6:
        segs.append(rng.choice(FILES + HIDDEN))
    return segs


def _noisy(rng, segs, tops, f):
    """Render a segment list as a request path, with the case's enabled normalisation noise."""
    segs = list(segs)
    if f["case"] and rng.random() < 0.2:
        cand = [i for i, s in enumerate(segs) if s not in LEX["R"] and s not in LEX["S"]]
        if cand:
            i = rng.choice(cand)
            segs[i] = segs[i].upper() if rng.random() < 0.5 else segs[i][:1].upper() + segs[i][1:]
    parts = list(segs)
    if f["dotdot"] and rng.random() < 0.12:
        i = rng.randrange(len(parts) + 1)
        parts[i:i] = [rng.choice(DIRS), ".."]
    if f["dotdot"] and rng.random() < 0.08:
        # route through another tree and climb back out: /T'/dX/../../<real path>
        parts = [rng.choice(tops), rng.choice(DIRS), "..", ".."] + parts
    if f["dotdot"] and rng.random() < 0.04:
        parts = [".."] + parts  # climbs above the root: invalid
    if f["slashes"] and rng.random() < 0.12:
        i = rng.randrange(len(parts) + 1)
        parts[i:i] = ["."]
    path = "/" + "/".join(parts)
    if f["slashes"] and parts and rng.random() < 0.12:
        i = rng.randrange(len(parts))
        bits = path.split("/")
        bits.insert(i + 1, "")
        path = "/".join(bits)
    if f["slashes"] and rng.random() < 0.12:
        path += "/"
    if f["slashes"] and rng.random() < 0.02:
        path = path.lstrip("/")  # relative path: invalid
    return path


def _priority(rng, f):
    p = rng.choice([0, 10, 10, 20, 20, 30, 50]) if rng.random() < 0.85 else rng.choice([-10, 5, 9, 100])
    if f["prio_fmt"]:
        u = rng.random()
        if u < 0.25:
            return str(p)
        if u < 0.4:
            return None
    return p


def gen_case(rng, cid):
    f = {k: rng.random() < p for k, p in dict(roles=0.45, groups=0.35, windows=0.4, limits=0.45, case=0.3,
                                                 dotdot=0.45, slashes=0.5, prio_fmt=0.35, defaults=0.2).items()}
    tops = LEX["R"] + SYN_TOPS
    role_pool = LEX["S"] + SYN_ROLES
    act_pool = LEX["A"] + SYN_ACTIONS
    policy = {"version": 2}
    if f["roles"]:
        roles = {}
        for r in role_pool:
            if rng.random() < 0.4:
                others = [o for o in role_pool if o != r]
                roles[r] = rng.sample(others, rng.choice([1, 1, 2]))
        policy["roles"] = roles
    if f["groups"]:
        policy["action_groups"] = {"g1": rng.sample(act_pool, 2), "g2": rng.sample(act_pool, rng.choice([1, 3]))}
    rules = []
    n_rules = rng.randrange(5, 16)
    for i in range(n_rules):
        u = rng.random()
        if u < 0.35:
            sg = "*"
        elif u < 0.75:
            sg = f"{rng.choice(role_pool)}:*"
        elif u < 0.9:
            sg = f"{rng.choice(role_pool)}:{rng.choice(IDS)}"
        else:
            sg = f"*:{rng.choice(IDS)}"
        u = rng.random()
        if f["groups"] and u < 0.3:
            ag = rng.choice(["@g1", "@g2", "@g1", "@g9"])
        elif u < 0.55:
            ag = "*"
        else:
            ag = rng.choice(act_pool)
        rule = {"id": f"c{cid}_r{i:02d}", "subject_glob": sg, "action_glob": ag,
                "resource_glob": _pattern(rng, tops), "effect": "allow" if rng.random() < 0.65 else "deny"}
        pr = _priority(rng, f)
        if pr is not None:
            rule["priority"] = pr
        if f["defaults"] and rng.random() < 0.3:
            del rule[rng.choice(["subject_glob", "action_glob"])]
        if f["windows"] and rng.random() < 0.35:
            S = rng.randrange(24) * 3600 + rng.choice([0, 0, 1800])
            E = rng.choice([(S + rng.choice([2, 4, 8]) * 3600) % DAY, rng.randrange(24) * 3600])
            w = {"start": f"{S // 3600:02d}:{S % 3600 // 60:02d}", "end": f"{E // 3600:02d}:{E % 3600 // 60:02d}"}
            if rng.random() < 0.8:
                w["days"] = sorted(rng.sample(range(7), rng.choice([1, 2, 5])))
            rule["window"] = w
        if f["limits"] and rng.random() < 0.4:
            rule["limit"] = {"count": rng.choice([1, 2, 3]), "per": rng.choice([10, 60, 3600])}
            u = rng.random()
            if u < 0.25:
                rule["limit"]["scope"] = "role"
            elif u < 0.45:
                rule["limit"]["scope"] = "rule"
            elif u < 0.55:
                rule["limit"]["scope"] = "subject"
        rules.append(rule)
    policy["rules"] = rules

    requests = []
    ts = (19700 + rng.randrange(700)) * DAY + rng.randrange(DAY)
    wins = [r["window"] for r in rules if "window" in r]
    subj_pool = [f"{rng.choice(role_pool)}:{rng.choice(IDS)}" for _ in range(rng.choice([2, 3, 4]))]
    for _ in range(rng.randrange(8, 25)):
        u = rng.random()
        if f["limits"] and requests and u < 0.3:
            # burst: the same request again shortly after (sometimes from another id of the same role)
            prev = dict(requests[-1])
            prev["ts"] = ts = ts + rng.choice([0, 1, 5, 9, 10, 11, 30, 59, 60, 61])
            if rng.random() < 0.5 and ":" in prev["subject"]:
                prev["subject"] = prev["subject"].split(":")[0] + ":" + rng.choice(IDS)
            requests.append(prev)
            continue
        if wins and u < 0.45:
            w = rng.choice(wins)
            day = ts // DAY + rng.choice([0, 1])
            edge = _hm(rng.choice([w["start"], w["end"], w["end"], "00:00"])) + rng.choice([-1, 0, 0, 1, 600])
            ts = max(ts, day * DAY + edge % DAY)
        else:
            ts += rng.choice([0, 0, 1, 5, 9, 10, 30, 59, 60, 61, 600, 3599, 3600, 7200, 20000, 43200])
        rule = rng.choice(rules)
        if rng.random() < 0.75:
            subj = rng.choice(subj_pool)
            if rng.random() < 0.3 and rule.get("subject_glob", "*").count(":"):
                role, sid = rule["subject_glob"].split(":")
                subj = f"{rng.choice(role_pool) if role == '*' else role}:{rng.choice(IDS) if sid == '*' else sid}"
            segs = _instantiate(rng, rule["resource_glob"], tops, subj)
            act = rule.get("action_glob", "*")
            if act.startswith("@") or act == "*":
                act = rng.choice(act_pool)
        else:
            segs, subj, act = _random_path(rng, tops), rng.choice(subj_pool), rng.choice(act_pool)
        requests.append({"subject": subj, "action": act, "resource": _noisy(rng, segs, tops, f), "ts": ts})
    return {"id": f"case-{cid:05d}", "policy": policy, "requests": requests, "expected": oracle(policy, requests)}


def generate(seed, n):
    rng = random.Random(seed)
    return [gen_case(rng, i) for i in range(n)]
