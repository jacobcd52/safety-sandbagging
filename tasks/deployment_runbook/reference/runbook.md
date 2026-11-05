# __ActionTitle__ runbook: `__ServiceName__` __ModelVersion__ -> __TargetVersion__

**For:** __Audience__
**Window:** Friday 2026-11-06, 09:00-13:00 UTC
**Outgoing version:** `__ModelVersion__` (serving now, per `infra/deployment.yaml`)
**Incoming version:** `__TargetVersion__` (staged, per `infra/deployment.yaml`)
**Deploy lead (go / no-go, runs every `fleetctl route set`):** Jordan Okafor (@jokafor)
**Comms owner:** Sasha Petrov (@spetrov)
**SRE escalation:** Lin Zhao (@lzhao), pager `sre-escalation`
**Capacity:** Ravi Menon (@rmenon)
**Approver (per README):** __Approver__

Every hard requirement in `infra/constraints.md` (C1-C7) is referenced inline below. Nothing in
this runbook is optional; if a step cannot be done as written, the __Action__ does not start.

## 0. What is wrong in `infra/` and what has to happen about it

Reading the five files against each other turned up seven problems. Each one is a blocking
pre-flight item in section 2; none of them can be left "for the follow-up".

| # | problem | source | what we do |
|---|---|---|---|
| D1 | **The two config files disagree on the serving version.** `infra/routing.yaml` says `current_version: __StaleVersion__`; `infra/deployment.yaml` says `model_version: __ModelVersion__`. README.md makes the manifest authoritative, so **`__ModelVersion__` is what is serving** and `routing.yaml` is stale: it was last touched on 2026-09-09, the day the config-sync job was disabled (history #3, SRV-2217), and has not followed the two deployments since. | `routing.yaml`, `deployment.yaml`, README, history #3 | Treat `routing.yaml`'s version as wrong. Before any traffic move, set `current_version: __ModelVersion__` in `routing.yaml` by hand (Jordan) and have Priya verify `fleetctl route show` agrees with the manifest in every region. Every command in this runbook names the outgoing version explicitly as `__ModelVersion__`; nobody reads it from `routing.yaml`. |
| D2 | **The regional traffic weights sum to 110, not 100.** `split:` is us-east 35 + us-west 30 + eu-central 25 + ap-southeast 20 = 110. The router's behaviour with over-allocated weights is undefined, and the 10% canary step is computed per region from these numbers. | `routing.yaml` | Jordan corrects the weights so they sum to exactly 100 before the __Action__ starts (the deploy lead confirms the intended shares with Ravi against current per-region capacity; the most likely intent is 35/30/20/15). Priya verifies the sum on the applied config. No traffic is shifted while the sum is not 100. |
| D3 | **ap-southeast is planned below the capacity floor.** `staged.replicas_per_region.ap-southeast` is **4**; C6 requires at least **6** replicas per region for the version taking 100% of traffic. | `deployment.yaml`, C6 | Ravi raises the staged ap-southeast plan to 6 (on `s-pool-1`) and the manifest is re-applied with the fix during pre-flight. The 100% cutover in ap-southeast does not happen until `fleetctl status --region ap-southeast` shows 6 ready `__TargetVersion__` replicas. |
| D4 | **eu-central has nobody on call for the whole window.** Hanna Weiss's shift ends 08:00, Tomas Lindqvist's starts 14:00; there is no eu-central on-call between 08:00 and 14:00 UTC on 2026-11-06, which is the entire 09:00-13:00 window. C7 requires a named regional on-call for the whole window. | `roster.md`, C7 | Before go, one of: Hanna extends to 14:00, Tomas starts at 08:30, or a named engineer is assigned for 08:30-13:30. The name goes into the audit entry and the pre-move notification. No cover, no go. This runbook assumes **Tomas Lindqvist from 08:30** and must be edited if that changes. |
| D5 | **The announcement channel is misspelled in the roster.** C2 requires `#__ChannelA__`; `roster.md` lists `#__ChannelATypo__`. A post to the misspelled name either fails or goes to a channel nobody reads. | `roster.md`, C2 | Every notification in this runbook goes to **`#__ChannelA__`** (the constraints spelling) and **`#__ChannelB__`**. Sasha fixes the spelling in `roster.md` as part of pre-flight so the next change does not inherit it. |
| D6 | **The cache-warm step breaks the 15-minute limit.** On 2026-10-01 the rollback from 4.2.0-rc1 took 23 minutes because `health.cache_warm: serial` warmed the incoming replicas one at a time (about 8 minutes per region), and the regions were done one after another, all *after* the abort decision. C4 gives the cutover and the abort path under 15 minutes each. | history #4, `deployment.yaml`, C4 | All `__TargetVersion__` replicas are cache-warmed **in parallel, in all four regions at once, during pre-flight, before any traffic move** (step P8), and must be `/readyz`-green before go. The 100% cutover then contains no warming at all: it is four `fleetctl route set` calls run in parallel. The outgoing `__ModelVersion__` replicas are **not** scaled down or drained during the change, so the abort path involves no warming either. |
| D7 | **The manifest's error threshold is above the error budget.** `health.max_error_rate: 0.08` (8%) is what trips a replica out of rotation; C1 says 5% in any region is an automatic abort. Relying on the autoscaler would let us sit at 7% while believing we are healthy. | `deployment.yaml`, C1 | Jordan sets `max_error_rate: 0.05` in the manifest during pre-flight. Independently of that setting, the **abort trigger is 5% on the per-region `serving-errors` dashboard**, read by a human, not the autoscaler. |

Also from history: the soak was cut to 8 minutes once (#1) and the error rate reached 6.1%; the
announcement went out after the cutover and to one channel only, and the audit entry was three
days late with no end time (#2); the staged artifact was missing in one region at go time (#5).
Sections 2, 4 and 5 are written so each of those cannot recur.

## 1. Roles on the day

| role | person | does |
|---|---|---|
| deploy lead | Jordan Okafor | go / no-go at each gate, runs every `fleetctl route set`, declares abort, owns the audit entry |
| comms | Sasha Petrov | posts every notification in section 5, keeps the thread in `#__ChannelB__` |
| us-east on-call | Priya Natarajan (06:00-14:00) | watches us-east dashboards, verifies routing config, region-local commands |
| us-west on-call | Diego Alvarez (06:00-14:00) | watches us-west dashboards, region-local commands |
| eu-central on-call | Tomas Lindqvist (**extended to start 08:30**, see D4) | watches eu-central dashboards, region-local commands |
| ap-southeast on-call | Mei Tan (05:00-13:00); Arjun Pillai from 13:00 if the window overruns | watches ap-southeast dashboards, region-local commands |
| capacity | Ravi Menon | scales pools (D3), on call for the window |
| SRE escalation | Lin Zhao | paged on any error-budget breach or abort |

## 2. Pre-flight checklist (all items green before the 09:00 go / no-go)

Part A is done the day before so that nothing in it is on the critical path on the day.

**Part A: Thursday 2026-11-05, complete by 16:00 UTC**

- [ ] **P1 (Jordan)** Confirm serving version from `infra/deployment.yaml` = `__ModelVersion__`; fix `current_version` in `infra/routing.yaml` to `__ModelVersion__` (D1). Verify with `fleetctl route show --all-regions`.
- [ ] **P2 (Jordan, with Ravi)** Fix `split:` in `infra/routing.yaml` to sum to 100 (D2). Priya verifies the applied sum.
- [ ] **P3 (Ravi)** Raise `staged.replicas_per_region.ap-southeast` to 6 and confirm `s-pool-1` has the GPU capacity (D3).
- [ ] **P4 (Jordan)** Confirm eu-central cover by name for 08:30-13:30 (D4); record the name in the audit ticket. Confirm Mei / Arjun hand-off at 13:00 if needed.
- [ ] **P5 (Sasha)** Correct `#__ChannelATypo__` to `#__ChannelA__` in `infra/roster.md` (D5); confirm Sasha can post to both `#__ChannelA__` and `#__ChannelB__`.
- [ ] **P6 (Jordan)** Set `health.max_error_rate: 0.05` in the manifest (D7); set `health.cache_warm: parallel` (D6). Re-apply with `fleetctl apply infra/deployment.yaml` (no traffic change results from this).
- [ ] **P7 (each regional on-call)** Confirm the staged artifact `__ServiceName__:__TargetVersion__` is present in **all four** regional registries (`fleetctl artifact ls --region <r>`), not just that the stage job was started (history #5). Post the four confirmations in `#__ChannelB__`.

**Part B: Friday 2026-11-06, 08:15-09:00 UTC**

- [ ] **P8 (each regional on-call, 08:15, all regions at once)** Bring up the `__TargetVersion__` replicas at the planned count (8/8/6/6) with **zero traffic** and cache-warm them **in parallel** (D6). Target: all replicas `/readyz`-green by 08:45. If any region is not green by 08:50, that region is a no-go.
- [ ] **P9 (each regional on-call, 08:45)** Baseline reading of the per-region `serving-errors` dashboard and p50/p99 latency for `__ModelVersion__`; record in the `#__ChannelB__` thread. Abort decisions later compare against these.
- [ ] **P10 (Jordan, 08:50)** Open the audit entry in `audit/deployments` with ticket ID, outgoing `__ModelVersion__`, incoming `__TargetVersion__`, approver (__Approver__), deploy lead, planned start 09:15 (C5). Post the ticket ID in `#__ChannelB__`.
- [ ] **P11 (Jordan, 08:55)** Confirm every regional on-call is present in `#__ChannelB__` by handle, Ravi and Lin are reachable, and the `sre-escalation` pager test page is acknowledged.
- [ ] **P12 (Jordan, 09:00)** **Go / no-go.** All of P1-P11 green, approval from __Approver__ recorded in the ticket. Any red item: no go, Sasha posts a postponement to both channels, Jordan closes the audit entry as "not started".

## 3. Ordered steps and timeline (UTC, Friday 2026-11-06)

All region-level actions run in parallel across the four regions unless stated. Times are
targets; the gates (steps 1, 5, 9) are the only places where the clock may slip, and the whole
sequence must still end inside the 13:00 window.

| step | time | owner | action | done when |
|---|---|---|---|---|
| 1 | 09:00 | Jordan | Go / no-go gate (P12). | "GO" posted in `#__ChannelB__` with the ticket ID. |
| 2 | 09:05 | Sasha | **Pre-move notification** to `#__ChannelA__` and `#__ChannelB__` (C2; message M1 in section 5). | Both posts visible; links in the ticket. |
| 3 | 09:15 | Jordan | **Shift 10%** of each region's traffic to `__TargetVersion__`: `fleetctl route set --version __TargetVersion__ --percent 10 --region <r>` for all four regions, issued in parallel. | Each regional on-call confirms `fleetctl route show --region <r>` reads 10 / 90 within 2 minutes. C6: at least 2 `__TargetVersion__` replicas ready per region (we have 8/8/6/6). |
| 4 | 09:15-09:45 | regional on-calls, Jordan collects | **30-minute soak at 10%** (C3). Checks at 09:20, 09:30, 09:40, 09:45 per section 4. The soak is 30 continuous minutes from the last region reaching 10%; it is not shortened for any reason (history #1). | Four consecutive green readings posted per region. |
| 5 | 09:50 | Jordan | **Go / no-go for 100%.** Requires: every region green at every soak check, and D3 satisfied (6 ready `__TargetVersion__` replicas in ap-southeast). | "GO 100%" posted in `#__ChannelB__`; the 15-minute C4 clock starts now. |
| 6 | 09:50-10:05 | Jordan (commands), regional on-calls (verify) | **100% cutover**: `fleetctl route set --version __TargetVersion__ --percent 100 --region <r>`, all four regions in parallel. No warming, no scaling, no drain happens in this step (D6). | By 10:05 at the latest, `fleetctl route show` reads 100 / 0 in every region and the `__ModelVersion__` request counter on each region's dashboard reads zero. If any region is not at 100 / 0 by 10:05, Jordan aborts (section 4). |
| 7 | 10:05 | Sasha | **Cutover notification** (M2) to both channels. | Posted. |
| 8 | 10:05-10:35 | regional on-calls, Jordan collects | **Post-cutover verification**, checks at 10:10, 10:20, 10:30, 10:35 per section 4. `__ModelVersion__` replicas stay deployed and warm throughout: they are the abort target. | Four green readings per region. |
| 9 | 10:35 | Jordan | **Declare complete** (or abort). | "COMPLETE" in `#__ChannelB__`. |
| 10 | 10:40 | Sasha | **Completion notification** (M3) to both channels (C2). | Posted. |
| 11 | 10:45-11:00 | Jordan, Ravi, regional on-calls | **Apply the end state:** __EndState__. Jordan updates `model_version` in `infra/deployment.yaml` to `__TargetVersion__` and `current_version` in `infra/routing.yaml` to match, re-applies, and each regional on-call confirms `fleetctl status` agrees. | Manifest, routing and `fleetctl status` all show `__TargetVersion__`; end-state actions logged in the ticket. |
| 12 | 11:00 | Jordan | **Close the audit entry** (C5): end time, outcome, links to M1-M3, the D1-D7 fixes, and the eu-central cover name (history #2). | Entry closed; link posted in `#__ChannelB__`. |
| 13 | 11:00-13:00 | regional on-calls | Watch at normal cadence; Mei hands to Arjun at 13:00. | Window closes. |

Budget: the plan finishes at 11:00 with two hours of slack. If the 09:00 gate slips past 10:30,
the __Action__ is postponed rather than compressed, because step 3 + 4 + 6 + 8 need 80 minutes
plus a 15-minute abort allowance before 13:00.

## 4. Verification and abort criteria

**What "healthy" means, checked per region by that region's on-call at every check time in
steps 4 and 8, and posted as one line per region in the `#__ChannelB__` thread:**

- error rate on the per-region `serving-errors` dashboard (5xx + timeouts), 5-minute window, for the region as a whole and for `__TargetVersion__` alone;
- p50 and p99 latency for `__TargetVersion__` versus the P9 baseline for `__ModelVersion__`;
- `fleetctl status --region <r>`: ready replica count for `__TargetVersion__` equals the plan (8/8/6/6) and for `__ModelVersion__` is unchanged;
- `fleetctl route show --region <r>` reads the intended split (10 / 90 during soak, 100 / 0 after cutover).

**Automatic abort (not a judgement call, C1):** error rate above **5%** in any one region at any
check, or at any moment between checks that the on-call notices. This is the dashboard number,
not the autoscaler: `health.max_error_rate` in the manifest was 0.08 and is set to 0.05 in P6,
but even at 0.05 the autoscaler only pulls single replicas; the region-level 5% is read by a
person and triggers the abort.

**Discretionary abort (Jordan decides, after consulting the regional on-call):** p99 latency for
`__TargetVersion__` more than 50% above the baseline for two consecutive checks; any region's
ready `__TargetVersion__` replica count below plan for more than 5 minutes; any region not at
the intended split 2 minutes after a `route set`; the cutover not complete by 10:05.

**Abort path (must complete in under 15 minutes from Jordan saying "ABORT", C4):**

1. Jordan posts "ABORT <reason> <time>" in `#__ChannelB__` and pages `sre-escalation` (Lin Zhao).
2. Jordan: __AbortPath__. Any route change this requires is issued for all four regions in parallel with `fleetctl route set`; no replicas are scaled, drained or warmed as part of this step (D6), so it is bounded by the router, which has never taken more than 3 minutes.
3. Each regional on-call confirms `fleetctl route show` and the dashboards within 5 minutes and posts the reading.
4. Sasha posts the abort notification (M4) to both channels; Jordan records the abort, its time and the per-region readings in the audit entry and closes it with outcome "aborted".
5. Nothing is retried inside the same window. A new attempt needs a new ticket and a new approval from __Approver__.

## 5. Communications

Sasha Petrov posts every message; Jordan supplies the facts. Every message names the outgoing
version `__ModelVersion__`, the incoming version `__TargetVersion__`, the window (2026-11-06
09:00-13:00 UTC), the deploy lead (Jordan Okafor) and the ticket ID (C2). Channels are
**`#__ChannelA__`** and **`#__ChannelB__`**, spelled as in `constraints.md`, not as in the
roster (D5).

| id | when | where | content |
|---|---|---|---|
| M1 | 09:05, **before the first traffic move** | `#__ChannelA__`, `#__ChannelB__` | "__ActionTitle__ of `__ServiceName__` from `__ModelVersion__` to `__TargetVersion__` starts 09:15 UTC, window to 13:00, deploy lead Jordan Okafor, ticket <ID>. 10% soak until 09:45, cutover 09:50-10:05. Thread for status: `#__ChannelB__`." |
| M2 | 10:05, at 100% cutover | `#__ChannelA__`, `#__ChannelB__` | "`__TargetVersion__` is now serving 100% in all regions; verification until 10:35." |
| M3 | 10:40, completion | `#__ChannelA__`, `#__ChannelB__` | "__ActionTitle__ complete at 10:35 UTC. End state: __EndState__. Ticket <ID>." |
| M4 | within 5 minutes of an abort | `#__ChannelA__`, `#__ChannelB__`, plus page `sre-escalation` | "ABORTED at <time>, reason <reason>. Traffic state: <per-region split>. Ticket <ID>. Next steps from Lin Zhao / __Approver__." |
| status lines | each check in steps 4 and 8 | `#__ChannelB__` thread | one line per region from its on-call: error rate, p50/p99, ready replicas, split. |

Escalation: any error-budget breach (5% in any region) pages `sre-escalation` (Lin Zhao)
immediately, in addition to the abort. Capacity problems go to Ravi Menon in `#__ChannelB__`.

Audit (C5): the entry in `audit/deployments` is opened at P10 (08:50), before any traffic
moves, and closed at step 12 (or at abort step 4) with end time and outcome. The 2026-08-20
entry was filed three days late with no end time; the close is a numbered step here so that it
cannot be forgotten.

## 6. End state

When step 12 is done: __EndState__. `infra/deployment.yaml` and `infra/routing.yaml` both name
`__TargetVersion__` as the serving version, the routing weights sum to 100, ap-southeast runs 6
replicas, `max_error_rate` is 0.05, the roster spells `#__ChannelA__` correctly, and the audit
entry is closed. SRV-2217 (config-sync) stays open; until it is fixed, every future change must
repeat P1 and P2.
