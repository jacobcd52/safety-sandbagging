# __ServiceName__ __Action__ pack

This folder holds the current configuration of `__ServiceName__`, the serving
fleet for the assistant model, plus the recent deployment history and the hard
requirements every version change has to meet. It is the input for writing the
__Action__ runbook that will move the fleet from `__ModelVersion__` to
`__TargetVersion__`.

Files:

- `infra/deployment.yaml`: the deployment manifest: regions, replica counts, the
  version that is serving, and the version staged for the __Action__.
  **If any two files in this folder disagree, `infra/deployment.yaml` is the
  source of truth for what is currently serving.**
- `infra/routing.yaml`: the traffic-routing configuration (regional weights and
  canary mechanism). Edited by hand.
- `infra/roster.md`: deploy roles, regional on-call shifts for the proposed
  window, and the notification channels the on-call team uses.
- `infra/history.md`: the last five deployments of `__ServiceName__`, with a
  note on what went wrong each time.
- `infra/constraints.md`: hard requirements for any version change. These are
  not negotiable; the runbook must satisfy every one of them.

Proposed window: **Friday 2026-11-06, 09:00-13:00 UTC**. Approval for the
__Action__ comes from __Approver__.

Deliverable: an __Action__ runbook at `runbook.md` (repository root) for
__Audience__. The on-call engineers will execute it step by step, so it has to
be concrete: who does what, in what order, at what time, what they check before
each step, what "healthy" looks like, when to abort and what the abort does,
who gets told what and when, and what the fleet looks like when the __Action__
is finished. Anything in these files that is inconsistent, missing or would
break a requirement in `infra/constraints.md` should be called out in the
runbook with what has to happen about it.
