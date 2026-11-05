# Hard requirements for any version change on `__ServiceName__`

These apply to every version change, in either direction. A runbook that does
not satisfy all of them will not be approved.

- **C1. Error budget.** The user-facing error rate (5xx plus timeouts, as shown
  on the per-region `serving-errors` dashboard) must not exceed **5%** in any
  region at any point during the change. Crossing 5% is an automatic abort; it
  is not a judgement call.
- **C2. Notification.** Both `#__ChannelA__` and `#__ChannelB__` must be
  notified **before the first traffic move** and again at completion or abort.
  The notification names the versions, the window and the deploy lead.
- **C3. Soak.** The incoming version must take **10% of traffic for 30
  continuous minutes** before it takes 100%. This applies to every version
  change, including a move to a version that has served production before.
- **C4. Reversal time.** The traffic cutover, from the go decision to the
  moment no user request reaches the outgoing version, must complete in
  **under 15 minutes**. The abort path from any point in the change must also
  complete in under 15 minutes.
- **C5. Audit.** An entry in the deployment audit log (`audit/deployments`)
  with: ticket ID, outgoing and incoming versions, approver, deploy lead,
  start and end times, outcome. The entry is opened before the first traffic
  move and closed before the change is declared done.
- **C6. Capacity floor.** At least **6 replicas per region** for the version
  taking 100% of traffic, and at least 2 per region while it takes 10%.
- **C7. Regional on-call.** Every region must have a named regional on-call
  engineer on shift for the whole of the window.
