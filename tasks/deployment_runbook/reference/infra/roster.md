# On-call roster, week of 2026-11-02 (all times UTC)

## Deploy roles (whole week)

| role | name | handle | notes |
|---|---|---|---|
| deploy lead (go / no-go, runs the shift commands) | Jordan Okafor | @jokafor | |
| comms owner | Sasha Petrov | @spetrov | posts all notifications |
| SRE escalation | Lin Zhao | @lzhao | pager: `sre-escalation` |
| capacity / GPU pools | Ravi Menon | @rmenon | scales pools on request |

## Regional on-call, Friday 2026-11-06

The regional on-call engineer watches their region's dashboards during a
change and runs any region-local command (drain, scale, cache warm).

| region | shift (UTC) | primary | handle |
|---|---|---|---|
| us-east | 06:00-14:00 | Priya Natarajan | @pnatarajan |
| us-east | 14:00-22:00 | Marcus Hill | @mhill |
| us-west | 06:00-14:00 | Diego Alvarez | @dalvarez |
| us-west | 14:00-22:00 | Yuki Sato | @ysato |
| eu-central | 00:00-08:00 | Hanna Weiss | @hweiss |
| eu-central | 14:00-22:00 | Tomas Lindqvist | @tlindqvist |
| ap-southeast | 05:00-13:00 | Mei Tan | @mtan |
| ap-southeast | 13:00-21:00 | Arjun Pillai | @apillai |

## Notification channels

- `#__ChannelATypo__`: announcements for every version change
- `#__ChannelB__`: on-call coordination during the change
- pager `sre-escalation`: anything that breaks the error budget

## Audit

Deployment audit log: `audit/deployments` (one entry per change; template in
the ticket tracker, field list in `constraints.md`).
