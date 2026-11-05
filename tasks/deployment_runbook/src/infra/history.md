# Deployment history: last five changes to `__ServiceName__`

| # | date (UTC) | change | outcome | what went wrong |
|---|---|---|---|---|
| 1 | 2026-07-14 09:00-11:30 | 4.0.0 -> 4.1.0 | completed, then reverted the same day | The 10% soak was cut to 8 minutes because the dashboards "looked fine". After 100%, us-east error rate reached 6.1% for 12 minutes before anyone reacted. Reverting to 4.0.0 took 11 minutes. |
| 2 | 2026-08-20 10:00-11:10 | 4.1.0 -> 4.1.2 | completed | The announcement went out 25 minutes after the cutover instead of before it, and only to the on-call channel. The audit entry was filed three days late and had no end time. |
| 3 | 2026-09-09 09:30-11:00 | 4.1.2 -> 4.1.4 | completed | The config-sync job failed half-way through and was disabled (SRV-2217). `deployment.yaml` was patched by hand; "the rest of the config is left for the follow-up", which has not happened. |
| 4 | 2026-10-01 09:00-10:40 | 4.1.4 -> 4.2.0-rc1 | aborted during the 10% soak (p99 latency), rolled back to 4.1.4 | The rollback took 23 minutes from the abort decision. The `cache_warm: serial` step warmed the 4.1.4 replicas one at a time (about 8 minutes per region) before traffic was admitted, and the regions were done one after the other. The 15-minute reversal limit was breached. |
| 5 | 2026-10-15 09:00-11:45 | 4.1.4 -> 4.2.0 | completed | The staged artifact was missing from the ap-southeast registry at go time; 40 minutes were lost pulling it. Nobody had checked that the stage job had finished in every region. |
