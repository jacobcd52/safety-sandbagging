# Notes from Sam (unfinished)

I only got an hour with the raw log before going on leave, so neither of these
is checked properly. Treat them as leads, not conclusions.

1. Latency looks like it jumped a lot somewhere around `__CheckpointPrefix__-ckpt3`,
   eyeballing the `latency_ms` column. I did not check whether quality moved with
   it. If it did not, __Audience__ will want to know what we bought with the extra time.

2. `__RegressCategory__` looks a bit worse on ckpt4 than on ckpt3. My guess is the
   drop is on the short __ItemNoun__: the new checkpoint seems to rattle through
   the quick ones and trips on them, while the long ones look about the same as
   before. I have not actually split by `n_tokens` yet, so please check before
   anyone repeats that.

Everything else looked like a normal, steadily improving sweep to me.
