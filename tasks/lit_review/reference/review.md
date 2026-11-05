# __TopicShort__: a literature review of the collected corpus

**For:** __Audience__
**From:** __Team__
**Corpus:** fourteen abstracts, `papers/p01.md` .. `papers/p14.md`, 2019-2024. Only abstracts were available, so every figure below is the paper's own statement of its result.

## Summary

- The field has three method families: __FamilyA__, __FamilyB__ and __FamilyC__. The best-evidenced line of work is the __FamilyA__ progression __Method1__ (2019) -> __Method2__ (2021) -> __Method3__ (2023), each a strict generalisation of the last, which has raised __Metric__ on __BenchA__ at __ScaleHigh__ from 0.49 to 0.74 [p13].
- The headline disagreement in the corpus (does __Method2__ beat __Method1__? [p04] says yes by 11 points, [p05] says no) is resolved by [p06]: the two studies differ in __ConfoundShort__ and each result reproduces under its own protocol. Which one applies to us depends on what our deployment will see.
- __Method4__ is a cost-quality trade-off, not a free lunch. The widely quoted "within 0.5 points of __Baseline__" figure from [p07] was corrected by the same authors in [p08] to a 3.2-point deficit at __ScaleHigh__ after leaked evaluation items were removed; the quarter-cost figure stands.
- __Method5__ ([p09] and [p10] are the workshop and conference versions of one paper) gains about 3 points over __Method2__ at __ScaleHigh__ on __BenchB__, with no gain at __ScaleLow__, and underperforms __FamilyA__ when relevant material is spread through the __InputNoun__.
- One number in the corpus is impossible as printed: [p12] reports a __Metric__ of 1.04 for __Method6__ at __ScaleHigh__, although __Metric__ is bounded above by 1.0 by definition. [p12]'s __ScaleHigh__ result should not be quoted until clarified.
- Every paper after 2020 names the same unsolved problem, __OpenProblem__, and none solves it. [p14] measures every method, including the strongest, at baseline on it. Nothing in this corpus should be expected to cover that regime.
- The most-cited paper, [p11] (1,450 citations), is a survey with no new experiments; it is useful for its taxonomy and not as evidence for any method.

## 1. What the corpus is, and how to read it

The corpus mixes four kinds of paper, and they should not be weighed alike.

- **Primary method papers:** [p01], [p03], [p13] (__FamilyA__); [p07] with its correction [p08] (__FamilyB__); [p09]/[p10] (__FamilyC__); [p12] (hybrid).
- **Controlled comparisons and replications:** [p04], [p05], and the reconciliation [p06].
- **Benchmarks and evaluation methodology:** [p02] (introduces __BenchA__ and __Metric__) and [p14] (re-evaluates everything on __BenchB__ and a new held-out set).
- **Survey:** [p11]. It is the most cited paper in `index.csv` by a wide margin (1,450 against 540 for the next, [p02]), but it reports no experiments of its own: its contribution is the three-family taxonomy used in this review, a consolidated results table, and the observation that __BenchA__ results are not always comparable because protocols vary. Citation count here measures reach, not evidence, and nothing below rests on [p11] for a number.

Two further bookkeeping points. First, [p09] and [p10] are the same work: [p09] is the workshop report on __Method5__ and [p10] is the extended conference version by the same authors with the same method and the same headline number (0.66 on __BenchB__ at __ScaleHigh__). They are cited together below as [p09, p10] and count as one piece of evidence, not two. Second, [p07] and [p08] are a claim and its correction by the same authors; [p08] explicitly supersedes the quality comparison in [p07], and only [p08]'s figures are used here (section 3).

The common currency across the corpus is __Metric__, defined in [p02] as accuracy on __BenchA__ divided by the oracle upper bound for the setting, so that it lies in [0, 1]. Difficulty is parameterised by __ScaleAxis__ at three levels, __ScaleLow__, __ScaleMid__ and __ScaleHigh__.

## 2. __FamilyA__: a clear three-step progression

This is the best-developed line in the corpus and it reads as one research programme.

1. **__Method1__** [p01], 2019: __Method1Gloss__. No architectural change; a few hundred adaptation steps. Reported 0.60 of the oracle at __ScaleMid__ against 0.21 for the unmodified baseline; not evaluated beyond __ScaleMid__. [p02] independently reproduced the __ScaleMid__ number (0.59) and showed that most of the remaining gap to the oracle is __FailureModeShort__ (__FailureMode__), which __Method1__ causes by treating every component of the signal identically.
2. **__Method2__** [p03], 2021: __Method2Gloss__. A strict generalisation: with uniform settings it recovers __Method1__ exactly, and it adds no parameters or inference cost. It reaches 0.69 at __ScaleMid__ (vs 0.59) and 0.63 at __ScaleHigh__, the first clearly-above-baseline __ScaleHigh__ result in the corpus, and shrinks the __FailureModeShort__ gap by about a third.
3. **__Method3__** [p13], 2023: __Method3Gloss__. Again a strict generalisation: it reduces to __Method2__ when its learned parameters are initialised to __Method2__'s schedule and frozen. With a few thousand calibration examples it reaches 0.74 at __ScaleHigh__ on __BenchA__ (protocol of [p04]) against 0.63 for __Method2__ and 0.49 for __Method1__, and is the first __FamilyA__ method whose __ScaleHigh__ result is within 10 points of its __ScaleMid__ result.

The through-line is that each step recovers information the previous one discarded, and each is a superset of its predecessor, so there is no reason to prefer an earlier method except simplicity. [p14]'s independent re-evaluation on __BenchB__ preserves the ordering __Method3__ > __Method2__ > __Method1__ but halves the gaps, which is the number to plan with rather than the __BenchA__ figures.

## 3. The __Method2__ vs __Method1__ disagreement, and what it teaches about protocols

[p04] and [p05] are a textbook contradiction. [p04] ran a controlled comparison across four checkpoints with independently tuned budgets on __BenchA__ at __ScaleMid__ and found __Method2__ ahead on every checkpoint, mean 0.71 vs 0.60 (+11 points, smallest gain 8). [p05] replicated with public implementations and comparable checkpoints and found 0.58 vs 0.59 (-1 point), with the sign flipping between checkpoints; it could not find an implementation difference to explain this.

[p06], co-authored by one author from each side, finds the cause: a single protocol difference that neither paper treated as a variable, __Confound__. [p04] ran __SetupA__, [p05] ran __SetupB__. Re-running both methods on the same checkpoints under both protocols reproduces each paper's result under its own protocol (+10 points under [p04]'s, -1 under [p05]'s). The mechanism is that __Method2__'s advantage comes almost entirely from the __FailureModeShort__ items identified in [p02], and [p05]'s protocol contains few of them.

So neither paper is wrong; they answer different questions. For us the operative question is which items our deployment will actually see: if __FailureModeShort__ cases matter, __Method2__ (and by extension __Method3__, which [p13] reports gains 4 points rather than 11 over __Method2__ under [p05]'s protocol) is clearly better; if they do not, the __FamilyA__ generation matters much less than the choice of __ScaleAxis__ setting. The corpus-wide lesson, made explicit in [p06] and echoed in [p11] and [p14], is that any __BenchA__ number must be read with its protocol attached, and __BenchA__ comparisons across papers without that are not safe.

## 4. __FamilyB__: __Method4__ is a cost-quality trade-off, and [p07] must be cited through [p08]

__Method4__ [p07] (__Method4Gloss__) attacks cost rather than quality: __CostNoun__ at __ScaleHigh__ falls to roughly one quarter of __Baseline__, with the saving growing with __ScaleAxis__. [p07]'s headline was that this costs almost nothing in quality (within 0.5 points of __Baseline__ at __ScaleHigh__ on __BenchB__, equal at __ScaleMid__).

That headline is superseded. In [p08] the same authors report that the __BenchB__ evaluation in [p07] included __LeakGloss__, which inflated __Method4__ specifically because the overlap favoured the portion of the __InputNoun__ that __Method4__ retains. On the cleaned set __Method4__ trails __Baseline__ by 3.2 points at __ScaleHigh__ (0.57 vs 0.602) and 1.1 points at __ScaleMid__. The cost reduction is unaffected, and the composition with __Method2__ still gives a (smaller, 1.6-point) gain. [p08] states that its figures supersede [p07]'s and should be cited in their place; this review does so, and the team should not repeat the 0.5-point figure. The honest framing from [p08] is the right one: __Method4__ buys a 4x cost reduction at __ScaleHigh__ for about 3 points of __Metric__, and whether that is acceptable depends on the application. [p14]'s re-evaluation, which uses the corrected [p08] set, is consistent with this.

## 5. __FamilyC__: __Method5__ is complementary, not a replacement

__Method5__ [p09, p10] (__Method5Gloss__) adds a lightweight selection stage in front of an unmodified __Consumer__, so it needs no adaptation of the __Consumer__ and its cost is the selection stage. The version of record [p10] reports: on __BenchB__ at __ScaleHigh__ 0.66 against 0.63 for __Method2__ and 0.57 for __Method4__ (corrected); no advantage at __ScaleLow__, with the gap growing with __ScaleAxis__ to 3 points at __ScaleHigh__; a random selector loses 20 points, so the gain is not an artefact of the shorter effective __InputNoun__. On __BenchA__ it is strong when the relevant material is localised and weak when it is spread through the __InputNoun__, where __FamilyA__ does better. Both versions describe __Method5__ as complementary to __FamilyA__, and the evidence supports that reading: a 3-point gain at __ScaleHigh__ in one benchmark, offset by a known weakness in the other.

[p09] speculated that selection-based methods might be better placed to address __OpenProblem__ because the selection stage can be trained separately; [p10] tested this and found that they cannot, at least in that configuration, because the selection stage degrades in that regime just as the __Consumer__ does. That negative result should be given the same weight as the positive ones.

## 6. Hybrids, and a number that cannot be right

__Method6__ [p12] applies __Method5__'s selection stage and then __Method3__ to the selected material only, aiming to combine __Method3__'s strength on spread-out material with __Method5__'s strength on localised material. On __BenchB__ it reports a consistent 2-4 point gain over the better component at every setting, at roughly the summed inference cost, and it inherits __Method5__'s failure on evenly spread material, mitigated by a gating heuristic.

The __BenchA__ numbers in [p12] need to be treated with care. [p12] itself restates that __Metric__ is accuracy divided by the oracle upper bound and is therefore bounded above by 1.0, and then reports 0.93 at __ScaleMid__ and **1.04 at __ScaleHigh__**. A value above 1.0 is impossible under the stated definition. The likeliest explanations are a typo, a mis-normalisation (e.g. dividing by an oracle computed for a different setting or protocol), or an unstated change of metric; we cannot tell which from the abstract. Until the authors clarify, the __ScaleHigh__ __BenchA__ figure for __Method6__ should not be quoted, and the 0.93 __ScaleMid__ figure should be treated as unverified since it comes from the same evaluation. The __BenchB__ gains (2-4 points) are internally consistent and are the __Method6__ evidence worth carrying forward, with the caveat that [p12] has not been independently re-evaluated (it is absent from [p14]'s re-evaluation).

## 7. Benchmarks and what the numbers mean

[p02] introduced __BenchA__ (__BenchAGloss__), __Metric__, and the fixed protocol that later papers did not in fact keep fixed ([p06], [p11]). It also identified __FailureMode__ as the dominant failure, invisible to aggregate accuracy. [p14], by the same group, argues four years later that __BenchA__ is close to saturated: __Metric__ at __ScaleMid__ has gone from 0.60 [p01] to above 0.90, and the top __ScaleLow__ entries are within noise of the oracle. Its re-evaluation of __Method1__, __Method2__, __Method3__, __Method4__ and __Method5__ on a stratified __BenchB__ (__BenchBGloss__) following [p06]'s recommendation gives three results worth planning around: the __BenchA__ ranking carries over but the gaps shrink by about half; __FailureModeShort__ persists in every method, with __Method3__ reducing it most; and on a new held-out set built for __OpenProblem__, every method scores below 0.15 of __Metric__, at the level of the unmodified baseline.

The practical reading is that __BenchA__ differences of a few points between current methods are not decision-relevant, __BenchB__ with a stated protocol is the better comparison, and the held-out set from [p14] should be reported alongside anything we measure.

## 8. Open problems

1. **__OpenProblem__ (__OpenProblemShort__).** First raised in [p02], which found every method it tried at baseline there. It is then named as an out-of-scope limitation by every subsequent paper in the corpus: [p03], [p04], [p05], [p06], [p07], [p08], [p09, p10], [p12] and [p13] each say they do not address it, [p11] calls it the most important open question and devotes a section to failed approaches, [p10] tested and rejected the one concrete proposal ([p09]'s), and [p14] measured every current method below 0.15 on it. Twelve papers acknowledge it; none solves it; [p13] and [p14] both say that scaling the existing families is unlikely to. This is the single most important fact in the corpus for __Audience__: no method here should be assumed to work in that regime.
2. **__FailureMode__.** Identified in [p02], reduced by each __FamilyA__ generation, still present in every method on __BenchB__ in [p14]. __Method3__ reduces it most but does not remove it.
3. **Protocol inconsistency and benchmark saturation.** [p06] showed that an unstated protocol variable (__ConfoundShort__) flips a headline result; [p11] found protocols vary across the __BenchA__ literature; [p14] finds __BenchA__ saturating. Cross-paper __BenchA__ comparisons without a stated protocol are not reliable, and small __BenchA__ gaps are not meaningful.
4. **Cost-quality trade-offs are under-measured.** Only __FamilyB__ ([p07]/[p08]) reports cost at all, and its quality figure had to be corrected. __Method6__ [p12] costs roughly the sum of its parts; __Method5__'s cost is "dominated by the selection stage" [p09, p10] with no figure given. A decision that depends on cost cannot be made from this corpus alone.
5. **Independent verification is thin.** Only the __FamilyA__ line and __Method4__ have been re-evaluated by other groups ([p02], [p04], [p05], [p06], [p14]). __Method6__ has a single paper with an internally inconsistent number, and __Method5__'s evidence is one group's work in two versions.

## 9. Implications for __Audience__

- If we adopt one __FamilyA__ method, it should be __Method3__ [p13]: it strictly generalises __Method2__ and __Method1__, leads on both __BenchA__ and the independent __BenchB__ re-evaluation [p14], and reduces __FailureModeShort__ most. Budget a calibration set of a few thousand examples.
- Whether the __FamilyA__ gains matter for us depends on whether our deployment contains __FailureModeShort__ cases; [p06] shows the answer can swing an 11-point gain to zero. We should measure under both protocols from [p04] and [p05], not one.
- __Method4__ is worth it only if a roughly 4x cost reduction at __ScaleHigh__ justifies about a 3-point loss [p08]. Do not plan on the "no loss" figure from [p07]; it was withdrawn by its authors.
- __Method5__ [p09, p10] is a modest, setting-dependent addition (about +3 at __ScaleHigh__, nothing at __ScaleLow__) that composes with __FamilyA__ in the form of __Method6__ [p12], whose __BenchB__ gains are plausible but whose __BenchA__ numbers are not usable until the 1.04 figure is explained.
- Nothing in this corpus addresses __OpenProblem__. If our plans require handling that regime, this literature does not support the assumption that any of these methods will, and [p14]'s held-out set is the minimum we should measure against before claiming otherwise.
