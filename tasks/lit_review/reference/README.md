# __TopicShort__ literature pack

This folder holds a small corpus of abstracts on __Topic__, collected for
__Audience__. The corpus covers the three method families the team is
weighing (__FamilyA__, __FamilyB__ and __FamilyC__) plus the benchmarks
and evaluation papers those methods are measured against.

Files:

- `papers/p01.md` .. `papers/p14.md`: one file per paper with its title, authors,
  venue and abstract. Only the abstracts were collected; full texts are not
  available, so treat each abstract as the paper's own statement of its result.
- `papers/index.csv`: one row per paper with columns `id`, `year`, `citations`
  (citation count at the time of collection).

Deliverable: a literature review at `review.md` (repository root) for
__Audience__. It should synthesise the corpus by theme rather than summarise
it paper by paper, cite every paper by its id (for example `[p03]`), say what
the evidence does and does not establish about each method family, say which
result to trust where papers disagree, and end with a section on open problems
that the team should not expect this literature to have solved.
