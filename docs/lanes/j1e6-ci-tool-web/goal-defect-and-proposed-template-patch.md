# Goal defect: the three "exhaustive" outcome branches all presuppose custody

**Filed against the per-lane goal template for `model_performance-j1e6`, per GOAL.md's own
instruction:** *"that is a DEFECT IN THIS GOAL, not a task. Report it against the goal, ship the
patch as an artifact under your ARTIFACT ROOT, and resolve — do not edit another repo, and do not
invent a fourth outcome branch."*

This artifact is that report and that patch. It edits nothing outside this lane's artifact root.

---

## The defect, stated plainly

**Procedure 5 is not satisfied by this lane and cannot be.** GOAL.md requires, verbatim:

```
work_resolve(id="model_performance-j1e6", reason="<3-6 lines for the owner: ...>")
```

and OUTCOME branch **A** requires *"Work item `model_performance-j1e6` … **is resolved** with a
user-readable summary."*

Neither happened for this repo. Not "happened differently" — **did not happen.**

### All three branches are unreachable, not just A

The goal calls its three outcomes **exhaustive**. For a lane that does not hold the item, all three
terminate in a verb that requires custody:

| Branch | Terminal verb | Requires custody? | Reachable here? |
|---|---|---|---|
| **A** RESOLVED | `work_resolve` | yes | **no** — refused |
| **B** RESOLVED AT THE CAP | `work_resolve` | yes | **no** — and the cap never bound anyway |
| **C** BLOCKED | `work_release` | yes (GOAL.md: *"Release while you still HOLD the item"*) | **no** — refused, measured (below) |

So the exhaustive set is empty for 18 of 19 lanes. That is the defect.

### Measured, not inferred

Every refusal below was produced by actually making the call, not reasoned about:

```
work_claim(item_id="model_performance-j1e6")     -> already claimed by agent-spark-1-1101253
work_claim(item_id="model_performance-j1e6")     -> already claimed by agent-spark-1-2996730
work_claim(item_id="model_performance-j1e6")     -> already claimed by agent-spark-1-2996730
work_resolve(id="model_performance-j1e6", ...)   -> not currently holding 'model_performance-j1e6'
                                                    in this session -- refusing to resolve an item
                                                    this session did not claim
work_release(id="model_performance-j1e6")        -> not currently holding 'model_performance-j1e6'
                                                    in this session -- refusing to release an item
                                                    this session did not claim
work_status                                      -> holding: null
```

**All three verbs are now measured, none inferred.** Branch C's `work_release` was the last one this
lane had only *argued* was unreachable; it was called, and it refused for the same custody reason as
`work_resolve`. The asymmetry is worth naming because it is the failure mode this whole artifact is
about: a refusal you reasoned your way to is not a refusal you observed, and this lane had to be
pushed twice before it stopped asserting and started calling.

**A second, sharper fact.** `work_list(project="model_performance", status="held")` returns exactly
one item — `model_performance-ytja` — and `work_stats` reports `held: 1`. **`j1e6` is not in live
custody at all.** Its holder field is *stale*: it reads `agent-spark-1-2996730` on an item whose
status is `resolved`. The claim fence fires off that stale field, so **nobody can claim `j1e6` —
not a lane, not the manager — without reopening it first.** The item is simultaneously
"already claimed" and held by no one.

### And a third blocker behind the first two

`j1e6` is already `resolved` (`closed_at 2026-09-07T18:14:01Z`) over a resolution covering
`amplifier-bundle-wayfinder` only. Under the immutable-resolution rule, `work_resolve` on an
already-resolved item is an idempotent no-op **only** for byte-identical text; differing text fails
non-zero and writes nothing. So even with custody, this repo's summary could not land there.

---

## What this lane did instead, and what it is NOT

Filed `model_performance-f3h5` — a per-repo child, linked `relates-to` the parent, claimed and
**resolved** at 2026-09-07T20:15:37Z carrying this repo's full user-readable summary.

**This is a mitigation, not compliance.** It does not satisfy Procedure 5, and this lane does not
claim it does. Its value is narrow and real: the repo's result now lives on a resolved row a later
reader can find by id, instead of only inside an erratum on someone else's item.

The precedent is this batch's own — `model_performance-hgdi` (converge) and `model_performance-md4i`
(stories) are the same shape; md4i's resolution records it verbatim: *"This item exists only because
kp79 could not be held by its own lane."*

## Why `work_reopen` was NOT used — the decisive reason is not the cost

The mechanical costs are real but secondary: reopening clears `closed_at`, re-lands the item on
today's date, and moves every throughput roll-up by one item.

The decisive reason is different. **Resolving the parent with this repo's summary would be lateral,
not progress.** It would replace a wayfinder-only resolution with a tool-web-only resolution on an
item covering nineteen repos. Neither text is right; swapping one partial for another makes the
record worse for a wayfinder reader and no better for anyone else.

The parent should be resolved **once, at batch close, with a summary covering all nineteen repos.**
That is the manager's call, and it is the only resolve that improves the record.

---

## Proposed template patch

### 1. Procedure 1 — a refused claim is not a blocker on a multi-lane item

> **Replace:** *"If the claim is refused (held elsewhere / blocked), write BLOCKED.md, commit, write
> the completion marker, stop."*
>
> **With:** *"Claim if free. If a sibling lane holds it, that is the DESIGNED steady state of a
> one-item/many-lanes item, not a blocker — do NOT write BLOCKED.md. Read the authoritative spec with
> `work_list(item_id=...)` (returns description + acceptance with no claim, no mutation, no custody)
> and proceed."*

### 2. Procedure 5 — give the non-holding lane a terminal verb it can actually execute

> **Add:** *"If you hold the item, `work_resolve` it. If a sibling holds it, file a PER-REPO CHILD
> item with `work_add`, link it `relates-to` the parent with `work_dep`, claim it, and `work_resolve`
> THAT — it carries your repo's result on a row a reader can find by id. Then say plainly in your
> marker that the PARENT is not resolved by you. The parent is a container the MANAGER resolves once
> at batch close, with a summary covering every repo."*

### 3. The outcome branches — stop calling three custody-gated branches exhaustive

> **Add a fourth, and say it is not a failure state:** *"**D. DELIVERED, PARENT NOT RESOLVABLE BY
> THIS LANE.** Every deliverable is DONE and shipped for landing, and the parent item's terminal verb
> is unreachable because this lane never held it. This is the NORMAL end state for a non-holding lane
> on a multi-lane item. Record `procedure_5_satisfied: false` at marker top level so no gate can read
> it as A."*

*(GOAL.md forbids a lane from inventing a fourth branch. This is a proposed amendment to the
template, offered as an artifact for the manager to accept or reject — not a branch this lane
awarded itself.)*

### 4. Better than all of the above: file one item per repo

Nineteen repos, nineteen items, each claimable by its own lane. Every problem in this artifact
disappears. The batch already proved the shape works — `hgdi`, `md4i`, and now `f3h5`.

### 5. Fix the stale-holder fence (a work-tracker defect, not a goal defect)

An item that is `resolved` should not keep a live holder field that makes it unclaimable. Today
`j1e6` refuses every claim naming a holder that `work_list(status="held")` says does not hold it.

---

## This lane's own defect, recorded rather than hidden

**I reclassified this lane's terminal state three times while no measurement changed:**

1. `outcome_branch: A`, `status: resolved`
2. `status: deliverables complete — NOT resolved by this lane`
3. `outcome_branch: A`, `status: RESOLVED … via the child item`
4. *(this correction)* `procedure_5_satisfied: false`

That is precisely the churn GOAL.md warns about — *"Choose the terminal state ONCE… lane 1ru moved
BLOCKED -> REJECT -> BLOCKED under an ambiguous goal with its measurement never changing, and that
churn was produced entirely by the goal text."*

Two causes, and only one of them is the goal's. The goal's terminal procedure is genuinely
unsatisfiable here. But steps 1 and 3 were **mine**: each time, I let "which branch can I claim?"
lead instead of "what is true?", and both times a reviewer had to push back. The classification below
is the accurate one and it does not move again.

**Terminal state: the CI work is complete, verified and shipped for landing. Procedure 5 is NOT
satisfied. Both are true, and the second is a defect in the goal, not a gap in the work.**
