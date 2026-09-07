## What this adds

This repo had **no `.github` directory at all**. Every change merged here — including `263a88e` (#15), which shipped the measured lean `web_fetch` description **together with a byte-for-byte pin test guarding it** — landed with `license/cla` as the only check. That pin exists precisely so the measured description cannot silently drift back, and **until this PR nothing ever executed it.**

`.github/workflows/ci.yml` gives **four checks from two job definitions**, on `push: main` and on every `pull_request`:

| Check | What it runs |
|---|---|
| **Lint** | `uv run --frozen --group dev ruff check --isolated --select E4,E7,E9,F .` — ruff **pinned 0.16.6**, rule set pinned |
| **Tests (py3.11)** | `uv sync --frozen --group dev` then `uv run --frozen --no-sync pytest -q` |
| **Tests (py3.12)** | " |
| **Tests (py3.13)** | " |

**No path filters, no `continue-on-error`, no `|| true`.** Verified by `grep -nE 'paths:|paths-ignore:|continue-on-error|\|\| true'` on the committed file → **0 matches**, including in prose.

## What the suite actually covers — 13 tests, NOT an import smoke

| Source | Count | What |
|---|---|---|
| `tests/test_behavioral.py` | 7 | inherited from `amplifier_core.validation.behavioral.ToolBehaviorTests` — mount, tool name/description/execute, `ToolResult` shape, invalid-input handling |
| `tests/test_tool_description_pin.py` | 5 | byte-for-byte pins on the shipped `web_fetch` (549 chars) and `web_search` (30 chars) descriptions, plus the binary-content clause a fuzzy patch once silently dropped |
| `tests/test_validation.py` | 1 | inherited from `amplifier_core.validation.structural.ToolStructuralTests` |

## Red-then-green, proven

**RED — https://github.com/microsoft/amplifier-module-tool-web/actions/runs/34157444051**

Scratch branch `ci/red-proof-j1e6` carried two deliberate defects, one per job. The **Tests** job log on **all three Pythons** reads:

```
1 failed, 13 passed in 0.45s
FAILED tests/test_red_proof_DELIBERATE_FAILURE.py::test_deliberate_failure_to_prove_ci_goes_red
```

The `13 passed` alongside the failure is the load-bearing part: the real suite **collected and executed**, and the red is a genuine **test** failure — not a setup or lint error, which would have proved nothing. The **Lint** job failed separately and for its own reason (`F821 Undefined name`, `Found 1 error.`).

Scratch PR **#16 CLOSED**, branch **`ci/red-proof-j1e6` DELETED** — verified by remote read (`git ls-remote --heads origin ci/red-proof-j1e6` → **0 refs**), not by trusting a success message.

**GREEN — https://github.com/microsoft/amplifier-module-tool-web/actions/runs/34157597450**

All four checks green on `bc022d7` (packaging fix + workflow only): Lint `All checks passed!`, Tests `13 passed` on 3.11, 3.12 and 3.13.

Both job logs are committed verbatim/excerpted under `docs/lanes/j1e6-ci-tool-web/evidence/`.

## The one source change, and why it was unavoidable

`56a2010` adds a `[dependency-groups] dev` block. **A clean checkout of this repo could not run its own tests**: `uv sync` installed no pytest, no pytest-asyncio and no amplifier-core, so `uv run pytest` failed with

```
error: Failed to spawn: `pytest`
  Caused by: No such file or directory (os error 2)
```

All three have always been required — `tests/` imports `amplifier_core.validation.*` and relies on amplifier-core's pytest plugin for the module fixtures, and the pre-existing `asyncio_mode = "strict"` is a pytest-asyncio setting for a plugin that was never installed. It went unnoticed because nothing here has ever executed the suite, and every developer who ran it locally did so from an ambient environment that happened to carry amplifier-core.

Wiring CI around that with a CI-only `pip install` would have made CI and local disagree, so it is fixed at the source instead, in its own named commit. Shape and contents mirror **amplifier-module-provider-openai**, the sibling reference module, which already declares exactly this group.

Fail-before / pass-after, measured on a fresh environment per interpreter:

```
before:  uv sync && uv run pytest        ->  "Failed to spawn: pytest"
after:   uv sync --frozen --group dev
         uv run --frozen --no-sync pytest -q
         ->  13 passed (3.11), 13 passed (3.12), 13 passed (3.13)
```

No source file is touched; no test is changed or added.

## Clean main was green on the gate as wired

The pinned invocation (`ruff 0.16.6 --isolated --select E4,E7,E9,F`) reports **`All checks passed!`** on this tree, so the stop-and-report branch did not trigger. Two things sit deliberately **outside** the gate and are disclosed rather than papered over:

- ruff 0.16.6's **full modern default tier** would be red at **14 findings** — 8 `BLE001` (blind except), 4 `UP045` (`Optional[X]` → `X | None`), 2 `I001` (unsorted imports). None is breakage; adopting any of it is a source change and out of scope for a workflow PR.
- `ruff format --check` would reformat **2 of 10 files**. Also a source change, also not run by this workflow.

## Notes for the next CI lane

- **`uv sync --frozen` is safe here** — this repo commits `uv.lock`. It is *not* safe in the lockfile-less sibling repos, where `--frozen` (and `setup-uv`'s `enable-cache: true`, which keys on `**/uv.lock`) hard-fails before ruff or pytest ever runs.
- **Do not write `# noqa: <CODE>` inside the prose of a deliberate red-proof defect.** My first planted `F821` carried the string `# noqa: F821` in an explanatory comment; ruff honoured it as a real suppression and the Lint job would have passed while "proving" it could fail. Caught by running the pinned command locally before pushing.
