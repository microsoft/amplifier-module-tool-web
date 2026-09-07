"""SCRATCH ONLY -- deliberate failure proving the CI test job can go red.

This file exists on the throwaway branch ci/red-proof-j1e6 and nowhere else.
Its whole purpose is to make the `Tests` job fail INSIDE the suite, so the red
run's job log reads "12 passed, 1 failed" -- proving the real suite collected
and executed -- rather than failing at setup or lint, which would prove
nothing about whether the tests actually run.

Delete this file and its branch once the red run has been observed.
"""


def test_deliberate_failure_to_prove_ci_goes_red():
    assert 1 == 2, "deliberate red-proof failure (scratch branch only)"
