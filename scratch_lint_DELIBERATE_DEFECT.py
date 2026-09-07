"""SCRATCH ONLY -- deliberate F821 proving the CI lint job can go red.

Lives only on ci/red-proof-j1e6. Not imported by anything, so it cannot
disturb pytest collection: the test job's failure must come from the test
file, and this file's failure must come from ruff.
"""


def deliberate_undefined_reference():
    return this_name_is_not_defined_anywhere
