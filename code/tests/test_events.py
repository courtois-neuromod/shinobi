"""Event-file tests for the shinobi dataset.

Run from the dataset root:

    pytest code/tests/

These are the V0 (schema and controlled vocabulary) and V1 (invariants recomputed from
the raw RAM) layers described in `code/annotations/README.md`. They are skipped when no
annotated events files are present, so a fresh clone without replays still passes.
"""

import os.path as op
from glob import glob

import pandas as pd
import pytest

from videogames_utils.events import validation, vocabulary

TASK = "shinobi"
DATAPATH = op.abspath(op.join(op.dirname(__file__), "..", ".."))

ANNOTATED = sorted(glob(op.join(DATAPATH, "sub-*", "ses-*", "func",
                                "*_desc-annotated_events.tsv")))

pytestmark = pytest.mark.skipif(
    not ANNOTATED, reason="no annotated events files; run generate_annotations.py first")


@pytest.fixture(scope="module")
def report():
    """One validation pass shared by the tests below."""
    return validation.validate_dataset(DATAPATH, TASK)


def _fail(findings):
    return "\n\n".join(str(f) for f in findings[:40])


def test_schema_and_vocabulary(report):
    """V0: every file has the right columns and only known trial_types."""
    bad = [f for f in report.findings if f.layer == "V0" and f.severity == "error"]
    assert not bad, f"{len(bad)} schema problem(s):\n\n{_fail(bad)}"


def test_ram_invariants(report):
    """V1: counts recomputed from _variables.json agree with the events."""
    bad = [f for f in report.findings if f.layer == "V1" and f.severity == "error"]
    assert not bad, f"{len(bad)} invariant violation(s):\n\n{_fail(bad)}"


def test_events_not_empty():
    """Every annotated file has more than a header."""
    empty = [p for p in ANNOTATED if len(pd.read_csv(p, sep="\t")) < 2]
    assert not empty, f"{len(empty)} near-empty events file(s): {empty[:5]}"


def test_sidecar_present_and_complete():
    """task-<task>_events.json exists and covers every trial_type actually emitted."""
    sidecar = op.join(DATAPATH, f"task-shinobi_events.json")
    assert op.exists(sidecar), f"missing {sidecar}"

    import json
    with open(sidecar) as handle:
        levels = set(json.load(handle)["trial_type"]["Levels"])

    used = set()
    for path in ANNOTATED:
        used |= set(pd.read_csv(path, sep="\t", usecols=["trial_type"])["trial_type"])

    undocumented = sorted(
        t for t in used
        if t not in levels and t.rsplit("/", 1)[0] + "/*" not in levels)
    assert not undocumented, f"trial_types missing from the sidecar: {undocumented[:10]}"


def test_every_emitted_type_is_allowed_for_this_task():
    """Guards against a generator leaking another game's events into this dataset."""
    used = set()
    for path in ANNOTATED:
        used |= set(pd.read_csv(path, sep="\t", usecols=["trial_type"])["trial_type"])
    wrong = sorted(t for t in used if not vocabulary.is_valid(str(t), TASK))
    assert not wrong, f"trial_types not allowed for shinobi: {wrong[:10]}"


def test_stim_files_resolve():
    """Every stim_file referenced by the events exists on disk."""
    missing = set()
    for path in ANNOTATED:
        frame = pd.read_csv(path, sep="\t", usecols=["stim_file"])
        for stim in frame["stim_file"].dropna().unique():
            if not isinstance(stim, str) or stim.strip().lower() == "missing file":
                continue
            if not op.exists(op.join(DATAPATH, stim)):
                missing.add(stim)
    assert not missing, f"{len(missing)} unresolvable stim_file(s): {sorted(missing)[:5]}"
