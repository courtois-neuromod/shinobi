#!/usr/bin/env python
"""Generate annotated BIDS event files for the shinobi dataset.

The event logic lives in ``videogames_utils.events`` so that all four CNeuroMod
videogame datasets share one controlled vocabulary, one set of RAM decode tables and one
validator. This script is only the command-line front end.

Usage:
    python generate_annotations.py -d /path/to/shinobi
    python generate_annotations.py -d . --overwrite
    python generate_annotations.py -d . -sub sub-01 -ses ses-001 -o /tmp/out

Requires the replay sidecars (``gamelogs/*_variables.json``); run
``code/replays/generate_replays.py`` first if they are missing.
"""

import argparse
import sys

from videogames_utils.events import run, validation

TASK = "shinobi"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--datapath", default=".",
                        help="Root of the shinobi dataset.")
    parser.add_argument("-o", "--output_path", default=None,
                        help="Where to write the annotated files. Defaults to writing "
                             "next to each input events.tsv.")
    parser.add_argument("--subjects", "-sub", nargs="+", default=None,
                        help="Subjects to process, e.g. sub-01 sub-02.")
    parser.add_argument("--sessions", "-ses", nargs="+", default=None,
                        help="Sessions to process, e.g. ses-001 ses-002.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate files that already exist.")
    parser.add_argument("--no-sidecar", action="store_true",
                        help="Skip writing task-shinobi_events.json.")
    parser.add_argument("--validate", action="store_true",
                        help="Run the schema and RAM-invariant checks afterwards and "
                             "exit non-zero if any error is found.")
    args = parser.parse_args()

    print(f"Generating shinobi annotations in: {args.datapath}")
    stats = run.annotate_dataset(
        args.datapath, TASK, output_dir=args.output_path,
        subjects=args.subjects, sessions=args.sessions, overwrite=args.overwrite)
    print(f"\n{stats['written']} written, {stats['skipped']} skipped "
          f"(already present), {stats['empty']} with no usable replay")

    if not args.no_sidecar and not args.output_path:
        path = run.write_events_sidecar(args.datapath, TASK)
        print(f"wrote {path}")

    if args.validate:
        target = args.output_path or args.datapath
        report = validation.validate_dataset(target, TASK)
        print("\n" + report.summary())
        for finding in report.findings[:40]:
            print("\n" + str(finding))
        if report.errors:
            print(f"\n{len(report.errors)} error(s)")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
