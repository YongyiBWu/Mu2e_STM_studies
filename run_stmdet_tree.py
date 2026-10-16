#!/usr/bin/env python3
"""Driver for STMDetTree over a (source x scan point) grid.

For each catalog of stage-2 .art files, write a temp fcl that includes
Offline/STMMC/fcl/STMDetTree.fcl and overrides the output path, then run mu2e over the
catalog with -S. Full output goes to a per-run log; the endJob summary is extracted into its
own short file beside it.

The scan is NOT defined here
----------------------------
A CAMPAIGN (stmedep00, ...) is one production at one fixed configuration, holding the four
source datasets; a SCAN is the set of campaigns whose configurations differ. Which campaigns
belong to a scan, what configuration each stands for, and how the file names are built all
live in run_stm_scans.py - because the analysis scripts need exactly the same mapping to turn a
produced file back into (source, campaign), and the STMDetTree module records neither. Pick
a scan with --scan; add a configuration by adding a CampaignPoint there, not by editing this
file.

What comes out, per (source, campaign)
--------------------------------------
  <output_dir>/STMDet_tree.<source>_<campaign>.root      ttree + ttree_genealogy
  <log_dir>/STMDet_tree.<source>_<campaign>.log          full mu2e output
  <log_dir>/STMDet_tree.<source>_<campaign>.summary.txt  just the endJob summary

Input files must contain compressSTMDet products (StepPointMCs in STMDet) -- NOT *ToVD*
files, which stop at the virtual detector and have no crystal energy deposits. The module
throws if StepPointMCsTag and SimParticlemvTag name different compression chains, so a
mis-paired catalog fails loudly on the first event rather than producing wrong attribution.
"""

import argparse
import datetime
import os
import subprocess
import sys

from run_stm_scans import SOURCES, SCANS, DEFAULT_SCAN, get_scan

# The STMDetTree endJob banner. Everything between these two lines is the summary worth
# reading; the rest of a mu2e log is startup chatter and per-event messages. The start
# line's longest '=' run is shorter than the end line, so the end test cannot match the
# start line and terminate the capture immediately.
SUMMARY_START = '========== STMDet tree summary =========='
SUMMARY_END = '========================================='


def extract_summary(log_path, summary_path, label):
    """Pull the endJob summary out of a full mu2e log into its own short file.

    Returns the summary text, or None when the banner never appeared - which is itself worth
    reporting, since a job can exit 0 having produced no summary (no input events reached the
    analyzer, or the log was truncated).

    Scanning line by line rather than reading the whole log: these can reach hundreds of MB
    and only a few dozen lines are wanted.
    """
    lines = []
    capturing = False
    try:
        # mu2e logs are ASCII but a truncated write can leave an invalid byte; replacing
        # avoids losing the whole summary to one bad character.
        with open(log_path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if SUMMARY_START in line:
                    capturing = True
                    lines = [line]          # restart: keep the LAST summary if several
                    continue
                if capturing:
                    lines.append(line)
                    if SUMMARY_END in line:
                        capturing = False   # keep scanning in case a later one exists
    except OSError as e:
        print(f'!!! {label}: could not read log {log_path}: {e}', file=sys.stderr)
        return None

    if not lines:
        return None

    text = ''.join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'# {label}: STMDetTree summary extracted from\n# {log_path}\n\n')
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', default=DEFAULT_SCAN,
                    help=f'Which scan to run (default: {DEFAULT_SCAN}). '
                         f'Defined in run_stm_scans.py: {", ".join(sorted(SCANS))}')
    ap.add_argument('--sources', nargs='*', metavar='LABEL',
                    help='Subset of source labels to run (default: all). '
                         f'Choices: {", ".join(s for s, _ in SOURCES)}')
    ap.add_argument('--campaigns', nargs='*', metavar='NAME',
                    help='Subset of campaigns to run (default: all in the scan). Run with '
                         '--list to see them.')
    ap.add_argument('--list', action='store_true',
                    help='Print the selected scan\'s variables and points, then exit.')
    ap.add_argument('--max-events', type=int, default=None,
                    help='Cap events per run (-n). The summary tallies stay meaningful on a '
                         'partial sample, but RATES do not - do not plot a capped run.')
    ap.add_argument('--no-genealogy', action='store_true',
                    help='Set WriteGenealogy to false. Much smaller output; only do this if '
                         'the back-trace plots are not wanted from this run.')
    ap.add_argument('--group-by-top-parent', action='store_true',
                    help='Set GroupByTopParentInDetector to true, reproducing HPGeTree\'s '
                         'grouping with time clustering DISABLED. For comparing against the '
                         'old modules, not for analysis.')
    ap.add_argument('--min-edep', type=float, default=None,
                    help='Override MinEdep [MeV]. Cutting here cannot be undone downstream; '
                         'prefer cutting in the analysis.')
    ap.add_argument('--keep-fcl', action='store_true',
                    help='Leave the generated temp fcl in place for inspection.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Write the temp fcl and print the command without running mu2e.')
    ap.add_argument('--output-root', default=None,
                    help='Override the output root; the scan still gets its own '
                         'subdirectory under it.')
    ap.add_argument('--log-dir', default=None,
                    help='Override the log directory (default: <output-dir>/logs).')
    ap.add_argument('--tee', action='store_true',
                    help='Also stream mu2e output to the terminal while logging it. Off by '
                         'default - the full log overflows the console.')
    ap.add_argument('--timestamp', action='store_true',
                    help='Suffix log names with the run time instead of overwriting the '
                         'previous log for that run.')
    ap.add_argument('--skip-existing', action='store_true',
                    help='Skip a (source, point) whose output ROOT file already exists. '
                         'Useful for resuming an interrupted grid.')
    args = ap.parse_args()

    scan = get_scan(args.scan)
    output_dir = scan.output_dir(args.output_root)
    log_dir = args.log_dir or os.path.join(output_dir, 'logs')

    if args.list:
        print(f'scan {scan.name!r}')
        print('  variables : ' + ', '.join(v.axis_label() for v in scan.variables.values()))
        print(f'  varying   : {", ".join(scan.varying()) or "(nothing - single campaign)"}')
        print('  campaigns :')
        for p in scan.points:
            vals = ', '.join(f'{k}={scan.variable(k).value_label(v)}'
                             for k, v in sorted(p.values.items()))
            tail = f'   # {p.note}' if p.note else ''
            print(f'    {p.campaign:12s} {vals}{tail}')
            print(f'      catalogs: {scan.catalog_dir(p.campaign)}')
        print(f'  output    : {output_dir}')
        return

    selected_sources = SOURCES
    if args.sources:
        known = {s for s, _ in SOURCES}
        unknown = [s for s in args.sources if s not in known]
        if unknown:
            raise SystemExit(f'Unknown source label(s): {", ".join(unknown)}. '
                             f'Choices: {", ".join(sorted(known))}')
        selected_sources = [(s, t) for s, t in SOURCES if s in args.sources]

    selected_points = scan.points
    if args.campaigns:
        known = set(scan.campaigns())
        unknown = [c for c in args.campaigns if c not in known]
        if unknown:
            raise SystemExit(f'Campaign(s) not in scan {scan.name!r}: '
                             f'{", ".join(unknown)}. In this scan: '
                             f'{", ".join(scan.campaigns())}')
        selected_points = [p for p in scan.points if p.campaign in args.campaigns]

    os.makedirs(output_dir, exist_ok=True)
    if not args.dry_run:
        os.makedirs(log_dir, exist_ok=True)

    stamp = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S') if args.timestamp else ''

    failures, no_summary, skipped = [], [], []
    summaries = {}
    # Source-major so one source's scan completes together.
    runs = [(s, t, p) for s, t in selected_sources for p in selected_points]
    print(f'scan {scan.name!r}: {len(runs)} run(s) = {len(selected_sources)} source(s) x '
          f'{len(selected_points)} campaign(s)')

    for source, dataset_tag, point in runs:
        label = f'{source}_{point.campaign}'
        catalog_file = scan.catalog_path(dataset_tag, point.campaign)
        outfilename = scan.output_path(source, point.campaign, args.output_root)
        base = f'STMDet_tree.{label}{stamp}'
        log_path = os.path.join(log_dir, base + '.log')
        summary_path = os.path.join(log_dir, base + '.summary.txt')

        vals = ', '.join(f'{k}={scan.variable(k).value_label(v)}'
                         for k, v in sorted(point.values.items()))
        print(f'\n=== {label}  ({vals}) ===')

        if args.skip_existing and os.path.isfile(outfilename):
            print(f'Output exists, skipping: {outfilename}')
            skipped.append(label)
            continue

        if not os.path.isfile(catalog_file):
            # Not fatal: a grid may legitimately be missing one corner, and failing the whole
            # scan over it would waste the runs that can proceed.
            print(f'!!! catalog not found, skipping: {catalog_file}', file=sys.stderr)
            failures.append(label)
            continue
        with open(catalog_file) as f:
            nFiles = sum(1 for line in f if line.strip())
        if nFiles == 0:
            print(f'!!! catalog is empty, skipping: {catalog_file}', file=sys.stderr)
            failures.append(label)
            continue
        print(f'{catalog_file}: {nFiles} input files')

        tempfcl = f'temp_stmdettree_{scan.name}_{label}.fcl'
        with open(tempfcl, 'w') as f:
            f.write('#include "Offline/STMMC/fcl/STMDetTree.fcl"\n')
            f.write(f'services.TFileService.fileName: "{outfilename}"\n')
            if args.no_genealogy:
                f.write('physics.analyzers.STMDetTree.WriteGenealogy: false\n')
            if args.group_by_top_parent:
                f.write('physics.analyzers.STMDetTree.GroupByTopParentInDetector: true\n')
            if args.min_edep is not None:
                f.write(f'physics.analyzers.STMDetTree.MinEdep: {args.min_edep}\n')

        cmd = ['mu2e', '-c', tempfcl, '-S', catalog_file]
        if args.max_events is not None:
            cmd += ['-n', str(args.max_events)]
        print(f'Running command: {" ".join(cmd)}')

        if args.dry_run:
            print(f'(dry run - temp fcl left at {tempfcl})')
            continue

        print(f'Log: {log_path}')
        try:
            # shell=False with a list: paths here are user-supplied and may contain
            # characters a shell would reinterpret.
            #
            # stderr is folded into the same log as stdout rather than split off: mu2e
            # interleaves MessageFacility output across both, and two files would have to be
            # re-merged by hand to read a failure in the order it happened.
            with open(log_path, 'w', encoding='utf-8') as log:
                if args.tee:
                    # Line-buffered read so the terminal keeps up with a long job.
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT,
                                            text=True, bufsize=1)
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        log.write(line)
                    proc.stdout.close()
                    rc = proc.wait()
                else:
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
            print(f'--> {outfilename}')
        except subprocess.CalledProcessError as e:
            # Keep going so one bad run does not cost the rest of the grid; the failures are
            # reported together at the end and set the exit status.
            print(f'!!! {label} failed with exit code {e.returncode} - see {log_path}',
                  file=sys.stderr)
            failures.append(label)
        except OSError as e:
            # mu2e not on PATH, log directory unwritable, and similar: not a job failure.
            print(f'!!! {label} could not be launched: {e}', file=sys.stderr)
            failures.append(label)
        finally:
            if not args.keep_fcl and os.path.exists(tempfcl):
                os.remove(tempfcl)

        # Extract the summary even when the job failed: a run that died after endJob still
        # wrote one, and that is exactly when it is most worth reading.
        if os.path.exists(log_path):
            text = extract_summary(log_path, summary_path, label)
            if text is None:
                no_summary.append(label)
                print('    (no STMDetTree summary found in the log)')
            else:
                summaries[label] = summary_path
                print(f'Summary: {summary_path}')

    if not args.dry_run:
        print('\n================ grid complete ================')
        if summaries:
            print(f'{len(summaries)} summary file(s) written under {log_dir}')
        if skipped:
            print(f'Skipped (output existed): {", ".join(skipped)}')
        if no_summary:
            print(f'\nNo summary found for: {", ".join(no_summary)}')
            print('  A job can exit 0 without producing one - no input events reached the '
                  'analyzer,\n  or the log was truncated. Check the log tail before reading '
                  'anything into it.')
        print('\nPer run, check in the summary:\n'
              '  "not assigned to either" > 0 -> crystal volume names do not match the '
              'geometry;\n'
              '                                  that energy is MISSING from the trees\n'
              '  counts vs attributed particles -> how much the time clustering split\n'
              '  counts by PDG                  -> the per-species breakdown the plots use')
        if args.max_events is not None:
            print(f'\nNOTE: runs were capped at {args.max_events} events. Spectra and '
                  'fractions are\n  still meaningful; RATES are not - do not plot these.')

    if failures:
        raise SystemExit(f'\nFailed run(s): {", ".join(failures)}')


if __name__ == '__main__':
    main()
