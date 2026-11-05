#!/usr/bin/env python3
"""How many STMDetTree rows carry no energy at all?

Every STMDet step enters the time clustering, including steps that deposited nothing - a
neutron crossing the crystal, say. A cluster made only of such steps is written as a row
with totalE == 0, and its energy-weighted position falls back to (0, 0, 0), which is a real
coordinate ~40 m upstream rather than a sentinel. MinEdep : 0 keeps those rows.

This counts them, per (campaign, source, detector), so the decision about what the module
should do with them is made against the actual fraction rather than a guess. It reads only;
nothing is modified.

Run as:
    python check_zero_deposit.py                       # default scan
    python check_zero_deposit.py --scan old_geom_ssc_area
    python check_zero_deposit.py --campaigns stmedep10
"""

from __future__ import print_function

import argparse

import stm_analysis as A
from run_stm_scans import SCANS, DEFAULT_SCAN, get_scan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', default=DEFAULT_SCAN,
                    help='Scan to read (default: %s). Defined: %s'
                         % (DEFAULT_SCAN, ', '.join(sorted(SCANS))))
    ap.add_argument('--sources', nargs='*', help='Restrict to these sources.')
    ap.add_argument('--campaigns', nargs='*', help='Restrict to these campaigns.')
    ap.add_argument('--output-root', default=None, help='Where the ROOT files live.')
    ap.add_argument('--tree-dir', default=A.TREE_DIR,
                    help='TFileService directory holding the trees (default: %s).'
                         % A.TREE_DIR)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    scan = get_scan(args.scan)
    # No min_edep: the point is to see what a default run actually wrote.
    frames = A.load_scan(scan, sources=args.sources, campaigns=args.campaigns,
                         output_root=args.output_root, directory=args.tree_dir,
                         verbose=args.verbose)

    # Book every count before reading any, so each file is traversed once.
    booked = {}
    for key, frame in frames.items():
        per_det = {}
        for det, dname in A.DETECTORS:
            node = frame.Filter(A.detector_cut(det))
            zero = node.Filter('E + nonIonE <= 0.0')
            per_det[dname] = {
                'all':        node.Count(),
                'zero':       zero.Count(),
                # Of the zero rows, how many sit at the (0,0,0) fallback. Should be all of
                # them; a mismatch would mean the fallback is firing somewhere else too.
                'zero_at_origin': zero.Filter('x == 0.0 && y == 0.0 && z == 0.0').Count(),
                # Energy in the non-zero rows, to show what the zero ones are NOT carrying.
                'edep':       node.Sum('E'),
                # Steps, to see whether a zero row is one lone step or a real traversal.
                'zero_steps': zero.Sum('nSteps'),
            }
        booked[key] = per_det

    print('scan %r' % scan.name)
    print('%-12s %-10s %-6s %10s %10s %8s %10s %10s'
          % ('campaign', 'source', 'det', 'rows', 'zero', '%', 'zero@orig', 'MeV'))
    print('-' * 86)

    totals = {}
    for (source, campaign) in sorted(booked, key=lambda k: (k[1], k[0])):
        for det, dname in A.DETECTORS:
            r = booked[(source, campaign)][dname]
            n_all = int(r['all'].GetValue())
            n_zero = int(r['zero'].GetValue())
            n_orig = int(r['zero_at_origin'].GetValue())
            edep = float(r['edep'].GetValue())
            pct = (100.0 * n_zero / n_all) if n_all else 0.0
            print('%-12s %-10s %-6s %10d %10d %7.2f%% %10d %10.4g'
                  % (campaign, source, dname, n_all, n_zero, pct, n_orig, edep))
            t = totals.setdefault(dname, [0, 0, 0, 0.0, 0])
            t[0] += n_all
            t[1] += n_zero
            t[2] += n_orig
            t[3] += edep
            t[4] += int(r['zero_steps'].GetValue())

    print('-' * 86)
    for dname, (n_all, n_zero, n_orig, edep, zsteps) in totals.items():
        pct = (100.0 * n_zero / n_all) if n_all else 0.0
        print('%-12s %-10s %-6s %10d %10d %7.2f%% %10d %10.4g'
              % ('TOTAL', '', dname, n_all, n_zero, pct, n_orig, edep))
        if n_zero:
            print('%46s mean steps per zero row: %.2f'
                  % ('', float(zsteps) / n_zero))

    print('\nrows      - counts written by STMDetTree for that detector')
    print('zero      - of those, rows with E + nonIonE == 0: a cluster whose steps all')
    print('            deposited nothing, e.g. a neutron crossing the crystal')
    print('zero@orig - zero rows whose position is exactly (0,0,0), the fallback. Expect')
    print('            this to equal "zero"; anything less means the fallback also fires')
    print('            elsewhere and the diagnosis is incomplete')
    print('MeV       - total ionizing energy in ALL rows of that detector, for scale')
    print('\nWhat the fraction means: rows are what plot_scan_rates.py counts, so the zero')
    print('percentage is how much the quoted kcps overstates the real pulse rate.')


if __name__ == '__main__':
    main()
