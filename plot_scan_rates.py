#!/usr/bin/env python3
"""Requirements 1 and 2: rate and energy rate against a scanned variable.

Two quantities per detector, each against the scan's x variable:
    rate                 [kcps]
    ionizing energy rate [MeV*kcps]

and each in three breakdowns:
    --split none      one curve per beam intensity        (the reference figure)
    --split source    one curve per source per intensity  (Ele / Mu / 1809 / Neutrals)
    --split pdg       one curve per species per intensity (by the ATTRIBUTED particle)

Normalization is in stm_analysis.py: each file is 1e12 POT, scaled to 9.3e12 (1-batch) or
2.3e13 (2-batch) POT/s and divided by 1000 for kcps.

There is no non-ionizing plot: Geant4 never populates nonIonizingEDep in Mu2e, under any
physics list (verified against a ShieldingM campaign). See QUANTITIES below.

What "species" means here: the pdgId branch is the ATTRIBUTED particle, i.e. the VD116 root
under the default grouping. So a curve labelled gamma is energy CAUSED by a photon, not
energy deposited by one - the ionization itself is almost always done by secondaries.

Run as:
    python plot_scan_rates.py                          # default scan, no split
    python plot_scan_rates.py --split source
    python plot_scan_rates.py --scan physics_list --split pdg
"""

from __future__ import print_function

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stm_analysis as A
from run_stm_scans import SCANS, DEFAULT_SCAN, get_scan, SOURCES


# The y quantities: (key, axis label, how to pull it from one booked result).
#
# There is no non-ionizing quantity here, and that is not an omission. Geant4 only fills
# G4Step::GetNonIonizingEnergyDeposit() when a process calls SetNonIonizingEnergyDeposit(),
# which no standard EM or hadronic process does - including under ShieldingM, whose
# high-precision neutron physics improves cross sections and secondaries but performs no
# NIEL partitioning. This was checked against a real ShieldingM campaign, which returned
# zero exactly as QGSP_BERT_EMZ did. Every other consumer of the field in Offline is a
# pass-through, a dumper or a print; none interprets it.
#
# nonIonE is still written to the tree, so the day a process does populate it the plot is
# one entry away.
QUANTITIES = [
    ('rate',   'Rate [kcps]',
     lambda n, sums: n),
    ('edep',   r'Energy rate [MeV$\cdot$kcps]',
     lambda n, sums: sums['E']),
]


def collect(frames, scan, xkey, detector, split, batches):
    """Book every file once, run the loops, and return the curves.

    Structure: {curve label: {'x': [...], batch: [...]}}. Booking is separated from reading
    so each file is traversed a single time no matter how many curves it feeds.
    """
    # Which sub-selections each curve needs. A selection is (label, extra cut).
    # A selection is (key, label, extra cut). The key is what a colour is looked up by -
    # for pdg that is the integer id, since several ids can share a display label.
    if split == 'none':
        selections = [(None, None, None)]
    elif split == 'source':
        selections = [(src, src, None) for src, _ in SOURCES]
    else:
        species = A.species_present(frames, detector=detector, min_counts=10)
        selections = [(pdg, A.pdg_label(pdg), A.pdg_cut(pdg)) for pdg in species]

    det_cut = A.detector_cut(detector)

    # Book first: (campaign, selection label) -> lazy results, plus the per-point file count
    # that the rate denominator needs.
    booked = {}
    file_counts = {}
    for point in scan.points_sorted_by(xkey):
        for (sel_key, sel_label, sel_cut) in selections:
            key = (point.campaign, sel_key)
            booked[key] = []
            n_files = 0
            for (source, campaign), frame in frames.items():
                if campaign != point.campaign:
                    continue
                if split == 'source' and source != sel_key:
                    continue
                cut = det_cut if sel_cut is None else '(%s) && (%s)' % (det_cut, sel_cut)
                booked[key].append(A.book(frame, cut))
                n_files += 1
            file_counts[key] = n_files

    # Read: one event loop per file, covering everything booked on it above.
    totals = {}
    for key, entries in booked.items():
        n_total, sums_total = 0, {'E': 0.0, 'nonIonE': 0.0}
        for n, sums in A.materialize(entries):
            n_total += n
            for c in sums_total:
                sums_total[c] += sums[c]
        totals[key] = (n_total, sums_total)

    return selections, totals, file_counts


def build_curves(scan, xkey, selections, totals, file_counts, quantity_fn, batches, split):
    """(label, color, marker, xs, ys) per curve, from the materialized totals."""
    curves = []
    for sel_key, sel_label, _ in selections:
        for batch in batches:
            xs, ys = [], []
            for point in scan.points_sorted_by(xkey):
                key = (point.campaign, sel_key)
                n_files = file_counts.get(key, 0)
                if n_files == 0:
                    continue   # campaign absent: leave a gap, do not plot a zero
                n, sums = totals[key]
                raw = quantity_fn(n, sums)
                xs.append(point.get(xkey))
                ys.append(A.to_kcps(raw, batch, n_files=n_files))
            if not xs or not any(ys):
                continue

            if split == 'none':
                label = batch
                color = A.BATCH_COLOR_DICT.get(batch)
            elif split == 'source':
                label = '%s (%s)' % (sel_label, batch)
                color = A.SOURCE_COLOR_DICT.get(sel_key, 'grey')
            else:
                label = '%s (%s)' % (sel_label, batch)
                color = A.pdg_color(sel_key)
            marker = 'o' if batch == batches[0] else 's'
            fillstyle = 'full' if batch == batches[0] else 'none'
            curves.append((label, color, marker, fillstyle, xs, ys))
    return curves


def make_figure(curves, scan, xkey, ylabel, detector, outpath, categorical):
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    var = scan.variable(xkey)

    if categorical:
        # Strings have no numeric position; plot against an index and label the ticks.
        labels = [str(x) for x in curves[0][4]]
        positions = list(range(len(labels)))
        for label, color, marker, fillstyle, xs, ys in curves:
            ax.plot(positions[:len(ys)], ys, marker=marker, color=color,
                    fillstyle=fillstyle, linestyle='none', label=label, markersize=8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=15)
    else:
        for label, color, marker, fillstyle, xs, ys in curves:
            ax.plot(xs, ys, marker=marker, color=color, fillstyle=fillstyle,
                    linestyle='-', linewidth=1.5, label=label, markersize=7)

    ax.set_xlabel(var.axis_label())
    ax.set_ylabel(ylabel)
    ax.set_title(A.detector_name(detector))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print('  wrote ' + outpath)
    return outpath


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', default=DEFAULT_SCAN,
                    help='Scan to plot (default: %s). Defined: %s'
                         % (DEFAULT_SCAN, ', '.join(sorted(SCANS))))
    ap.add_argument('--x', default=None,
                    help="Variable for the x axis. Default: the scan's single varying "
                         'variable, or an error if several vary.')
    ap.add_argument('--split', choices=['none', 'source', 'pdg'], default='none',
                    help='none (default): one curve per beam intensity, as the reference '
                         'figure. source / pdg: break the same quantity down.')
    ap.add_argument('--batch', choices=sorted(A.BATCHES), action='append',
                    help='Beam intensity to draw; repeatable. Default: both.')
    ap.add_argument('--sources', nargs='*', help='Restrict to these sources.')
    ap.add_argument('--campaigns', nargs='*', help='Restrict to these campaigns.')
    ap.add_argument('--min-edep', type=float, default=None,
                    help='Drop counts whose TOTAL deposit is below this [MeV].')
    ap.add_argument('--output-root', default=None, help='Where the ROOT files live.')
    ap.add_argument('--outdir', default='plots', help='Where to write figures.')
    ap.add_argument('--format', default='png', help='Figure format (png, pdf, ...).')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    scan = get_scan(args.scan)
    batches = args.batch or A.DEFAULT_BATCHES

    # The x variable has to be unambiguous. A 2-D scan with both variables moving cannot
    # pick one on its own, and guessing would silently project two configurations onto the
    # same x.
    xkey = args.x
    if xkey is None:
        varying = scan.varying()
        if len(varying) == 1:
            xkey = varying[0]
        elif not varying:
            raise SystemExit('Scan %r has only one configuration - nothing to plot against.'
                             % scan.name)
        else:
            raise SystemExit(
                'Scan %r varies %s; pass --x to choose the axis. The others would be summed '
                'over, which is probably not what you want - consider --campaigns to fix '
                'them instead.' % (scan.name, ', '.join(varying)))

    categorical = scan.is_categorical(xkey)
    if categorical:
        print('NOTE: %r is categorical, so the x positions are an ordering with no physical '
              'meaning. Points are drawn unconnected - read them, not the slope.' % xkey,
              file=sys.stderr)

    frames = A.load_scan(scan, sources=args.sources, campaigns=args.campaigns,
                         output_root=args.output_root, min_edep=args.min_edep,
                         verbose=args.verbose)
    # Checks, rather than assumes, that nonIonE is zero; prints only if it is not.
    A.non_ionizing_total(frames)

    os.makedirs(args.outdir, exist_ok=True)
    print('scan %r, x = %s, split = %s, batches = %s'
          % (scan.name, xkey, args.split, ', '.join(batches)))

    suffix = '' if args.split == 'none' else '_by%s' % args.split
    for detector, dname in A.DETECTORS:
        selections, totals, file_counts = collect(
            frames, scan, xkey, detector, args.split, batches)
        for qkey, ylabel, qfn in QUANTITIES:
            curves = build_curves(scan, xkey, selections, totals, file_counts,
                                  qfn, batches, args.split)
            out = os.path.join(args.outdir, '%s_%s_%s%s.%s'
                               % (scan.name, qkey, dname, suffix, args.format))
            make_figure(curves, scan, xkey, ylabel, detector, out, categorical)


if __name__ == '__main__':
    main()
