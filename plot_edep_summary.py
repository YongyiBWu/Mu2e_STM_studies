#!/usr/bin/env python3
"""Requirement 3: per-detector summary figures from STMDetTree output.

The counterpart of edep_plot.py for the new trees. For each detector, a 3-panel figure:

    counts vs time          (hit rate within a microbunch, folded)
    total Edep vs time      (energy deposited per time bin)
    energy spectrum         (counts vs E)

plus the per-source and per-species copies requirement 3 asks for. Where edep_plot.py read
Stage2HPGe/ttree and Stage2LaBr/ttree and split by directory, this reads one STMDetTree
`ttree` and splits on the `detector` branch - and can therefore also split by the attributed
particle, which the old trees did not record.

What a row is
-------------
One COUNT: a time-clustered deposit, i.e. one pulse the detector would digitize, not one
Geant4 step. See the STMDetTree module header. The spectrum is therefore a per-pulse energy
spectrum, which is what a real spectrum is, rather than a per-step one. This is the main
difference from edep_plot.py's output and the reason the two are not directly comparable.

Axes are COUNTS, not rates
--------------------------
Unlike plot_scan_rates.py these panels are raw counts per bin. A rate needs the POT
normalization, which is per (source, campaign) - and these figures deliberately sum whatever
was selected, possibly several campaigns at different configurations. The totals printed
alongside always describe exactly what was summed; --single-campaign refuses the ambiguous
case outright.

Run as:
    python plot_edep_summary.py                              # default scan, all sources
    python plot_edep_summary.py --split source
    python plot_edep_summary.py --split pdg --campaigns stmedep00
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stm_analysis as A
from run_stm_scans import SCANS, DEFAULT_SCAN, get_scan, SOURCES

# Time axis, folded into one microbunch as edep_plot.py does: the fold puts the prompt peak
# and the late tail on one readable axis.
T_MIN, T_MAX, T_NBINS = 0.0, 1700.0, 170
T_PERIOD = 1700.0

# Spectrum axis. Wider than edep_plot.py's 0-20 MeV because grouping by attributed particle
# sums a whole shower into one count, so the high end carries more than the old
# per-top-parent rows did. Overflow is counted and reported rather than silently clipped.
E_MIN, E_MAX, E_NBINS = 0.0, 20.0, 400


def fetch(frame, cut, use_total):
    """(times, energies) as numpy arrays for one selection, in one pass over the file."""
    node = frame.Filter(cut) if cut else frame
    expr = 'E + nonIonE' if use_total else 'E'
    node = node.Define('_plot_energy', expr)
    times = node.Take['double']('time')
    energies = node.Take['double']('_plot_energy')
    return np.asarray(times.GetValue()), np.asarray(energies.GetValue())


def gather(frames, detector, split, min_counts, use_total=False):
    """[(label, color, times, energies), ...] for one detector.

    Selections are resolved first and every Take booked before any is read, so each file is
    traversed once regardless of how many series draw from it.
    """
    det_cut = A.detector_cut(detector)

    if split == 'none':
        selections = [('all', None, A.BATCH_COLOR_DICT['1-batch'])]
    elif split == 'source':
        selections = [(src, None, A.SOURCE_COLOR_DICT.get(src, 'grey'))
                      for src, _ in SOURCES]
    else:
        species = A.species_present(frames, detector=detector, min_counts=min_counts)
        selections = [(A.pdg_label(p), A.pdg_cut(p), A.pdg_color(p)) for p in species]

    out = []
    for label, extra_cut, color in selections:
        times, energies = [], []
        for (source, _campaign), frame in frames.items():
            if split == 'source' and source != label:
                continue
            cut = det_cut if extra_cut is None else '(%s) && (%s)' % (det_cut, extra_cut)
            t, e = fetch(frame, cut, use_total)
            if len(t):
                times.append(t)
                energies.append(e)
        if times:
            out.append((label, color,
                        np.concatenate(times), np.concatenate(energies)))
    return out


def draw(series, detector, title, outpath):
    """One 3-panel figure; `series` is [(label, color, times, energies), ...]."""
    if not series:
        print('  %s: no counts in this selection, skipped' % A.detector_name(detector))
        return None

    fig, axes = plt.subplots(3, 1, figsize=(8, 11))
    tbins = np.linspace(T_MIN, T_MAX, T_NBINS + 1)
    ebins = np.linspace(E_MIN, E_MAX, E_NBINS + 1)

    for label, color, times, energies in series:
        # Python's % is non-negative for a positive modulus, so an early time wraps onto
        # the axis rather than landing outside it.
        tfold = np.mod(times, T_PERIOD)
        axes[0].hist(tfold, bins=tbins, histtype='step', color=color,
                     linewidth=1.5, label=label)
        axes[1].hist(tfold, bins=tbins, weights=energies, histtype='step',
                     color=color, linewidth=1.5, label=label)
        axes[2].hist(energies, bins=ebins, histtype='step', color=color,
                     linewidth=1.5, label=label)

    tw = (T_MAX - T_MIN) / T_NBINS
    ew_kev = 1000.0 * (E_MAX - E_MIN) / E_NBINS
    for ax, (xlabel, ylabel) in zip(axes, [
            ('time [ns]', 'counts / %.0f ns' % tw),
            ('time [ns]', r'total $E_{dep}$ [MeV] / %.0f ns' % tw),
            (r'$E_{dep}$ [MeV]', 'counts / %.0f keV' % ew_kev)]):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # Log y: rates and spectra both span decades, and a linear scale buries the tails
        # that matter here.
        ax.set_yscale('log')
        ax.grid(True, linestyle=':', alpha=0.5)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=8, ncol=2)

    axes[0].set_title(title)

    # Totals annotated on the single-series case, where there is room and no ambiguity.
    if len(series) == 1:
        _, _, times, energies = series[0]
        n_over = int(np.sum((energies < E_MIN) | (energies >= E_MAX)))
        axes[0].text(0.02, 0.92, 'counts = %d' % len(times),
                     transform=axes[0].transAxes, fontsize=9)
        axes[1].text(0.02, 0.92, r'$\Sigma E$ = %.4g MeV' % energies.sum(),
                     transform=axes[1].transAxes, fontsize=9)
        axes[1].text(0.02, 0.84, 'mean = %.4g MeV/count' % energies.mean(),
                     transform=axes[1].transAxes, fontsize=9)
        if n_over:
            axes[2].text(0.02, 0.92, '%d outside axis' % n_over,
                         transform=axes[2].transAxes, fontsize=9, color='red')

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print('  wrote ' + outpath)

    # Printed alongside so the figure's absolute numbers are reproducible from the log, and
    # so an overflow is visible even on a multi-series plot where it is not annotated.
    dname = A.detector_name(detector)
    for label, _color, times, energies in series:
        n_over = int(np.sum((energies < E_MIN) | (energies >= E_MAX)))
        line = '    %s %s: %d counts, %.4g MeV' % (dname, label, len(times),
                                                   energies.sum())
        if n_over:
            line += '  (%d outside the %g-%g MeV spectrum axis)' % (n_over, E_MIN, E_MAX)
        print(line)
    return outpath


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', default=DEFAULT_SCAN,
                    help='Scan to read (default: %s). Defined: %s'
                         % (DEFAULT_SCAN, ', '.join(sorted(SCANS))))
    ap.add_argument('--split', choices=['none', 'source', 'pdg'], default='none',
                    help='none (default): one series, everything summed. source / pdg: one '
                         'series per source or per attributed species.')
    ap.add_argument('--sources', nargs='*', help='Restrict to these sources.')
    ap.add_argument('--campaigns', nargs='*', help='Restrict to these campaigns.')
    ap.add_argument('--single-campaign', action='store_true',
                    help='Refuse to run when the selection spans several campaigns, which '
                         'would sum different configurations into one figure.')
    ap.add_argument('--use-total-energy', action='store_true',
                    help='Use ionizing + non-ionizing instead of ionizing alone. No effect '
                         'in practice - nonIonE is structurally 0, see stm_analysis.')
    ap.add_argument('--min-edep', type=float, default=None,
                    help='Drop counts whose TOTAL deposit is below this [MeV].')
    ap.add_argument('--min-counts', type=int, default=10,
                    help='With --split pdg, omit species below this many counts (default '
                         '10) - a handful of counts is a line of noise.')
    ap.add_argument('--output-root', default=None, help='Where the ROOT files live.')
    ap.add_argument('--tree-dir', default=A.TREE_DIR,
                    help='TFileService directory holding the trees, i.e. the analyzer '
                         'label in the fcl (default: %s).' % A.TREE_DIR)
    ap.add_argument('--outdir', default='plots', help='Where to write figures.')
    ap.add_argument('--format', default='png', help='Figure format (png, pdf, ...).')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    scan = get_scan(args.scan)
    frames = A.load_scan(scan, sources=args.sources, campaigns=args.campaigns,
                         output_root=args.output_root, min_edep=args.min_edep,
                         directory=args.tree_dir, verbose=args.verbose)

    campaigns = sorted({c for _s, c in frames})
    if args.single_campaign and len(campaigns) > 1:
        raise SystemExit(
            '--single-campaign given but the selection spans %d campaigns (%s). Pass '
            '--campaigns to pick one.' % (len(campaigns), ', '.join(campaigns)))
    if len(campaigns) > 1:
        print('NOTE: summing %d campaigns (%s). These are different configurations, so read '
              'shapes rather than absolute numbers.'
              % (len(campaigns), ', '.join(campaigns)), file=sys.stderr)

    os.makedirs(args.outdir, exist_ok=True)
    tag = '_'.join(campaigns) if len(campaigns) <= 2 else '%dcampaigns' % len(campaigns)
    suffix = '' if args.split == 'none' else '_by%s' % args.split
    print('scan %r, split = %s, campaigns = %s'
          % (scan.name, args.split, ', '.join(campaigns)))

    for detector, dname in A.DETECTORS:
        series = gather(frames, detector, args.split, args.min_counts,
                        args.use_total_energy)
        title = '%s  %s' % (dname, tag)
        if args.split != 'none':
            title += '  by %s' % args.split
        out = os.path.join(args.outdir, '%s_%s_%s_summary%s.%s'
                           % (scan.name, tag, dname, suffix, args.format))
        draw(series, detector, title, out)


if __name__ == '__main__':
    main()
