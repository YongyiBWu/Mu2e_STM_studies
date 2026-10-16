#!/usr/bin/env python3
"""Requirement 4: genealogy back-trace views for STMDet counts.

Draws xz (top) and yz (side) views of where a count's energy came from, from the VD116
origin downstream to the crystal, following the convention of
plot_utils.draw_neutron_event_backtrace: one dotted segment per genealogy link, coloured by
the species of the ancestor that produced it, with a scatter at each vertex.

Two modes:
  --mode tracks   (default) one figure per (source, species) showing up to --max-tracks
                  individual back-traces, each a chain of segments. The detailed view.
  --mode origins  one figure per source showing only the ORIGIN of every count, coloured by
                  species - the aggregate view, closer to draw_particle_source_all. Scales
                  to the whole sample where per-track chains would be unreadable.

Geometry drawn
--------------
The Mu2e outlines come from plot_utils.draw_Mu2e_topview / draw_Mu2e_sideview, so these
figures sit on the same frame as the rest of the repo's displays. On top of that, this
script marks the STM detector region: the LaBr/HPGe split used by STMDetTree, plus a box
around the crystals. Those bounds are read from the module's own defaults - see
STM_REGION below - and are LOOSE region bounds, not tight crystal outlines; they are drawn
to show where the split falls, not to represent the hardware.

Input
-----
Needs ttree_genealogy, which the module writes only when WriteGenealogy is true. A file
without it is skipped with a warning rather than silently producing an empty figure.

Run as:
    python plot_stm_backtrace.py                             # tracks, default scan
    python plot_stm_backtrace.py --mode origins
    python plot_stm_backtrace.py --sources Neutrals --campaigns stmedep00
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import constants
import pdgid
import plot_utils
import stm_analysis as A
from run_stm_scans import SCANS, DEFAULT_SCAN, get_scan, SOURCES

# The detector-split bounds STMDetTree applies: x > XMIN and z > ZMIN is LaBr, the rest is
# HPGe. Kept in step with the module's LaBrRegionXMin / LaBrRegionZMin defaults - if those
# are overridden in the fcl, override them here too or the drawn split will not be the one
# the data was made with.
STM_REGION = {
    'labr_x_min': -4000.0,
    'labr_z_min': 40900.0,
}


def fetch_counts(frames, source=None, detector=None, pdg=None, max_rows=None):
    """Count rows as a dict of numpy arrays, for the join into the genealogy.

    Keyed on (run, subRun, event, detector, groupId, cluster) - the tuple that identifies a
    count and is the join key into ttree_genealogy.
    """
    cols = ['run', 'subRun', 'event', 'detector', 'groupId', 'cluster',
            'pdgId', 'E', 'time', 'x', 'y', 'z', 'startx', 'starty', 'startz', 'startE']
    out = {c: [] for c in cols}
    for (src, _campaign), frame in frames.items():
        if source is not None and src != source:
            continue
        node = frame
        if detector is not None:
            node = node.Filter(A.detector_cut(detector))
        if pdg is not None:
            node = node.Filter(A.pdg_cut(pdg))
        # Book every column before reading any, so the file is traversed once.
        taken = {c: node.Take['double' if c in ('E', 'time', 'x', 'y', 'z', 'startx',
                                                'starty', 'startz', 'startE')
                              else 'int'](c)
                 for c in cols}
        for c in cols:
            out[c].append(np.asarray(taken[c].GetValue()))
    merged = {c: (np.concatenate(v) if v else np.array([])) for c, v in out.items()}
    if max_rows is not None and len(merged['run']) > max_rows:
        merged = {c: v[:max_rows] for c, v in merged.items()}
    return merged


def fetch_genealogy(gframes, source=None):
    """Genealogy rows as a dict of numpy arrays, for one source."""
    cols = ['run', 'subRun', 'event', 'detector', 'groupId', 'cluster', 'depth',
            'pdgId', 'startx', 'starty', 'startz', 'endx', 'endy', 'endz']
    out = {c: [] for c in cols}
    for (src, _campaign), frame in gframes.items():
        if source is not None and src != source:
            continue
        taken = {c: frame.Take['double' if c in ('startx', 'starty', 'startz',
                                                 'endx', 'endy', 'endz')
                               else 'int'](c)
                 for c in cols}
        for c in cols:
            out[c].append(np.asarray(taken[c].GetValue()))
    return {c: (np.concatenate(v) if v else np.array([])) for c, v in out.items()}


def key_of(d, i):
    """The join tuple for row i of a column dict."""
    return (int(d['run'][i]), int(d['subRun'][i]), int(d['event'][i]),
            int(d['detector'][i]), int(d['groupId'][i]), int(d['cluster'][i]))


def index_genealogy(g):
    """{join key: [row indices]}, each list ordered by depth (parent first)."""
    index = {}
    for i in range(len(g['run'])):
        index.setdefault(key_of(g, i), []).append(i)
    for key in index:
        index[key].sort(key=lambda i: g['depth'][i])
    return index


def draw_stm_region(ax_top, ax_side):
    """Mark the LaBr/HPGe split and the STM detector region on both views.

    Drawn as the bounds the ANALYSIS uses, not as hardware: they are loose region bounds in
    the empty space between the crystals, so a reader can see which side of the split a
    track ended on.
    """
    zmin = STM_REGION['labr_z_min']
    xmin = STM_REGION['labr_x_min']
    # Top view (z, x): the LaBr quadrant is z > zmin AND x > xmin.
    ax_top.axvline(zmin, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_top.axhline(xmin, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_top.text(zmin + 10, xmin + 40, 'LaBr', fontsize=7, color='grey')
    ax_top.text(zmin - 250, xmin - 120, 'HPGe', fontsize=7, color='grey')
    # Side view (z, y): only the z bound is meaningful; x does not appear on this axis, so
    # drawing the x line here would assert a separation the view cannot show.
    ax_side.axvline(zmin, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)


def new_figure(title):
    """The repo's two-panel layout: xz on top, yz below, sharing the z axis."""
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(5, 1, figure=fig, wspace=0.05, hspace=0.00)
    ax_top = fig.add_subplot(gs[0:3])
    ax_side = fig.add_subplot(gs[3:], sharex=ax_top)
    plot_utils.draw_Mu2e_topview(ax_top)
    plot_utils.draw_Mu2e_sideview(ax_side)
    draw_stm_region(ax_top, ax_side)
    fig.suptitle(title, y=0.93)
    return fig, ax_top, ax_side


def zoom_to_stm(ax_top, ax_side, margin=600.0):
    """Restrict z to the STM neighbourhood.

    The Mu2e-wide frame spans tens of metres, in which a back-trace inside the STM is a few
    pixels. The y limits are left as the repo's helpers set them so the views stay
    comparable with the rest of the displays.
    """
    zlo = constants.VD101_Z - margin
    zhi = constants.VD101_Z + 910.0 + margin
    ax_top.set_xlim(zlo, zhi)
    ax_side.set_xlim(zlo, zhi)


def draw_tracks(counts, gindex, g, ax_top, ax_side, max_tracks):
    """One dotted chain per count: deposit -> ... -> origin. Returns the species seen."""
    seen = {}
    n_drawn = 0
    for i in range(len(counts['run'])):
        if n_drawn >= max_tracks:
            break
        key = key_of(counts, i)
        rows = gindex.get(key)
        if not rows:
            continue

        # Start at the deposit position, then walk outward through the ancestors. The
        # segment INTO a vertex is coloured by the species that vertex belongs to, matching
        # draw_neutron_event_backtrace.
        last = (counts['x'][i], counts['y'][i], counts['z'][i])
        for r in rows:
            pdg = int(g['pdgId'][r])
            color = A.pdg_color(pdg)
            sx, sy, sz = g['startx'][r], g['starty'][r], g['startz'][r]
            ax_top.add_line(Line2D([sz, last[2]], [sx, last[0]],
                                   linewidth=1.2, linestyle=':', color=color, alpha=0.8))
            ax_side.add_line(Line2D([sz, last[2]], [sy, last[1]],
                                    linewidth=1.2, linestyle=':', color=color, alpha=0.8))
            ax_top.scatter(sz, sx, c=color, s=4)
            ax_side.scatter(sz, sy, c=color, s=4)
            seen.setdefault(pdg, 0)
            seen[pdg] += 1
            last = (sx, sy, sz)
        n_drawn += 1
    return seen, n_drawn


def draw_origins(counts, gindex, g, ax_top, ax_side):
    """One marker per count at its ROOT origin, coloured by the root species.

    The aggregate view: where did the particles that deposited energy come from. Uses the
    deepest genealogy row, which is the VD116 primary under the default grouping.
    """
    seen = {}
    for i in range(len(counts['run'])):
        key = key_of(counts, i)
        rows = gindex.get(key)
        if rows:
            r = rows[-1]              # deepest = root
            pdg = int(g['pdgId'][r])
            ox, oy, oz = g['startx'][r], g['starty'][r], g['startz'][r]
        else:
            # No chain: the count's own particle is the root, so its start is the origin.
            pdg = int(counts['pdgId'][i])
            ox, oy, oz = counts['startx'][i], counts['starty'][i], counts['startz'][i]
        color = A.pdg_color(pdg)
        ax_top.scatter(oz, ox, c=color, s=5)
        ax_side.scatter(oz, oy, c=color, s=5)
        seen.setdefault(pdg, 0)
        seen[pdg] += 1
    return seen


def legend_for(seen, ax):
    handles = [Line2D([0], [0], linewidth=2, linestyle=':', color=A.pdg_color(pdg),
                      label='%s (%d)' % (A.pdg_label(pdg), n))
               for pdg, n in sorted(seen.items(), key=lambda kv: -kv[1])]
    if handles:
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', default=DEFAULT_SCAN,
                    help='Scan to read (default: %s). Defined: %s'
                         % (DEFAULT_SCAN, ', '.join(sorted(SCANS))))
    ap.add_argument('--mode', choices=['tracks', 'origins'], default='tracks',
                    help='tracks (default): per-species figures of individual chains. '
                         'origins: one figure per source, origin of every count.')
    ap.add_argument('--sources', nargs='*', help='Restrict to these sources.')
    ap.add_argument('--campaigns', nargs='*', help='Restrict to these campaigns.')
    ap.add_argument('--detector', choices=['HPGe', 'LaBr'], default=None,
                    help='Restrict to one detector (default: both, drawn together).')
    ap.add_argument('--max-tracks', type=int, default=200,
                    help='Chains per figure in tracks mode (default 200). More than a few '
                         'hundred is unreadable.')
    ap.add_argument('--min-counts', type=int, default=10,
                    help='Omit species below this many counts (default 10).')
    ap.add_argument('--min-edep', type=float, default=None,
                    help='Drop counts whose TOTAL deposit is below this [MeV].')
    ap.add_argument('--no-zoom', action='store_true',
                    help='Keep the full Mu2e z range instead of zooming to the STM.')
    ap.add_argument('--output-root', default=None, help='Where the ROOT files live.')
    ap.add_argument('--tree-dir', default=A.TREE_DIR,
                    help='TFileService directory holding the trees, i.e. the analyzer '
                         'label in the fcl (default: %s).' % A.TREE_DIR)
    ap.add_argument('--outdir', default='plots', help='Where to write figures.')
    ap.add_argument('--format', default='png', help='Figure format (png, pdf, ...).')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    scan = get_scan(args.scan)
    detector = None
    if args.detector:
        detector = dict((n, c) for c, n in A.DETECTORS)[args.detector]

    frames = A.load_scan(scan, sources=args.sources, campaigns=args.campaigns,
                         output_root=args.output_root, min_edep=args.min_edep,
                         directory=args.tree_dir, verbose=args.verbose)
    gframes = A.load_genealogy(scan, sources=args.sources, campaigns=args.campaigns,
                               output_root=args.output_root,
                               directory=args.tree_dir, verbose=args.verbose)
    if not gframes:
        raise SystemExit(
            'No ttree_genealogy in any input file, so there are no chains to draw. Re-run '
            'run_stmdet_tree.py without --no-genealogy.')

    os.makedirs(args.outdir, exist_ok=True)
    campaigns = sorted({c for _s, c in frames})
    tag = '_'.join(campaigns) if len(campaigns) <= 2 else '%dcampaigns' % len(campaigns)
    dtag = args.detector or 'both'
    sources = args.sources or [s for s, _ in SOURCES]

    print('scan %r, mode = %s, campaigns = %s' % (scan.name, args.mode, ', '.join(campaigns)))

    for source in sources:
        g = fetch_genealogy(gframes, source=source)
        if not len(g['run']):
            print('  %s: no genealogy rows, skipped' % source)
            continue
        gindex = index_genealogy(g)

        if args.mode == 'origins':
            counts = fetch_counts(frames, source=source, detector=detector)
            if not len(counts['run']):
                print('  %s: no counts, skipped' % source)
                continue
            title = '%s  %s  %s  count origins' % (source, tag, dtag)
            fig, ax_top, ax_side = new_figure(title)
            seen = draw_origins(counts, gindex, g, ax_top, ax_side)
            legend_for(seen, ax_top)
            if not args.no_zoom:
                zoom_to_stm(ax_top, ax_side)
            out = os.path.join(args.outdir, '%s_%s_%s_%s_origins.%s'
                               % (scan.name, tag, source, dtag, args.format))
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print('  wrote %s  (%d counts)' % (out, len(counts['run'])))
            continue

        # tracks mode: one figure per species of the attributed particle.
        species = A.species_present(frames, detector=detector,
                                    min_counts=args.min_counts)
        for pdg in species:
            counts = fetch_counts(frames, source=source, detector=detector, pdg=pdg,
                                  max_rows=args.max_tracks * 4)
            if not len(counts['run']):
                continue
            title = ('%s  %s  %s  %s back-traces'
                     % (source, tag, dtag, A.pdg_label(pdg)))
            fig, ax_top, ax_side = new_figure(title)
            seen, n_drawn = draw_tracks(counts, gindex, g, ax_top, ax_side,
                                        args.max_tracks)
            if n_drawn == 0:
                plt.close(fig)
                continue
            legend_for(seen, ax_top)
            if not args.no_zoom:
                zoom_to_stm(ax_top, ax_side)
            out = os.path.join(args.outdir, '%s_%s_%s_%s_pdg%d_tracks.%s'
                               % (scan.name, tag, source, dtag, pdg, args.format))
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            note = ''
            if n_drawn >= args.max_tracks:
                note = '  (capped at --max-tracks; not the whole sample)'
            print('  wrote %s  (%d chains)%s' % (out, n_drawn, note))


if __name__ == '__main__':
    main()
