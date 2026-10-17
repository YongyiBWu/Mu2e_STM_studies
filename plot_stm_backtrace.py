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
script outlines the STM hardware: the front shielding, the SSC, and the two crystals. The
HPGe is rotated 45 degrees about Y, so it is drawn angled in the top view and square in the
side view, matching constructSTM.cc.

Those outlines are HAND-ENTERED estimates, not read from the geometry - see STM_ELEMENTS
below for what is exact and what is not.

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
from matplotlib.patches import Polygon, Rectangle

import pdgid
import plot_utils
import stm_analysis as A
from run_stm_scans import SCANS, DEFAULT_SCAN, get_scan, SOURCES

# ---------------------------------------------------------------------------
# Hardware outlines drawn on the views.
#
# Anchored on STMShieldingRef, the FRONT FACE OF THE STEEL SUPPORT CRADLE, which STM_v11.txt
# fixes absolutely: VD101 sits 15 mm upstream of it at z = 40310.29, so the reference is
# 40325.29. Everything else is placed from there with the thicknesses the geometry file
# gives, so the layout follows the geometry rather than being eyeballed.
#
# NOT read from the geometry at run time, though - constructSTM.cc composes these at
# construction, and constants.VD101_Z in this repo is a STALE position from before the
# detector moved, so it is deliberately not used here. Re-derive STM_SHIELDING_REF if the
# cradle moves again.
# ---------------------------------------------------------------------------
X_BEAM = -3904.0            # beam centre in x
OFFSET_SPOT = 40.64         # stm.STM_SSC.offset_Spot: the two apertures about X_BEAM
OFFSET_HPGE = -10.0         # stm.HPGe.offset_HPGe
OFFSET_LABR = 0.0           # stm.LaBr.offset_LaBr

# The z anchor: east wall inner face - stm.STM_SSC.FrontToWall - BackShielding.BPThick
# - BackShielding.ShieldingPipeGap = 41351.2 - 924.3 - 25.4 - 76.2. VD101 sits 15 mm
# upstream of it, at 40310.3.
STM_SHIELDING_REF = 40325.3                           # the cradle's front face
SSC_SUPPORT_DEPTH = 110.300                           # stm.SSCSupport.depth, the cradle in z
WALL_TO_CRADLE_GAP = 1.200                            # stm.FrontShieldingRight.wallToCradleGap
FRONT_SHIELD_DZ = 203.200                             # 2+1+2+1+2 in of Pb/BP layers
FSR_LINING_DZ = 12.700                                # the copper lining behind the last lead
SIDE_WALL_DZ = 609.600                                # 24 in

# The front shielding starts behind the cradle, 111.5 mm past its front face (STMMaker
# fsrFrontZ = SSCSupport depth + wallToCradleGap); the side walls start where its last lead
# layer ends, 314.7 mm past it, and run to FrontToWall at 924.3 mm.
SHIELD_HOUSE_Z0 = STM_SHIELDING_REF + SSC_SUPPORT_DEPTH + WALL_TO_CRADLE_GAP
# Front_T in constructSTM.cc: the back face of the front shielding right's copper lining,
# 327.4 mm past the cradle front, which every detector z is measured from.
FRONT_T = SHIELD_HOUSE_Z0 + FRONT_SHIELD_DZ + FSR_LINING_DZ

# SSC block. v11 is symmetric and roughly half v09 in both width and height, and drops
# W_middle / delta_WlL / delta_WlR entirely.
SSC_W_WIDTH = 162.560                                 # stm.STM_SSC.W_width,  6.4 in
SSC_W_HEIGHT = 81.280                                 # stm.STM_SSC.W_height, 3.2 in
SSC_WDEPTH_F, SSC_ZGAP, SSC_ZGAP_BACK = 50.800, 8.70, 4.35
SSC_DZ = SSC_WDEPTH_F + 50.800                        # Wdepth_f + Wdepth_b
# The tungsten's upstream face is ZGap - ZGapBack past the cradle front; both slabs follow.
SSC_Z_FRONT = STM_SHIELDING_REF + SSC_ZGAP - SSC_ZGAP_BACK            # 40329.65

# Detector standoffs from FRONT_T.
Z_LABR = 310.0              # stm.LaBr.Z_LaBr
Z_HPGE = 50.431             # stm.HPGe.Z_HPGe

# constructSTM.cc rotates the HPGe by rotateY(45 deg). The sign here is fixed by the
# placement, not by the G4 convention: the crystal centre sits at -proj in x and +proj in z
# from its aperture, so the long axis tilts toward -x as z increases. In the (z, x) top view
# a positive angle turns +z toward +x, so that is -45.
HPGE_ANGLE_DEG = -45.0
HPGE_CRYSTAL_R = 36.05
HPGE_CRYSTAL_L = 78.5
LABR_CRYSTAL_R = 38.1
LABR_CRYSTAL_L = 76.2

# The HPGe crystal CENTRE is not at its aperture: constructSTM.cc places it a distance
#   d = WindowD + AirD + Capsule_Windowthick + CrystalL/2
# along the ROTATED axis from the face point, which at 45 degrees projects equally onto -x
# and +z. The LaBr is unrotated, so its centre is simply its face point plus CrystalL/2 in z.
_HPGE_FACE_TO_CENTRE = 1.5 + 4.0 + 0.03 + HPGE_CRYSTAL_L / 2.0        # 44.78 mm
_HPGE_PROJ = _HPGE_FACE_TO_CENTRE * np.sqrt(2.0) / 2.0                # 31.66 mm

# Each entry: (label, z_c, x_c, y_c, dz, dx, dy, angle_deg, colour). angle_deg is a rotation
# about Y, so it tilts the TOP (z, x) view only - which is why the HPGe is angled in one
# panel and square in the other.
# The zoomed view: the STM shield house, in Mu2e coordinates [mm], from STM_v11. In z it
# runs from the SSC front shield's poly block (about 540 mm upstream of the cradle front at
# 40325.3; its upstream face is at 39787.7) to the hall wall's inner face at 41351.2. In x
# it covers the house from the left wall's outer face (806.45 mm on the -x side of the beam
# at -3904) to the right wall's outer edge prism (311.15 mm on the +x side). In y it covers
# the baseplate (top 236 mm, bottom 261.4 mm below the bore) up to the right wall's edge
# prism top (254 mm above the bore). These bounds include a margin and are not exact
# outlines; change them here if the geometry changes.
SHIELD_HOUSE = {
    'z': (39750.0, 41400.0),
    'x': (-4800.0, -3300.0),
    'y': (-450.0, 550.0),
}

# Listed outermost first so the house draws behind what it contains. The house and its front
# shielding take their transverse extent from SHIELD_HOUSE above: v11 builds
# the house from individually placed bricks with no single width to read, so those bounds are
# the best available description of it and there is no second set to drift from them.
_HOUSE_X_C = sum(SHIELD_HOUSE['x']) / 2.0
_HOUSE_X_W = SHIELD_HOUSE['x'][1] - SHIELD_HOUSE['x'][0]
_HOUSE_Y_C = sum(SHIELD_HOUSE['y']) / 2.0
_HOUSE_Y_W = SHIELD_HOUSE['y'][1] - SHIELD_HOUSE['y'][0]

STM_ELEMENTS = [
    ('Shield house',
     SHIELD_HOUSE_Z0 + (FRONT_SHIELD_DZ + SIDE_WALL_DZ) / 2.0,
     _HOUSE_X_C, _HOUSE_Y_C,
     FRONT_SHIELD_DZ + SIDE_WALL_DZ, _HOUSE_X_W, _HOUSE_Y_W,
     0.0, 'dimgrey'),
    ('Front shielding',
     SHIELD_HOUSE_Z0 + FRONT_SHIELD_DZ / 2.0,
     _HOUSE_X_C, _HOUSE_Y_C,
     FRONT_SHIELD_DZ, _HOUSE_X_W, _HOUSE_Y_W,
     0.0, 'tab:brown'),
    ('SSC',
     SSC_Z_FRONT + SSC_DZ / 2.0,
     X_BEAM, 0.0,
     SSC_DZ, SSC_W_WIDTH, SSC_W_HEIGHT,
     0.0, 'tab:olive'),
    ('HPGe',
     FRONT_T + Z_HPGE + _HPGE_PROJ,
     X_BEAM - OFFSET_SPOT + OFFSET_HPGE - _HPGE_PROJ,
     0.0,
     HPGE_CRYSTAL_L, 2 * HPGE_CRYSTAL_R, 2 * HPGE_CRYSTAL_R,
     HPGE_ANGLE_DEG, 'tab:red'),
    ('LaBr',
     FRONT_T + Z_LABR + 1.5 + 4.0 + LABR_CRYSTAL_L / 2.0,   # WindowD + AirD + L/2
     X_BEAM + OFFSET_SPOT + OFFSET_LABR,
     0.0,
     LABR_CRYSTAL_L, 2 * LABR_CRYSTAL_R, 2 * LABR_CRYSTAL_R,
     0.0, 'tab:blue'),
]



def fetch_counts(frames, source=None, campaign=None, detector=None, pdg=None,
                 max_rows=None):
    """Count rows as a dict of numpy arrays, for the join into the genealogy.

    Keyed on (run, subRun, event, detector, groupId, cluster) - the tuple that identifies a
    count and is the join key into ttree_genealogy. Note the key does NOT include the
    campaign, so rows from two campaigns can collide on it; selecting one campaign is what
    keeps the join unambiguous.
    """
    cols = ['run', 'subRun', 'event', 'detector', 'groupId', 'cluster',
            'pdgId', 'E', 'time', 'x', 'y', 'z', 'startx', 'starty', 'startz', 'startE']
    out = {c: [] for c in cols}
    for (src, camp), frame in frames.items():
        if source is not None and src != source:
            continue
        if campaign is not None and camp != campaign:
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


def fetch_genealogy(gframes, source=None, campaign=None):
    """Genealogy rows as a dict of numpy arrays, for one (source, campaign)."""
    cols = ['run', 'subRun', 'event', 'detector', 'groupId', 'cluster', 'depth',
            'trackId', 'parentId',
            'pdgId', 'startx', 'starty', 'startz', 'endx', 'endy', 'endz']
    out = {c: [] for c in cols}
    for (src, camp), frame in gframes.items():
        if source is not None and src != source:
            continue
        if campaign is not None and camp != campaign:
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
    """{join key: [row indices]}, ordered shallowest first.

    The rows of one key form a TREE, not a chain: a cluster's depositors share ancestors,
    and the module records each particle once. The order here is only for readability;
    draw_tree() uses trackId/parentId to find the edges.
    """
    index = {}
    for i in range(len(g['run'])):
        index.setdefault(key_of(g, i), []).append(i)
    for key in index:
        index[key].sort(key=lambda i: g['depth'][i])
    return index


def draw_stm_region(ax_top, ax_side):
    """Outline the STM hardware on both views.

    A rotation about Y tilts the top (z, x) view only, so an element with a non-zero angle
    is drawn as a rotated rectangle there and as a plain box in the side (z, y) view.
    """
    for (label, zc, xc, yc, dz, dx, dy, angle, color) in STM_ELEMENTS:
        # Corners computed here rather than via Rectangle(angle=..., rotation_point=...),
        # which needs matplotlib >= 3.6; this works on any version and keeps the rotation
        # visible.
        half = np.array([[-dz / 2.0, -dx / 2.0], [dz / 2.0, -dx / 2.0],
                         [dz / 2.0, dx / 2.0], [-dz / 2.0, dx / 2.0]])
        a = np.radians(angle)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        corners = half.dot(rot.T) + np.array([zc, xc])
        ax_top.add_patch(Polygon(corners, closed=True, edgecolor=color,
                                 facecolor='none', linewidth=1.2, zorder=3))
        # The side (z, y) view is unaffected by a rotation about Y.
        ax_side.add_patch(Rectangle((zc - dz / 2.0, yc - dy / 2.0), dz, dy,
                                    edgecolor=color, facecolor='none',
                                    linewidth=1.2, zorder=3))
        ax_top.text(zc, corners[:, 1].max() + 15.0, label, fontsize=6, color=color,
                    ha='center', va='bottom', zorder=3)


def new_figure(title, zoom=True):
    """The repo's two-panel layout: xz on top, yz below, sharing the z axis.

    With zoom, both panels show only the shield house (SHIELD_HOUSE). Equal aspect on both
    panels would let matplotlib shrink each box to a different width, so the two z axes
    would no longer line up. Instead the aspect is left free and the panel heights and the
    figure size are set from the x, y and z spans, which keeps the scale close to 1:1 while
    the two boxes have exactly the same width.
    """
    if not zoom:
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(5, 1, figure=fig, wspace=0.05, hspace=0.00)
        ax_top = fig.add_subplot(gs[0:3])
        ax_side = fig.add_subplot(gs[3:], sharex=ax_top)
        plot_utils.draw_Mu2e_topview(ax_top)
        plot_utils.draw_Mu2e_sideview(ax_side)
        draw_stm_region(ax_top, ax_side)
        fig.suptitle(title, y=0.93)
        return fig, ax_top, ax_side

    (zlo, zhi), (xlo, xhi), (ylo, yhi) = (SHIELD_HOUSE['z'], SHIELD_HOUSE['x'],
                                          SHIELD_HOUSE['y'])
    axes_w = 9.0                                    # inches of z axis
    scale = axes_w / (zhi - zlo)                    # inches per mm, same on every axis
    h_top, h_side = (xhi - xlo) * scale, (yhi - ylo) * scale
    fig = plt.figure(figsize=(axes_w + 1.5, h_top + h_side + 1.2))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[h_top, h_side], hspace=0.0)
    ax_top = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1], sharex=ax_top)
    plot_utils.draw_Mu2e_topview(ax_top)
    plot_utils.draw_Mu2e_sideview(ax_side)
    draw_stm_region(ax_top, ax_side)
    for ax, (lo, hi) in ((ax_top, (xlo, xhi)), (ax_side, (ylo, yhi))):
        ax.set_aspect('auto')
        ax.set_xlim(zlo, zhi)
        ax.set_ylim(lo, hi)
    fig.suptitle(title)
    return fig, ax_top, ax_side


def draw_tracks(counts, gindex, g, ax_top, ax_side, max_tracks):
    """Draw each count's genealogy TREE. Returns the species seen and how many were drawn.

    One segment per particle, from where it was born to where it ended, coloured by its
    species. Because the module records each particle of a cluster once, drawing every row
    gives the tree with no edge repeated - no reconstruction from parentId is needed to
    avoid overlap, though parentId is available for anything that needs the topology.
    """
    seen = {}
    n_drawn = 0
    for i in range(len(counts['run'])):
        if n_drawn >= max_tracks:
            break
        key = key_of(counts, i)
        rows = gindex.get(key)
        if not rows:
            continue

        for r in rows:
            pdg = int(g['pdgId'][r])
            color = A.pdg_color(pdg)
            sx, sy, sz = g['startx'][r], g['starty'][r], g['startz'][r]
            ex, ey, ez = g['endx'][r], g['endy'][r], g['endz'][r]
            # start -> end of this particle: its own path, so branches meet at the vertex
            # where the daughter was born rather than being chained end to end.
            ax_top.add_line(Line2D([sz, ez], [sx, ex],
                                   linewidth=1.2, linestyle=':', color=color, alpha=0.8))
            ax_side.add_line(Line2D([sz, ez], [sy, ey],
                                    linewidth=1.2, linestyle=':', color=color, alpha=0.8))
            ax_top.scatter(sz, sx, c=color, s=4)
            ax_side.scatter(sz, sy, c=color, s=4)
            seen.setdefault(pdg, 0)
            seen[pdg] += 1
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
        # The root is the row with no parent, not the deepest: the rows form a tree, so the
        # deepest is only the end of the longest branch.
        root_rows = [r for r in (rows or []) if int(g['parentId'][r]) < 0]
        if root_rows:
            r = root_rows[0]
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
                    help='Keep the full Mu2e frame instead of zooming to the STM shield '
                         'house (SHIELD_HOUSE).')
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
    dtag = args.detector or 'both'
    sources = args.sources or [s for s, _ in SOURCES]

    print('scan %r, mode = %s, campaigns = %s' % (scan.name, args.mode, ', '.join(campaigns)))

    # One figure per (campaign, source, ...): a campaign is one geometry, and a track's
    # path through the shielding is a property of it. The join key also omits the campaign,
    # so mixing them could pair a count with another campaign's chain.
    for campaign in campaigns:
        for source in sources:
            g = fetch_genealogy(gframes, source=source, campaign=campaign)
            if not len(g['run']):
                print('  %s/%s: no genealogy rows, skipped' % (campaign, source))
                continue
            gindex = index_genealogy(g)

            if args.mode == 'origins':
                counts = fetch_counts(frames, source=source, campaign=campaign,
                                      detector=detector)
                if not len(counts['run']):
                    print('  %s/%s: no counts, skipped' % (campaign, source))
                    continue
                title = '%s  %s  %s  count origins' % (source, campaign, dtag)
                fig, ax_top, ax_side = new_figure(title, zoom=not args.no_zoom)
                seen = draw_origins(counts, gindex, g, ax_top, ax_side)
                legend_for(seen, ax_top)
                out = os.path.join(args.outdir, '%s_%s_%s_%s_origins.%s'
                                   % (scan.name, campaign, source, dtag, args.format))
                fig.savefig(out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print('  wrote %s  (%d counts)' % (out, len(counts['run'])))
                continue

            # tracks mode: one figure per species of the attributed particle, chosen from this
            # campaign so a species absent here gets no empty figure.
            scoped = {k: v for k, v in frames.items() if k[1] == campaign}
            species = A.species_present(scoped, detector=detector,
                                        min_counts=args.min_counts)
            for pdg in species:
                counts = fetch_counts(frames, source=source, campaign=campaign,
                                      detector=detector, pdg=pdg,
                                      max_rows=args.max_tracks * 4)
                if not len(counts['run']):
                    continue
                title = ('%s  %s  %s  %s back-traces'
                         % (source, campaign, dtag, A.pdg_label(pdg)))
                fig, ax_top, ax_side = new_figure(title, zoom=not args.no_zoom)
                seen, n_drawn = draw_tracks(counts, gindex, g, ax_top, ax_side,
                                            args.max_tracks)
                if n_drawn == 0:
                    plt.close(fig)
                    continue
                legend_for(seen, ax_top)
                out = os.path.join(args.outdir, '%s_%s_%s_%s_pdg%d_tracks.%s'
                                   % (scan.name, campaign, source, dtag, pdg, args.format))
                fig.savefig(out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                note = ''
                if n_drawn >= args.max_tracks:
                    note = '  (capped at --max-tracks; not the whole sample)'
                print('  wrote %s  (%d chains)%s' % (out, n_drawn, note))


if __name__ == '__main__':
    main()
