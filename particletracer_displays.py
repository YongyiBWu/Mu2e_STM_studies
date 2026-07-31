#!/usr/bin/env python
"""ParticleTracer event displays.

Reads the ROOT files written by Offline/STMMC/src/ParticleTracer_module.cc
(fcl: Offline/STMMC/fcl/ParticleTracer.fcl) and draws one top/side view display per
*matched* particle: starting from the particle that reached the VD and walking back
through its genealogy tree to the primary.

One TTree entry = one SimParticle: either a particle that reached the seeded
StepPointMCs and passed the PDG filter (matched == True) or one of its ancestors up to
the primary (matched == False). A single art event can hold several matched particles,
each with its own chain, so the unit of a display is the matched particle rather than
the art event.

  solid line  - the entry has a stored MCTrajectory (hasTrajectory == True); the line
                runs through all of its points.
  dashed line - no MCTrajectory was stored (the particle failed the G4 trajectory cuts),
                so the entry only holds the SimParticle start and end positions and the
                line is a straight segment between them.

Colour is by PDG ID (pdgid.pdgid_color_dict); the matched particle is drawn slightly
thicker than its ancestors.

Script version of MDC2025_neutron.ipynb. Displays go to a PDF, one page per matched
particle. Everything is read from the ROOT files on every run -- there is no pkl cache.

Examples:
  # read every tag, draw every matched particle
  python particletracer_displays.py

  # only Ele, and only neutrons at the VD, draw 50
  python particletracer_displays.py --tags Ele --pdgid 2112 --nshow 50

  # only forward-going matched particles
  python particletracer_displays.py --positive-pz

  # add a facing page per display holding the chain table
  python particletracer_displays.py --print-chains

  # just the first 20
  python particletracer_displays.py --nshow 20
"""
from __future__ import print_function
import sys
import argparse

# ROOT first and in batch mode: IgnoreCommandLineOptions keeps PyROOT from parsing our
# own argv (this script takes --nshow etc.), and SetBatch stops it opening a display.
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)
from ROOT import gROOT, gStyle, gDirectory, gPad

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no interactive display; must precede the pyplot import
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import filepath
import portROOT2pd_particletracer
from pdgid import pdgid_dict
import plot_utils
import constants

# Output of ParticleTracer.fcl.
particle_tracer_root_files = {
    "MDC2025ab": {
        "Ele"        : ["/exp/mu2e/data/users/yongyiwu/MDC2025ab/datasets/MDC2025/rootfiles/EleBeamToVD101Neutron.root"],
        "Mu"         : ["/exp/mu2e/data/users/yongyiwu/MDC2025ab/datasets/MDC2025/rootfiles/MuBeamToVD101Neutron.root"],
        "1809"       : ["/exp/mu2e/data/users/yongyiwu/MDC2025ab/datasets/MDC2025/rootfiles/1809BeamToVD101Neutron.root"],
        "Neutrals101": ["/exp/mu2e/data/users/yongyiwu/MDC2025ab/datasets/MDC2025/rootfiles/NeutralsToVD101Neutron.root"],
        # "Neutrals116": ["/exp/mu2e/data/users/yongyiwu/MDC2025ab/datasets/MDC2025/rootfiles/NeutralsToVD116Neutron.root"],
    }
}

chaincols = ['simId', 'pdgId', 'matched', 'hasTrajectory', 'nPoints',
             'parentSimId', 'parentPdgId', 'creationCode', 'isPrimary',
             'startx', 'starty', 'startz', 'starttime', 'startkE',
             'endx', 'endy', 'endz', 'endtime', 'endkE']

# Short header labels for the PDF chain table. The full names are much wider than the
# values under them (hasTrajectory is 13 chars over a True/False), and to_string() keeps
# every header on one line, so the long ones alone would set the table width.
chainheads = {
    'hasTrajectory': 'hasTraj',
    'parentSimId'  : 'parSimId',
    'parentPdgId'  : 'parPdgId',
    'creationCode' : 'crCode',
    'isPrimary'    : 'prim',
    'starttime'    : 'startt',
    'nPoints'      : 'nPts',
    'matched'      : 'match',
}


def chain_title(seed_, nchain):
    """Figure title: leads with the tag and the matched particle's simId."""
    try:
        seed_name = pdgid_dict[seed_['pdgId']]
    except KeyError:
        seed_name = str(int(seed_['pdgId']))
    return ("[%s] simId %i  %s" % (str(seed_['tag']), seed_['simId'], seed_name) +
            "  back trace, file %03i" % seed_['fileno'] +
            "  run %i subRun %i event %i" % (seed_['run'], seed_['subRun'], seed_['event']) +
            "  (%i in chain)" % nchain)


def load(geometry, tags, verbose=True):
    """Read the ROOT files into one DataFrame.

    Reads via RDataFrame.AsNumpy() (bulk C++ read), not a per-entry GetEntry() loop.
    See the note in portROOT2pd_particletracer about ROOT.EnableImplicitMT() before
    turning it on: it drops the tree entry ordering of the 'entry' column.
    """
    df_traj = pd.DataFrame()
    for tag in tags:
        fileList_ = particle_tracer_root_files[geometry][tag]
        dft_ = portROOT2pd_particletracer.PortToDF(geometry, tag, fileList_, verbose=verbose,
                                                   treedir="particleTracer", treename="ttree",
                                                   weighted=False)
        df_traj = pd.concat([df_traj, dft_], ignore_index=True)
    return df_traj


def summarize(df_traj):
    """Print the counts the notebook's summary cell displayed."""
    # count the True rows and subtract, so the pair always sums to len(df_traj) even if
    # the column arrives as something other than a real bool
    nentries = len(df_traj)
    ntraj = int(df_traj['hasTrajectory'].astype(bool).sum())
    nmatched = int(df_traj['matched'].astype(bool).sum())
    print("entries:              ", nentries)
    print("with MCTrajectory:    ", ntraj, "(solid)")
    print("start/end only:       ", nentries - ntraj, "(dashed)")
    print("matched (VD) parts:   ", nmatched)
    print("ancestors:            ", nentries - nmatched)
    print("pdgIds present:       ", np.sort(df_traj['pdgId'].unique()))
    print("art events:           ", len(portROOT2pd_particletracer.getEventList(df_traj)))

def summarize_matched(sel):
    """Count the matched (VD) particles per pdgId within each tag."""
    print("===== matched particles per tag =====")
    if not len(sel):
        print("  (none)")
        return
    for tag in sel['tag'].unique():
        dft_ = sel[sel['tag'] == tag]
        print("  %-12s %6i" % (tag, len(dft_)))
        counts = dft_['pdgId'].value_counts()
        for pdg in np.sort(counts.index.values):
            try:
                name = pdgid_dict[pdg]
            except KeyError:
                name = ""
            print("      %11i %-10s %6i" % (pdg, name, counts[pdg]))
    print("  %-12s %6i" % ("TOTAL", len(sel)))
    print("=====================================")

def select(df_traj, tags=None, pdgids=None, positive_pz=False, pz_field='endPz'):
    """One row per particle that reached the VD, optionally cut by tag / pdgId / pz.

    positive_pz keeps only forward-going matched particles. Note the tree has no
    StepPointMC momentum at the VD hit -- the module's own keepParticlesWithPositivePz
    fcl option cuts on that, but does not write it. Only the SimParticle momentum at
    creation (startPz) and at stop/exit (endPz) are stored, so the cut here is on
    pz_field: 'endPz' (default, the endpoint at or past the VD) or 'startPz'.
    """
    sel = portROOT2pd_particletracer.getMatched(df_traj)
    if tags:
        sel = sel[sel['tag'].isin(tags)].reset_index(drop=True)
    if pdgids:
        sel = sel[sel['pdgId'].isin(pdgids)].reset_index(drop=True)
    if positive_pz:
        nbefore = len(sel)
        sel = sel[sel[pz_field] > 0].reset_index(drop=True)
        print("pz cut (%s > 0):        %i -> %i matched particles"
              % (pz_field, nbefore, len(sel)))
    return sel


def chain_table_figure(dfc_, title, cols=None, figsize=(15, 6)):
    """Render the genealogy chain table as a monospace text page for the PDF.

    figsize matches plot_utils.draw_particle_tracer_event so the table page is the same
    width as the display it follows (both are saved without bbox_inches='tight', which
    would otherwise crop each page to its own content).

    Floats are shown to 3 decimals and the wide column names are abbreviated via
    chainheads, both to keep the line short enough to stay on the page.
    """
    if cols is None:
        cols = chaincols
    use = [c for c in cols if c in dfc_.columns]

    # short header labels, so the wide names do not set their column's width
    text = dfc_[use].rename(columns=chainheads).to_string(
        float_format=lambda v: "%.3f" % v)
    lines = text.split("\n")
    width = max(len(l) for l in lines) if lines else 1

    fig = plt.figure(figsize=figsize)
    # ~0.60 em per character for monospace; fit the widest line across the page width
    fontsize = min(10.0, figsize[0] * 72.0 * 0.95 / (width * 0.60))
    fig.text(0.01, 0.98, title + "\n\n" + text,
             family="monospace", fontsize=fontsize,
             va="top", ha="left")
    return fig


# Planes at which the kE / position summary is drawn, in the order the pages appear.
# VD101 is where the particles were seeded; the DS cryostat entrance is upstream of it.
kE_planes = [
    ("VD101",          constants.VD101_Z),
    ("DS cryo start",  constants.ds_cryo_start_Z),
    ("IFB end",        constants.IFB_end_Z),
]


def draw_kE_pages(df_traj, pdf, planes=None, tags=None):
    """One kE/position page per plane per tag. Returns the number of pages written.

    Grouped by plane: every tag at the first plane, then every tag at the next, so the
    pages for one z sit together and are directly comparable.
    """
    if planes is None:
        planes = kE_planes
    if tags is None:
        tags = list(df_traj['tag'].unique()) if len(df_traj) else []

    npage = 0
    for zname, z0 in planes:
        print("----- %s (z = %.1f mm) -----" % (zname, z0))
        for tag in tags:
            dft_ = df_traj[df_traj['tag'] == tag]
            if not len(dft_):
                continue
            title = "%s (z = %.1f mm)  --  [%s]" % (zname, z0, tag)
            print(title)
            fig, ax_face, ax_spec, dfx_ = plot_utils.draw_kE_at_z(dft_, z0, title)
            print("    %i crossing(s)" % len(dfx_))
            pdf.savefig(fig)
            plt.close(fig)
            npage += 1
    return npage


def draw(df_traj, sel, pdfname, nshow=-1, stride=1, print_chains=False):
    """The summary pages, then one PDF page per matched particle with its genealogy."""
    if not len(df_traj):
        print("nothing read; no PDF written.")
        return 0
    if not len(sel):
        # still worth writing the per-tag summary pages, which come from df_traj
        print("no matched particles selected; writing the summary pages only.")

    last = len(sel) if nshow < 0 else min(nshow*stride, len(sel))
    npage = 0
    with PdfPages(pdfname) as pdf:
        # summary pages first: one per tag per plane, over every entry (not just the
        # matched selection), so ancestors crossing the plane are counted too
        print('===== kE / position at the summary planes =====')
        npage += draw_kE_pages(df_traj, pdf)

        for ii in range(0, last, stride):
            seed_ = sel.iloc[ii]
            # seed first, then parent, grandparent, ... , primary last
            dfc_ = portROOT2pd_particletracer.getGenealogy(df_traj, seed_)
            title = chain_title(seed_, len(dfc_))
            print('------------------------------------------------------------------------------')
            print(title)
            fig, ax_top, ax_side = plot_utils.draw_particle_tracer_event(dfc_, title)
            # no bbox_inches='tight' here: it would crop each page to its own content,
            # so the table page below would not line up with the display width
            pdf.savefig(fig)
            plt.close(fig)
            npage += 1
            if print_chains:
                # the chain table goes on its own page, facing the display it describes
                figt = chain_table_figure(dfc_, title)
                pdf.savefig(figt)
                plt.close(figt)
                npage += 1
    print("written %s (%i pages)" % (pdfname, npage))
    return npage


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometry", default="MDC2025ab",
                   help="key into particle_tracer_root_files (default: %(default)s)")
    p.add_argument("--tags", nargs="+", default=None,
                   help="tags to read; default is every tag of the geometry")
    p.add_argument("--pdf", default=None,
                   help="output PDF (default: <geometry>_particletracer_displays.pdf)")
    p.add_argument("--nshow", type=int, default=-1,
                   help="matched particles to draw, -1 for all (default: all)")
    p.add_argument("--stride", type=int, default=1,
                   help="step through the matched particle list (default: %(default)s)")
    p.add_argument("--pdgid", type=int, nargs="+", default=None,
                   help="only draw matched particles with these pdgIds, e.g. 2112")
    p.add_argument("--positive-pz", action="store_true",
                   help="only keep forward-going (pz > 0) matched particles")
    p.add_argument("--pz-field", default="endPz", choices=["endPz", "startPz"],
                   help="which stored momentum --positive-pz cuts on; the VD-hit "
                        "momentum is not in the tree (default: %(default)s)")
    p.add_argument("--print-chains", action="store_true",
                   help="add a page to the PDF with each genealogy chain as a table "
                        "(doubles the page count)")
    p.add_argument("--no-draw", action="store_true",
                   help="only read/summarize; skip the PDF")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    geometry = args.geometry
    if geometry not in particle_tracer_root_files:
        print("ERROR: unknown geometry '%s'; known: %s"
              % (geometry, sorted(particle_tracer_root_files)))
        return 2

    pdfname = args.pdf or (geometry + "_particletracer_displays.pdf")
    readtags = args.tags or list(particle_tracer_root_files[geometry])

    unknown = [t for t in readtags if t not in particle_tracer_root_files[geometry]]
    if unknown:
        print("ERROR: unknown tag(s) %s for geometry '%s'; known: %s"
              % (unknown, geometry, sorted(particle_tracer_root_files[geometry])))
        return 2
    df_traj = load(geometry, readtags)

    if not len(df_traj):
        print("no entries read; nothing to do.")
        return 1

    summarize(df_traj)

    # --tags already limited what was read, so only the pdgId/pz cuts are left to apply
    sel = select(df_traj, pdgids=args.pdgid,
                 positive_pz=args.positive_pz, pz_field=args.pz_field)
    print("matched particles:    ", len(sel), "(selected)")
    summarize_matched(sel)

    if args.no_draw:
        return 0

    draw(df_traj, sel, pdfname, nshow=args.nshow, stride=args.stride,
         print_chains=args.print_chains)
    return 0


if __name__ == "__main__":
    sys.exit(main())
