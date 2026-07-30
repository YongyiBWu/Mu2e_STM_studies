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
particle.

Examples:
  # read the ROOT files, cache to pkl, write 20 displays
  python particletracer_displays.py

  # reuse the cached pkl instead of re-reading ROOT, draw 50
  python particletracer_displays.py --from-pkl --nshow 50

  # only Ele, and only neutrons at the VD
  python particletracer_displays.py --from-pkl --tags Ele --pdgid 2112

  # every matched particle, and dump the chain tables as text too
  python particletracer_displays.py --from-pkl --nshow -1 --print-chains
"""
from __future__ import print_function
import sys, os
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no interactive display; must precede the pyplot import
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import ROOT
from ROOT import gROOT, gStyle, gDirectory, gPad

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
        "Neutrals116": ["/exp/mu2e/data/users/yongyiwu/MDC2025ab/datasets/MDC2025/rootfiles/NeutralsToVD116Neutron.root"],
    }
}

chaincols = ['simId', 'pdgId', 'matched', 'hasTrajectory', 'nPoints',
             'parentSimId', 'parentPdgId', 'creationCode', 'isPrimary',
             'startx', 'starty', 'startz', 'starttime', 'startkE',
             'endx', 'endy', 'endz', 'endtime', 'endkE']


def chain_title(seed_, nchain):
    try:
        seed_name = pdgid_dict[seed_['pdgId']]
    except KeyError:
        seed_name = str(int(seed_['pdgId']))
    return ("Back trace of " + str(seed_['tag']) + " %03i " % seed_['fileno'] + seed_name +
            "  run %i subRun %i event %i simId %i" % (seed_['run'], seed_['subRun'],
                                                      seed_['event'], seed_['simId']) +
            "  (%i in chain)" % nchain)


def load(geometry, tags, pklname, verbose=True):
    """Read the ROOT files into one DataFrame and cache it to pklname.

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
    with open(pklname, 'wb') as f:
        pickle.dump(df_traj, f)
    if verbose:
        print("cached " + pklname)
    return df_traj


def load_pkl(pklname):
    with open(pklname, 'rb') as f:
        return pickle.load(f)


def summarize(df_traj):
    """Print the counts the notebook's summary cell displayed."""
    print("entries:              ", len(df_traj))
    print("with MCTrajectory:    ", int(df_traj['hasTrajectory'].sum()), "(solid)")
    print("start/end only:       ", int((~df_traj['hasTrajectory']).sum()), "(dashed)")
    print("matched (VD) parts:   ", int(df_traj['matched'].sum()))
    print("ancestors:            ", int((~df_traj['matched']).sum()))
    print("pdgIds present:       ", np.sort(df_traj['pdgId'].unique()))
    print("art events:           ", len(portROOT2pd_particletracer.getEventList(df_traj)))


def select(df_traj, tags=None, pdgids=None):
    """One row per particle that reached the VD, optionally cut by tag / pdgId."""
    sel = portROOT2pd_particletracer.getMatched(df_traj)
    if tags:
        sel = sel[sel['tag'].isin(tags)].reset_index(drop=True)
    if pdgids:
        sel = sel[sel['pdgId'].isin(pdgids)].reset_index(drop=True)
    return sel


def draw(df_traj, sel, pdfname, nshow=20, stride=1, print_chains=False):
    """One PDF page per matched particle, walking its genealogy back to the primary."""
    if not len(sel):
        print("no matched particles selected; nothing to draw.")
        return 0

    last = len(sel) if nshow < 0 else min(nshow*stride, len(sel))
    npage = 0
    with PdfPages(pdfname) as pdf:
        for ii in range(0, last, stride):
            seed_ = sel.iloc[ii]
            # seed first, then parent, grandparent, ... , primary last
            dfc_ = portROOT2pd_particletracer.getGenealogy(df_traj, seed_)
            title = chain_title(seed_, len(dfc_))
            if print_chains:
                print('------------------------------------------------------------------------------')
                print(title)
                print(dfc_[chaincols].to_string())
            fig, ax_top, ax_side = plot_utils.draw_particle_tracer_event(dfc_, title)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
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
                   help="tags to read; default is every tag of the geometry. "
                        "With --from-pkl this filters the cached frame instead.")
    p.add_argument("--pkl", default=None,
                   help="cache file (default: <geometry>_particletracer_neutron.pkl)")
    p.add_argument("--from-pkl", action="store_true",
                   help="load the cached pkl instead of re-reading the ROOT files")
    p.add_argument("--pdf", default=None,
                   help="output PDF (default: <geometry>_particletracer_displays.pdf)")
    p.add_argument("--nshow", type=int, default=20,
                   help="matched particles to draw, -1 for all (default: %(default)s)")
    p.add_argument("--stride", type=int, default=1,
                   help="step through the matched particle list (default: %(default)s)")
    p.add_argument("--pdgid", type=int, nargs="+", default=None,
                   help="only draw matched particles with these pdgIds, e.g. 2112")
    p.add_argument("--print-chains", action="store_true",
                   help="also print each genealogy chain as a table")
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

    pklname = args.pkl or (geometry + "_particletracer_neutron.pkl")
    pdfname = args.pdf or (geometry + "_particletracer_displays.pdf")
    # when reading ROOT, --tags picks which files to open; when loading the cache it is
    # applied as a filter further down instead
    readtags = args.tags or list(particle_tracer_root_files[geometry])

    if args.from_pkl:
        if not os.path.exists(pklname):
            print("ERROR: %s not found; run without --from-pkl first." % pklname)
            return 2
        print("loading " + pklname)
        df_traj = load_pkl(pklname)
    else:
        unknown = [t for t in readtags if t not in particle_tracer_root_files[geometry]]
        if unknown:
            print("ERROR: unknown tag(s) %s for geometry '%s'; known: %s"
                  % (unknown, geometry, sorted(particle_tracer_root_files[geometry])))
            return 2
        df_traj = load(geometry, readtags, pklname)

    if not len(df_traj):
        print("no entries read; nothing to do.")
        return 1

    summarize(df_traj)

    sel = select(df_traj, tags=args.tags, pdgids=args.pdgid)
    print("matched particles:    ", len(sel), "(selected)")

    if args.no_draw:
        return 0

    draw(df_traj, sel, pdfname, nshow=args.nshow, stride=args.stride,
         print_chains=args.print_chains)
    return 0


if __name__ == "__main__":
    sys.exit(main())
