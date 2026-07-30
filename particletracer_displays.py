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

  # dump the chain tables as text too
  python particletracer_displays.py --print-chains

  # just the first 20
  python particletracer_displays.py --nshow 20
"""
from __future__ import print_function
import sys
import argparse

import numpy as np
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
        # "Neutrals116": ["/exp/mu2e/data/users/yongyiwu/MDC2025ab/datasets/MDC2025/rootfiles/NeutralsToVD116Neutron.root"],
    }
}

chaincols = ['simId', 'pdgId', 'matched', 'hasTrajectory', 'nPoints',
             'parentSimId', 'parentPdgId', 'creationCode', 'isPrimary',
             'startx', 'starty', 'startz', 'starttime', 'startkE',
             'endx', 'endy', 'endz', 'endtime', 'endkE']


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
    print("entries:              ", len(df_traj))
    print("with MCTrajectory:    ", int(df_traj['hasTrajectory'].sum()), "(solid)")
    print("start/end only:       ", int((~df_traj['hasTrajectory']).sum()), "(dashed)")
    print("matched (VD) parts:   ", int(df_traj['matched'].sum()))
    print("ancestors:            ", int((~df_traj['matched']).sum()))
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


def print_z(seed_, perline=8):
    """Print the distinct z values of the matched particle's own trajectory.

    These are the points drawn for the thick line on its display: the full MCTrajectory
    when hasTrajectory is True, otherwise just the SimParticle start and end z. Repeated
    z values are collapsed (a particle can sit at one z over many points), so this shows
    where it actually went rather than every stored point.
    """
    z = np.asarray(seed_['z'], dtype=float)
    # drop consecutive repeats, keeping the order the trajectory was walked in
    uniq = z[np.insert(z[1:] != z[:-1], 0, True)] if len(z) else z
    print("  z [mm]: %i point%s -> %i distinct (%s)"
          % (len(z), "" if len(z) == 1 else "s", len(uniq),
             "MCTrajectory" if seed_['hasTrajectory'] else "start/end only"))
    for ii in range(0, len(uniq), perline):
        print("    " + "  ".join("%12.3f" % v for v in uniq[ii:ii+perline]))
    if len(z):
        print("    first %.3f  last %.3f  min %.3f  max %.3f"
              % (z[0], z[-1], z.min(), z.max()))


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


def draw(df_traj, sel, pdfname, nshow=-1, stride=1, print_chains=False):
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
            print('------------------------------------------------------------------------------')
            print(title)
            print_z(seed_)
            if print_chains:
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
