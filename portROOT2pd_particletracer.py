from __future__ import print_function
import sys, os
import numpy as np
from array import array

import ROOT
from ROOT import gROOT, gStyle, gDirectory, gPad
import pandas as pd

import filepath
import constants

# Reader for the TTree written by Offline/STMMC/src/ParticleTracer_module.cc.
# One TTree entry = one SimParticle (a seeded StepPointMC particle, matched==True,
# or one of its ancestors, matched==False). The point branches (x/y/z/t/kE/E) are
# std::vector<double>: either the stored MCTrajectory points (hasTrajectory==True)
# or just the SimParticle start/end positions (hasTrajectory==False).
#
# The tree lives in the TFileService directory named after the analyzer module label
# (see Offline/STMMC/fcl/ParticleTracer.fcl -> "particleTracer").

def PortToDF(geometry, tag, fileList, verbose = False, treedir = "particleTracer",
             treename = "ttree", weighted = True):
    """Return one DataFrame with a row per TTree entry (one SimParticle trajectory).

    Point branches are kept as numpy arrays in the columns x, y, z, t, kE, E so a
    whole trajectory can be plotted from a single row.
    """
    df_traj = pd.DataFrame()

    weight = 1.
    if weighted:
        try:
            myPOT = sum(filepath.input_event_count[geometry][tag])/filepath.last_stage_output[tag]*filepath.last_stage_POT[tag]
            weight = filepath.goal_POT/myPOT
        except KeyError:
            print("WARNING: no POT bookkeeping for geometry '%s' tag '%s'; weight set to 1."%(geometry, tag))

    rows = []
    for ii, filename in enumerate(fileList):
        if verbose:
            print("Opening "+filename)

        fFile = ROOT.TFile(filename, "READ")
        fDir  = fFile.GetDirectory(treedir)
        if not fDir:
            print("WARNING: directory '%s' not found in %s; skipping."%(treedir, filename))
            fFile.Close()
            continue
        fTree = fDir.Get(treename)

        nEntry = fTree.GetEntries()
        if verbose:
            print(nEntry, " trajectory entries are found")

        for i in range(nEntry):
            fTree.GetEntry(i)

            rows.append({
                'tag'          : tag,          # Ele, Mu, 1809, Neutrals
                'fileno'       : ii,
                'entry'        : i,
                'run'          : int(fTree.run),
                'subRun'       : int(fTree.subRun),
                'event'        : int(fTree.event),
                'simId'        : int(fTree.simId),
                'pdgId'        : int(fTree.pdgId),
                'matched'      : bool(fTree.matched),        # True: seeded StepPointMC particle
                'hasTrajectory': bool(fTree.hasTrajectory),  # False: start/end positions only
                'nPoints'      : int(fTree.nPoints),
                # copy=True: the vector branches are backed by a buffer ROOT reuses on the
                # next GetEntry, so the row must own its own points
                'x'            : np.array(fTree.x, dtype=float, copy=True),   # mm
                'y'            : np.array(fTree.y, dtype=float, copy=True),   # mm
                'z'            : np.array(fTree.z, dtype=float, copy=True),   # mm
                't'            : np.array(fTree.t, dtype=float, copy=True),   # ns
                'kE'           : np.array(fTree.kE, dtype=float, copy=True),  # MeV
                'E'            : np.array(fTree.E, dtype=float, copy=True),   # MeV
                'startPx'      : fTree.startPx,   # MeV/c
                'startPy'      : fTree.startPy,
                'startPz'      : fTree.startPz,
                'startP'       : fTree.startP,
                'endPx'        : fTree.endPx,
                'endPy'        : fTree.endPy,
                'endPz'        : fTree.endPz,
                'endP'         : fTree.endP,
                'parentSimId'  : int(fTree.parentSimId),  # -1 if primary/unavailable
                'parentPdgId'  : int(fTree.parentPdgId),
                'creationCode' : int(fTree.creationCode),
                'isPrimary'    : bool(fTree.isPrimary),
                'ancestorSimIds': np.array(fTree.ancestorSimIds, dtype=int, copy=True), # parent-first, primary last
                'ancestorPdgIds': np.array(fTree.ancestorPdgIds, dtype=int, copy=True),
                'weight'       : weight,
            })
            if len(rows)%50000==1 and verbose:
                print(len(rows), " trajectory entries are collected")

        fFile.Close()

    df_traj = pd.DataFrame(rows)

    # Convenience scalars derived from the point vectors (first/last point of the entry).
    if len(df_traj):
        for col, src, idx in [('startx','x',0), ('starty','y',0), ('startz','z',0),
                              ('starttime','t',0), ('startkE','kE',0),
                              ('endx','x',-1), ('endy','y',-1), ('endz','z',-1),
                              ('endtime','t',-1), ('endkE','kE',-1)]:
            df_traj[col] = df_traj[src].apply(lambda v, k=idx: v[k] if len(v) else np.nan)

    return df_traj


def getEventList(df_traj):
    """Unique (tag, fileno, run, subRun, event) keys present in the DataFrame."""
    return df_traj[['tag','fileno','run','subRun','event']].drop_duplicates().reset_index(drop=True)


def getEvent(df_traj, tag, fileno, run, subRun, event):
    """All trajectory entries belonging to one art event."""
    return df_traj.query("tag==@tag and fileno==@fileno and run==@run and "
                         "subRun==@subRun and event==@event").reset_index(drop=True)


def getMatched(df_traj):
    """The seeded particles: one row per StepPointMC particle that passed the PDG filter."""
    return df_traj.query("matched == True").reset_index(drop=True)


def getGenealogy(df_traj, seed_):
    """Walk from one matched (VD) particle back through its genealogy to the primary.

    seed_ is a single row of getMatched(df_traj). Returns the entries of the seed's own
    art event that lie on its ancestor chain, ordered seed first then parent, grandparent,
    ... , primary last -- i.e. the order ancestorSimIds is written in. Ancestors whose
    entry is missing are skipped (the module writes an entry for every ancestor it walks,
    so this only bites if the DataFrame was filtered).
    """
    dfe_ = getEvent(df_traj, seed_['tag'], seed_['fileno'],
                    seed_['run'], seed_['subRun'], seed_['event'])
    chain = [seed_['simId']] + list(seed_['ancestorSimIds'])
    present = set(dfe_['simId'])
    return dfe_.set_index('simId').loc[[s for s in chain if s in present]].reset_index()
