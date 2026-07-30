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
#
# Reading goes through RDataFrame.AsNumpy(), which pulls every branch in bulk in C++;
# the per-entry GetEntry() loop this replaced was orders of magnitude slower. Same
# idiom as nucap_MuBeamStopVD.ipynb.
#
# NOTE on ROOT.EnableImplicitMT(): with implicit MT on, AsNumpy() does not preserve
# tree entry order. Nothing here depends on the order being the tree's (rows are
# addressed by run/subRun/event/simId, and getGenealogy() reorders the chain itself),
# but the 'entry' column is then a row counter rather than the TTree entry number.
# The reads are IO bound anyway, so MT buys little; leave it off if you want 'entry'
# to mean the TTree entry number.

# vector<double>/vector<int> branches: one array per entry
VECTOR_COLUMNS = {
    'x' : float, 'y' : float, 'z' : float,      # mm
    't' : float,                                # ns
    'kE': float, 'E' : float,                   # MeV
    'ancestorSimIds': int, 'ancestorPdgIds': int,
}

# bool branches (/O in the module). AsNumpy can hand these back as an integer or object
# dtype rather than numpy bool; they are cast on read so that ~col negates instead of
# doing a bitwise NOT (~1 == -2, which silently corrupts any count built from it).
BOOL_COLUMNS = [
    'matched',        # True: seeded StepPointMC particle
    'hasTrajectory',  # False: start/end positions only
    'isPrimary',
]

SCALAR_COLUMNS = [
    'run', 'subRun', 'event', 'simId', 'pdgId',
    'nPoints',
    'startPx', 'startPy', 'startPz', 'startP',  # MeV/c
    'endPx', 'endPy', 'endPz', 'endP',
    'parentSimId',    # -1 if primary/unavailable
    'parentPdgId', 'creationCode',
] + BOOL_COLUMNS

COLUMNS = SCALAR_COLUMNS + list(VECTOR_COLUMNS)


def _materialize_vectors(df_):
    """Turn the RVec proxies AsNumpy hands back into plain owned numpy arrays.

    AsNumpy returns a vector branch as an object array of ROOT RVec objects. Those are
    thin views over memory RDataFrame owns, so they are converted (and copied) here --
    otherwise the DataFrame outlives the buffers and cannot be pickled.
    """
    for col, dtype in VECTOR_COLUMNS.items():
        if col in df_.columns:
            df_[col] = df_[col].apply(lambda v, d=dtype: np.array(v, dtype=d, copy=True))
    return df_


def _materialize_bools(df_):
    """Cast the /O branches to real numpy bool.

    AsNumpy may return them as int8/uint8 (or object), where ~col is a bitwise NOT
    rather than a logical negation -- ~1 == -2, so a count like (~col).sum() comes out
    negative and wrong instead of counting the False rows.
    """
    for col in BOOL_COLUMNS:
        if col in df_.columns:
            df_[col] = df_[col].astype(bool)
    return df_


def PortToDF(geometry, tag, fileList, verbose = False, treedir = "particleTracer",
             treename = "ttree", weighted = True, columns = None):
    """Return one DataFrame with a row per TTree entry (one SimParticle trajectory).

    Point branches are kept as numpy arrays in the columns x, y, z, t, kE, E so a
    whole trajectory can be plotted from a single row.

    columns: branches to read; defaults to COLUMNS. Narrowing it speeds the read up
    further -- but the drop the notebook cares about is the point vectors, which the
    displays need.
    """
    if columns is None:
        columns = COLUMNS

    weight = 1.
    if weighted:
        try:
            myPOT = sum(filepath.input_event_count[geometry][tag])/filepath.last_stage_output[tag]*filepath.last_stage_POT[tag]
            weight = filepath.goal_POT/myPOT
        except KeyError:
            print("WARNING: no POT bookkeeping for geometry '%s' tag '%s'; weight set to 1."%(geometry, tag))

    treepath = treedir + "/" + treename
    frames = []
    for ii, filename in enumerate(fileList):
        if verbose:
            print("Opening "+filename)

        rdf_ = ROOT.RDataFrame(treepath, filename)

        # only ask for branches the file actually has, so a tree written by an older
        # version of the module still reads
        available = set(str(c) for c in rdf_.GetColumnNames())
        missing = [c for c in columns if c not in available]
        if missing:
            print("WARNING: %s has no branch(es) %s; skipping those."%(filename, missing))
        usecols = [c for c in columns if c in available]

        df_ = pd.DataFrame(rdf_.AsNumpy(columns=usecols))
        if verbose:
            print(len(df_), " trajectory entries are found")

        df_ = _materialize_vectors(df_)
        df_ = _materialize_bools(df_)
        df_.insert(0, 'tag', tag)      # Ele, Mu, 1809, Neutrals
        df_.insert(1, 'fileno', ii)
        df_.insert(2, 'entry', np.arange(len(df_)))
        df_['weight'] = weight

        frames.append(df_)

    if not frames:
        return pd.DataFrame()

    df_traj = pd.concat(frames, ignore_index=True)
    if verbose:
        print(len(df_traj), " trajectory entries are collected")

    # Convenience scalars derived from the point vectors (first/last point of the entry).
    for col, src, idx in [('startx','x',0), ('starty','y',0), ('startz','z',0),
                          ('starttime','t',0), ('startkE','kE',0),
                          ('endx','x',-1), ('endy','y',-1), ('endz','z',-1),
                          ('endtime','t',-1), ('endkE','kE',-1)]:
        if src in df_traj.columns:
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
