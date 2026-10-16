#!/usr/bin/env python3
"""Shared loading, normalization and styling for the STMDetTree analyses.

Everything that has to agree across the plotting scripts lives here: how a produced ROOT
file maps back to (source, campaign), how counts become rates, and what the detectors are
called on a plot. Species names and colours come from pdgid.py rather than a second copy.

Why one RDataFrame per file
---------------------------
STMDetTree records neither the source nor the campaign - both are properties of which
dataset was fed in, not of any event - so provenance has to come from the file a row was
read out of. Keeping one RDataFrame per (source, campaign) preserves that exactly, and
makes the rate denominator trivially correct: a point summing N files is normalized by
N * POT_PER_FILE, and N is just how many frames went into it.

RDataFrame is lazy, so building the dict costs nothing; the event loops run when results
are materialized. Where several results come from the same frame, book them all first and
read them afterwards (see materialize) so one pass over the file serves all of them.

Normalization
-------------
Each produced file represents a FIXED 1e12 POT per (source, campaign) by construction - a
property of how the campaign was generated, not of how many events a job happened to write:

    1-batch   9.3e12 POT/s  ->  counts / 1e12 * 9.3e12 / 1000 kcps
    2-batch   2.3e13 POT/s  ->  counts / 1e12 * 2.3e13 / 1000 kcps

Energy rates are the same arithmetic with summed MeV in place of counts, giving MeV*kcps.
The intensities are written as POT and converted here rather than as the bare factors 9.3
and 23: the factor only means anything relative to POT_PER_FILE, and stating the two
separately is what makes a wrong assumption visible instead of buried in a multiplier.

IMPORTANT: a run capped with --max-events does NOT represent 1e12 POT, so its rates are
meaningless. Nothing in the file records the cap, so the driver's warning is the only guard.
"""

from __future__ import print_function

import os
import sys

import ROOT

import pdgid

# POT each produced file stands for, by construction of the campaign.
POT_PER_FILE = 1.0e12

# Beam intensities [POT/s], keyed by the label that appears in a legend.
BATCHES = {
    '1-batch': 9.3e12,
    '2-batch': 2.3e13,
}
DEFAULT_BATCHES = ['1-batch', '2-batch']

# Detector codes as written by STMDetTree's "detector" branch.
DETECTORS = [(0, 'HPGe'), (1, 'LaBr')]
DETECTOR_NAME = dict(DETECTORS)

# TFileService puts a module's trees in a directory named after its ANALYZER LABEL, not at
# the file root - so the trees are at "<label>/ttree", exactly as edep_plot.py reads
# "Stage2HPGe/ttree". The default matches the label in Offline/STMMC/fcl/STMDetTree.fcl;
# change it (or pass directory=) if the fcl renames the analyzer.
TREE_DIR = 'STMDetTree'
COUNT_TREE = 'ttree'
GENEALOGY_TREE = 'ttree_genealogy'


def tree_path(name, directory=TREE_DIR):
    """'STMDetTree/ttree' from 'ttree'. An empty directory means the file root."""
    return '%s/%s' % (directory, name) if directory else name

# Matplotlib colours, in the same spirit as pdgid.pdgid_color_dict.
SOURCE_COLOR_DICT = {
    'Ele':      'red',
    'Mu':       'blue',
    '1809':     'green',
    'Neutrals': 'orange',
}
# Matching the reference figure: 1-batch blue, 2-batch orange.
BATCH_COLOR_DICT = {
    '1-batch': 'tab:blue',
    '2-batch': 'tab:orange',
}


def detector_name(code):
    return DETECTOR_NAME.get(code, 'detector %d' % code)


def pdg_label(pdg):
    """Species name for a legend, from pdgid.py, falling back to the bare id."""
    try:
        return pdgid.pdgid_dict[pdg]
    except KeyError:
        return str(int(pdg))


def pdg_color(pdg, fallback='yellow'):
    """Species colour from pdgid.py - same fallback convention as plot_utils.py."""
    try:
        return pdgid.pdgid_color_dict[pdg]
    except KeyError:
        return fallback


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def to_kcps(value, batch, n_files=1, pot_per_file=POT_PER_FILE):
    """counts (or summed MeV) from ONE source file -> kcps (or MeV*kcps) at an intensity.

    To combine sources, normalize each file separately and ADD the rates. Every source file
    represents pot_per_file of the FULL beam - Ele, Mu, 1809 and Neutrals are different
    processes occurring together in the same beam, not repeated samples of one thing - so
    summing their counts and dividing by the file count understates the answer by that
    factor.

    n_files > 1 is therefore only for genuinely repeated running of the SAME source, where
    the statistics do accumulate over a larger POT.
    """
    if batch not in BATCHES:
        raise KeyError('Unknown batch %r. Known: %s' % (batch, ', '.join(BATCHES)))
    total_pot = pot_per_file * n_files
    if total_pot <= 0:
        raise ValueError('Total POT must be positive.')
    return value / total_pot * BATCHES[batch] / 1000.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scan(scan, sources=None, campaigns=None, output_root=None,
              min_edep=None, tree=COUNT_TREE, directory=TREE_DIR, verbose=False):
    """{(source, campaign): RDataFrame} over a scan's produced files.

    Nothing is read here - RDataFrame is lazy, so this only opens the files and applies the
    min_edep filter to the graph. The event loops run when results are materialized.

    min_edep cuts on TOTAL deposit (ionizing + non-ionizing), matching the module's own
    MinEdep, so a cut applied here and one applied at production mean the same thing.

    A file that does not exist is SKIPPED WITH A WARNING rather than treated as empty: an
    absent campaign and a campaign that genuinely produced no counts would otherwise plot
    identically, and only one of them is a result.
    """
    from run_stm_scans import SOURCES

    source_list = [s for s, _ in SOURCES]
    if sources:
        unknown = [s for s in sources if s not in source_list]
        if unknown:
            raise SystemExit('Unknown source(s): %s' % ', '.join(unknown))
        source_list = [s for s in source_list if s in sources]

    point_list = scan.points
    if campaigns:
        known = set(scan.campaigns())
        unknown = [c for c in campaigns if c not in known]
        if unknown:
            raise SystemExit('Campaign(s) not in scan %r: %s'
                             % (scan.name, ', '.join(unknown)))
        point_list = [p for p in scan.points if p.campaign in campaigns]

    frames, missing = {}, []
    for source in source_list:
        for point in point_list:
            path = scan.output_path(source, point.campaign, output_root)
            if not os.path.isfile(path):
                missing.append(path)
                continue
            if verbose:
                print('Opening ' + path)
            df = ROOT.RDataFrame(tree_path(tree, directory), path)
            if min_edep is not None:
                df = df.Filter('E + nonIonE >= %r' % float(min_edep), 'MinEdep')
            frames[(source, point.campaign)] = df

    if missing:
        print('WARNING: %d expected file(s) not found and SKIPPED. A missing campaign is '
              'not the same as an empty one - the plots simply have no point there:'
              % len(missing), file=sys.stderr)
        for m in missing[:10]:
            print('  ' + m, file=sys.stderr)
        if len(missing) > 10:
            print('  ... and %d more' % (len(missing) - 10), file=sys.stderr)

    if not frames:
        raise SystemExit(
            'No input files found for scan %r. Run run_stmdet_tree.py first, or pass '
            '--output-root if the files are not under %s.'
            % (scan.name, scan.output_dir(output_root)))
    return frames


def load_genealogy(scan, sources=None, campaigns=None, output_root=None,
                   directory=TREE_DIR, verbose=False):
    """The same, over `ttree_genealogy`, for the back-trace plots.

    The module writes that tree only when WriteGenealogy is true, so a file may legitimately
    not have one; such files are dropped and reported rather than failing the job. The
    presence test is done with TFile::Get rather than by catching an RDataFrame exception,
    since a missing tree there is a C++ throw and not a Python one.
    """
    full = tree_path(GENEALOGY_TREE, directory)
    frames = {}
    skipped = []
    for key in load_scan(scan, sources, campaigns, output_root,
                         directory=directory, verbose=verbose):
        source, campaign = key
        path = scan.output_path(source, campaign, output_root)
        f = ROOT.TFile.Open(path)
        has_tree = bool(f and not f.IsZombie() and f.Get(full))
        if f:
            f.Close()
        if not has_tree:
            skipped.append(path)
            continue
        frames[key] = ROOT.RDataFrame(full, path)

    if skipped:
        print('WARNING: %d file(s) have no %s and were skipped. The back-trace plots need '
              'it - re-run the driver without --no-genealogy:' % (len(skipped), full),
              file=sys.stderr)
        for s in skipped[:5]:
            print('  ' + s, file=sys.stderr)
    return frames


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def detector_cut(detector):
    return 'detector == %d' % int(detector)


def pdg_cut(pdg):
    return 'pdgId == %d' % int(pdg)


def book(frame, cut=None, columns=('E', 'nonIonE')):
    """Book a count and a sum per column on one frame, WITHOUT running the loop.

    Returns (count_result, {column: sum_result}) of lazy RResultPtrs. Booking everything
    before reading any of it is what keeps a file to a single event loop; see materialize.
    """
    node = frame.Filter(cut) if cut else frame
    return node.Count(), {c: node.Sum(c) for c in columns}


def materialize(booked):
    """Force the event loops for a list of (count, sums) pairs and return plain numbers.

    RDataFrame runs every booked result of a frame in one pass the first time any of them is
    read, so the ordering here matters only for clarity - but reading them together keeps
    that property obvious at the call site.
    """
    out = []
    for count_result, sum_results in booked:
        out.append((int(count_result.GetValue()),
                    {c: float(r.GetValue()) for c, r in sum_results.items()}))
    return out


def species_present(frames, detector=None, min_counts=1):
    """PDG ids appearing across the frames, commonest first.

    Used to decide which per-species curves are worth drawing: a species with a handful of
    counts across a scan makes a line of statistical noise. One pass per file, booked
    together so each file is read once.
    """
    booked = []
    for frame in frames.values():
        node = frame.Filter(detector_cut(detector)) if detector is not None else frame
        booked.append(node.Take['int']('pdgId'))

    tally = {}
    for result in booked:
        for pdg in result.GetValue():
            tally[int(pdg)] = tally.get(int(pdg), 0) + 1
    return [pdg for pdg, n in sorted(tally.items(), key=lambda kv: -kv[1])
            if n >= min_counts]


def non_ionizing_total(frames):
    """Summed nonIonE across the frames - expected to be exactly 0.

    Geant4 fills G4Step::GetNonIonizingEnergyDeposit() only when a process calls
    SetNonIonizingEnergyDeposit(), which no standard EM or hadronic process does. This was
    checked against a real ShieldingM campaign: its high-precision neutron physics improves
    cross sections and secondary production but performs no NIEL partitioning, and the field
    came back zero exactly as under QGSP_BERT_EMZ. Every other consumer of it in Offline is
    a pass-through, a dumper or a print; none interprets it.

    So the plots carry no non-ionizing quantity. This helper exists to CHECK that assumption
    rather than assume it: a non-zero result means something in the simulation changed and
    the quantity has become worth plotting.
    """
    booked = [f.Sum('nonIonE') for f in frames.values()]
    total = sum(float(r.GetValue()) for r in booked)
    if total != 0.0:
        print('NOTE: non-ionizing energy sums to %g MeV, not 0. Something now populates '
              'nonIonizingEDep - it may be worth plotting after all. See the comment on '
              'QUANTITIES in plot_scan_rates.py.' % total, file=sys.stderr)
    return total
