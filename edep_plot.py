#!/usr/bin/env python3
"""Summarise the HPGe / LaBr trees written by the stage-2 dump.

Input is one or more ROOT files produced by the dump script, each containing
    Stage2HPGe/ttree   E (MeV), time (ns)
    Stage2LaBr/ttree   E (MeV), time (ns)

All files given on the command line are SUMMED into one result per detector -
the trees are chained, so the counts, totals and histograms describe the whole
set rather than any individual file. For each detector, produces a 3-pad canvas:
    counts vs time          (hit rate, 0-5000 ns)
    total EDep vs time      (energy deposited per time bin, 0-5000 ns)
    energy spectrum         (counts vs E)
with the entry count and total deposited energy printed on the pads.

NOTE on what 'E' means: for LaBr it is RAW IONIZING ENERGY DEPOSITED, with no
light yield, non-proportionality or resolution applied - it is not a simulated
LaBr spectrum. For HPGe it is the summed deposit per top-parent particle, not a
digitized pulse height. Neither is a detector response.

Run as: $ python plotDetectorTrees.py <file.root> [more.root ...]
        $ python plotDetectorTrees.py -o combined '/path/to/*.root'
"""

import argparse
import os
import sys
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

DETECTORS = [
    ('Stage2HPGe', 'HPGe', ROOT.kAzure + 2),
    ('Stage2LaBr', 'LaBr', ROOT.kOrange + 7),
]

T_MIN, T_MAX, T_NBINS = 0.0, 1700.0, 170     # 10 ns bins, folded by microbunch
T_PERIOD = 1700.0                            # fold period [ns]
E_MIN, E_MAX, E_NBINS = 0.0, 20.0, 20        # 10 keV bins; widen if E overflows


def summarise(paths, outPrefix):
    """Chain every input file and fill one set of histograms per detector."""
    for dirName, label, colour in DETECTORS:
        # TChain does the merge: one logical tree over all files. A file missing
        # this directory contributes nothing rather than erroring, which lets a
        # mixed set (e.g. HPGe-only jobs) be passed in without special-casing.
        chain = ROOT.TChain(f'{dirName}/ttree')
        nFiles = 0
        for path in paths:
            f = ROOT.TFile.Open(path)
            if not f or f.IsZombie():
                raise RuntimeError(f'Cannot open {path}')
            hasTree = bool(f.Get(f'{dirName}/ttree'))
            f.Close()
            if not hasTree:
                print(f'{label}: no {dirName}/ttree in {path}, skipping that file')
                continue
            chain.Add(path)
            nFiles += 1

        if nFiles == 0:
            print(f'{label}: no input file contained {dirName}/ttree, skipping')
            continue

        hN = ROOT.TH1D(f'hN_{label}', '', T_NBINS, T_MIN, T_MAX)
        hE = ROOT.TH1D(f'hE_{label}', '', T_NBINS, T_MIN, T_MAX)
        hS = ROOT.TH1D(f'hS_{label}', '', E_NBINS, E_MIN, E_MAX)
        for h in (hN, hE, hS):
            h.SetDirectory(0)   # detach, so closing a chained file cannot delete it

        # One pass over the chain. hN counts entries per time bin; hE uses E as
        # the fill weight so it accumulates TOTAL deposited energy per bin; hS is
        # the spectrum.
        nTotal, eTotal = 0, 0.0
        nOutOfTime, nOutOfE = 0, 0
        for entry in chain:
            E, t = entry.E, entry.time
            if E<0.1:
                continue
            nTotal += 1
            eTotal += E
            # Fold into one microbunch. Python's % returns a non-negative result
            # for a positive modulus even when t < 0, so early times wrap correctly
            # rather than landing outside the axis.
            tFold = t % T_PERIOD
            hN.Fill(tFold)
            hE.Fill(tFold, E)
            if E_MIN <= E < E_MAX:
                hS.Fill(E)
            else:
                nOutOfE += 1

        # Totals cover ALL entries in ALL files, including any outside a plotted
        # window, so the printed numbers describe the full sample even when the
        # histograms do not.
        print(f'--- {label} ({nFiles} file{"s" if nFiles != 1 else ""}) ---')
        print(f'  entries              : {nTotal}')
        print(f'  total deposited E    : {eTotal:.4g} MeV')
        if nTotal:
            print(f'  mean E per entry     : {eTotal / nTotal:.4g} MeV')
        if nOutOfTime:
            print(f'  outside {T_MIN:.0f}-{T_MAX:.0f} ns : {nOutOfTime} '
                  f'({100.0 * nOutOfTime / nTotal:.2f}%) - not in the time plots')
        if nOutOfE:
            print(f'  outside {E_MIN:.1f}-{E_MAX:.1f} MeV : {nOutOfE} '
                  f'({100.0 * nOutOfE / nTotal:.2f}%) - not in the spectrum')

        for h, xt, yt in ((hN, 'time [ns]', 'counts / 20 ns'),
                          (hE, 'time [ns]', 'total E_{dep} [MeV] / 20 ns'),
                          (hS, 'E_{dep} [MeV]', 'counts / 10 keV')):
            h.SetLineColor(colour)
            h.SetLineWidth(2)
            h.GetXaxis().SetTitle(xt)
            h.GetYaxis().SetTitle(yt)
            h.GetYaxis().SetTitleOffset(1.5)

        c = ROOT.TCanvas(f'c_{label}', label, 900, 1100)
        c.Divide(1, 3)

        txt = ROOT.TLatex()
        txt.SetNDC(True)
        txt.SetTextFont(42)

        header = f'#bf{{{label}}}'
        if nFiles > 1:
            header += f'  ({nFiles} files)'

        c.cd(1)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetRightMargin(0.04)
        ROOT.gPad.SetLogy(1)          # rates usually span decades
        hN.Draw('HIST')
        txt.SetTextSize(0.06)
        txt.DrawLatex(0.18, 0.85, header)
        txt.SetTextSize(0.05)
        txt.DrawLatex(0.18, 0.77, f'entries = {nTotal}')

        c.cd(2)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetRightMargin(0.04)
        ROOT.gPad.SetLogy(1)
        hE.Draw('HIST')
        txt.SetTextSize(0.05)
        txt.DrawLatex(0.18, 0.85, f'#Sigma E_{{dep}} = {eTotal:.4g} MeV')
        if nTotal:
            txt.DrawLatex(0.18, 0.77, f'mean E = {eTotal / nTotal:.4g} MeV')

        c.cd(3)
        ROOT.gPad.SetLeftMargin(0.14)
        ROOT.gPad.SetRightMargin(0.04)
        ROOT.gPad.SetLogy(1)
        hS.Draw('HIST')
        txt.SetTextSize(0.05)
        txt.DrawLatex(0.18, 0.85, f'entries = {nTotal}')

        c.Update()
        out = f'{outPrefix}_{label}_summary.png'
        c.SaveAs(out)
        print(f'  wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='input ROOT file(s); all are summed')
    ap.add_argument('-o', '--out', default=None,
                    help='output name prefix (default: derived from the first input)')
    args = ap.parse_args()

    # With one file, keep the old naming so single-file output is unchanged.
    # With several, a neutral prefix avoids implying the plots describe file #1.
    if args.out:
        prefix = args.out
    elif len(args.files) == 1:
        prefix = args.files[0][:-5] if args.files[0].endswith('.root') else args.files[0]
    else:
        prefix = os.path.join(os.path.dirname(args.files[0]) or '.', 'combined')

    summarise(args.files, prefix)