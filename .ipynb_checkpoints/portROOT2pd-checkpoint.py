from __future__ import print_function
import sys, os
import numpy as np
from array import array

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
from ROOT import gROOT, gStyle, gDirectory, gPad
import pandas as pd

import filepath
import constants

def PortToDF(geometry, tag, fileList, verbose = False, chooseVD = 101):
    df = pd.DataFrame()
    df_genealogy = pd.DataFrame()
    df_source = pd.DataFrame()
    # myPOT = sum(filepath.input_event_count[geometry][tag])/filepath.last_stage_output[tag]*filepath.last_stage_POT[tag]
    # weight = filepath.goal_POT/myPOT
    weight = 1
    
    for ii, filename in enumerate(fileList):
        if verbose:
            print("Opening "+filename)

        fFile = ROOT.TFile(filename, "READ")
        # fDir  = fFile.GetDirectory("Stage1virtualdetector")
        fDir  = fFile.GetDirectory("ParticleBackTracer")
        fTree = fDir.Get("ttree")
        fTree_genealogy = fDir.Get("ttree_genealogy")
        fTree_source = fDir.Get("ttree_source")

        nEntry = fTree.GetEntries()
        nEntry_genealogy = fTree_genealogy.GetEntries()
        nEntry_source = fTree_source.GetEntries()
        if nEntry != nEntry_source:
            print("ERROR: ttree and ttree_source have different number of entries!")
        if verbose:
            print(nEntry," entries are found")
            print(nEntry_genealogy, " back trace entries are found")

        for i in range(nEntry):
            fTree.GetEntry(i)

            if chooseVD != None:
                if fTree.virtualdetectorId != chooseVD:
                    continue

            df_ = pd.DataFrame()
            df_['tag'] = [tag] # Mu, Ele, 1809, Neutrals
            df_['fileno'] = [ii]
            df_['index'] = [fTree.index] #index/I
            df_['time'] = [fTree.time] #time/D
            df_['virtualdetectorId'] = [fTree.virtualdetectorId] #virtualdetectorId/l
            df_['creationCode'] = [fTree.creationCode] #creationCode/I
            df_['pdgId'] = [fTree.pdgId] #pdgId/I
            df_['x'] = [fTree.x] #x/D
            df_['y'] = [fTree.y] #y/D
            df_['z'] = [fTree.z] #z/D
            df_['px'] = [fTree.px*1000.] #px/D
            df_['py'] = [fTree.py*1000.] #py/D
            df_['pz'] = [fTree.pz*1000.] #pz/D
            df_['startx'] = [fTree.startx] #startx/D
            df_['starty'] = [fTree.starty] #starty/D
            df_['startz'] = [fTree.startz] #startz/D
            df_['startpx'] = [fTree.startpx*1000.] #startpx/D
            df_['startpy'] = [fTree.startpy*1000.] #startpy/D
            df_['startpz'] = [fTree.startpz*1000.] #startpz/D
            df_['starttime'] = [fTree.starttime] #starttime/D
            df_['E'] = [fTree.E*1000.] #E/D, in keV
            df_['r'] = [np.sqrt((fTree.x-constants.VD101_X)**2+(fTree.y)**2)] #radius w.r.t. beamline
            df_['weight'] = [weight]
            
            df = pd.concat([df,df_], ignore_index=True)
            if len(df)%10000==1 and verbose:
                print(len(df)," entries are collected")
                
        for j in range(nEntry_genealogy):
            fTree_genealogy.GetEntry(j)

            dfg_ = pd.DataFrame()
            dfg_['tag'] = [tag] # Mu, Ele, 1809, N0
            dfg_['fileno'] = [ii]
            dfg_['index'] = [fTree_genealogy.index] #index/I
            dfg_['starttime'] = [fTree_genealogy.startGlobalTime] #startGlobalTime/D
            dfg_['endtime'] = [fTree_genealogy.endGlobalTime] #endGlobalTime/D
            dfg_['creationCode'] = [fTree_genealogy.creationCode] #creationCode/I
            dfg_['pdgId'] = [fTree_genealogy.pdgId] #pdgId/I
            dfg_['startx'] = [fTree_genealogy.startx] #startx/D
            dfg_['starty'] = [fTree_genealogy.starty] #starty/D
            dfg_['startz'] = [fTree_genealogy.startz] #startz/D
            dfg_['endx'] = [fTree_genealogy.endx] #endx/D
            dfg_['endy'] = [fTree_genealogy.endy] #endy/D
            dfg_['endz'] = [fTree_genealogy.endz] #endz/D
            dfg_['startpx'] = [fTree_genealogy.startpx*1000.] #startpx/D
            dfg_['startpy'] = [fTree_genealogy.startpy*1000.] #startpy/D
            dfg_['startpz'] = [fTree_genealogy.startpz*1000.] #startpz/D
            dfg_['endpx'] = [fTree_genealogy.endpx*1000.] #endpx/D
            dfg_['endpy'] = [fTree_genealogy.endpy*1000.] #endpy/D
            dfg_['endpz'] = [fTree_genealogy.endpz*1000.] #endpz/D
            dfg_['E'] = [fTree_genealogy.E*1000.] #E/D, in keV
            
            df_genealogy = pd.concat([df_genealogy,dfg_], ignore_index=True)
            if len(df_genealogy)%10000==1 and verbose:
                print(len(df_genealogy)," back trace entries are collected")

        for k in range(nEntry_source):
            fTree_source.GetEntry(k)

            dfs_ = pd.DataFrame()
            dfs_['tag'] = [tag] # Mu, Ele, 1809, N0
            dfs_['fileno'] = [ii]
            dfs_['index'] = [fTree_source.index] #index/I
            dfs_['starttime'] = [fTree_source.starttime] # starttime/D, in ns
            dfs_['parentendtime'] = [fTree_source.parentendtime] # parentendtime/D, in ns
            dfs_['creationCode'] = [fTree_source.creationCode] # creationCode/I
            dfs_['parentpdgId'] = [fTree_source.parentpdgId] # parentpdgId/I
            dfs_['x'] = [fTree_source.x] # x/D, in mm
            dfs_['y'] = [fTree_source.y] # y/D, in mm
            dfs_['z'] = [fTree_source.z] # z/D, in mm
            
            df_source = pd.concat([df_source,dfs_], ignore_index=True)
            if len(df_source)%10000==1 and verbose:
                print(len(df_source)," parent info entries are collected")

        fFile.Close()

    return df, df_genealogy, df_source