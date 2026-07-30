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
    df_neutron = pd.DataFrame()
    df_genealogy = pd.DataFrame()
    myPOT = sum(filepath.input_event_count[geometry][tag])/filepath.last_stage_output[tag]*filepath.last_stage_POT[tag]
    weight = filepath.goal_POT/myPOT
    
    for ii, filename in enumerate(fileList):
        if verbose:
            print("Opening "+filename)

        fFile = ROOT.TFile(filename, "READ")
        fDir  = fFile.GetDirectory("NeutronBackTracer")
        fTree_neutron = fDir.Get("ttree_neutron")
        fTree_genealogy = fDir.Get("ttree_neutron_genealogy")

        nEntry_neutron = fTree_neutron.GetEntries()
        nEntry_genealogy = fTree_genealogy.GetEntries()
        if verbose:
            print(nEntry_neutron," neutrons are found")
            print(nEntry_genealogy, " back trace entries are found")

        for i in range(nEntry_neutron):
            fTree_neutron.GetEntry(i)

            if chooseVD != None:
                if fTree_neutron.virtualdetectorId != chooseVD:
                    continue

            dfn_ = pd.DataFrame()
            dfn_['tag'] = [tag] # Mu, Ele, 1809, N0
            dfn_['fileno'] = [ii]
            dfn_['index'] = [fTree_neutron.index] #index/I
            dfn_['time'] = [fTree_neutron.time] #time/D
            dfn_['virtualdetectorId'] = [fTree_neutron.virtualdetectorId] #virtualdetectorId/l
            dfn_['creationCode'] = [fTree_neutron.creationCode] #creationCode/I
            dfn_['pdgId'] = [fTree_neutron.pdgId] #pdgId/I
            dfn_['x'] = [fTree_neutron.x] #x/D
            # dfn_['x'] = [fTree_neutron.x-constants.VD101_X] #x/D
            dfn_['y'] = [fTree_neutron.y] #y/D
            dfn_['z'] = [fTree_neutron.z] #z/D
            dfn_['startx'] = [fTree_neutron.startx] #startx/D
            dfn_['starty'] = [fTree_neutron.starty] #starty/D
            dfn_['startz'] = [fTree_neutron.startz] #startz/D
            dfn_['starttime'] = [fTree_neutron.starttime] #startz/D
            dfn_['E'] = [fTree_neutron.E*1000.] #E/D, in keV
            # dfn_['r'] = [np.sqrt((fTree_neutron.x-constants.VD101_X)**2+(fTree_neutron.y)**2)] #radius w.r.t. beamline
            dfn_['weight'] = [weight]
            
            df_neutron = pd.concat([df_neutron,dfn_], ignore_index=True)
            if len(df_neutron)%50000==1 and verbose:
                print(len(df_neutron)," neutrons are collected")

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
            dfg_['startx'] = [fTree_genealogy.x] #x/D
            dfg_['starty'] = [fTree_genealogy.y] #y/D
            dfg_['startz'] = [fTree_genealogy.z] #z/D
            dfg_['E'] = [fTree_genealogy.E*1000.] #E/D, in keV
            
            df_genealogy = pd.concat([df_genealogy,dfg_], ignore_index=True)
            if len(df_genealogy)%50000==1 and verbose:
                print(len(df_genealogy)," back trace entries are collected")

        fFile.Close()

    return df_neutron, df_genealogy

