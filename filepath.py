from __future__ import print_function
import sys, os, subprocess
import numpy as np
from array import array

root_file = {
    "stm_pawel":
    {
        "Ele" :["/exp/mu2e/data/users/yongyiwu/stm_pawel/sim.plesniak.Stage1.Ele1.root",
                "/exp/mu2e/data/users/yongyiwu/stm_pawel/sim.plesniak.Stage1.Ele2.root"],
        "Mu"  :["/exp/mu2e/data/users/yongyiwu/stm_pawel/sim.plesniak.Stage1.Mu1.root",
                "/exp/mu2e/data/users/yongyiwu/stm_pawel/sim.plesniak.Stage1.Mu2.root",
                "/exp/mu2e/data/users/yongyiwu/stm_pawel/sim.plesniak.Stage1.Mu3.root"],
        "1809":["/exp/mu2e/data/users/yongyiwu/stm_pawel/dts.plesniak.Stage1.1809.root"]
    },
    "stm_baseline":
    {
        "Ele" :["/exp/mu2e/data/users/yongyiwu/stm_baseline/datasets/MDC2020/rootfiles/stm_baseline.MDC2020s10r0000_sim_BeamToVDEle.1204_000.root",
                "/exp/mu2e/data/users/yongyiwu/stm_baseline/datasets/MDC2020/rootfiles/stm_baseline.MDC2020s10r0000_sim_BeamToVDEle.1204_001.root",
                "/exp/mu2e/data/users/yongyiwu/stm_baseline/datasets/MDC2020/rootfiles/stm_baseline.MDC2020s10r0000_sim_BeamToVDEle.1204_002.root",
                "/exp/mu2e/data/users/yongyiwu/stm_baseline/datasets/MDC2020/rootfiles/stm_baseline.MDC2020s10r0000_sim_BeamToVDEle.1204_003.root"],
        "Mu"  :["/exp/mu2e/data/users/yongyiwu/stm_baseline/datasets/MDC2020/rootfiles/stm_baseline.MDC2020s11r0000_sim_BeamToVDMu.1205_000.root",
                "/exp/mu2e/data/users/yongyiwu/stm_baseline/datasets/MDC2020/rootfiles/stm_baseline.MDC2020s11r0000_sim_BeamToVDMu.1205_001.root",
                "/exp/mu2e/data/users/yongyiwu/stm_baseline/datasets/MDC2020/rootfiles/stm_baseline.MDC2020s11r0000_sim_BeamToVDMu.1205_002.root",
                "/exp/mu2e/data/users/yongyiwu/stm_baseline/datasets/MDC2020/rootfiles/stm_baseline.MDC2020s11r0000_sim_BeamToVDMu.1205_003.root"],
        "1809":["/exp/mu2e/data/users/yongyiwu/stm_baseline/datasets/MDC2020/rootfiles/stm_baseline.MDC2020s12r0000_sim_BeamToVD1809.1206_000.root"]
    },
    "stm_mod1":
    {
        "Ele" :["/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s10r0000_sim_BeamToVDEle.1207_000.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s10r0000_sim_BeamToVDEle.1207_001.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s10r0000_sim_BeamToVDEle.1207_002.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s10r0000_sim_BeamToVDEle.1207_003.root"],
        "Mu"  :["/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s11r0000_sim_BeamToVDMu.1208_000.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s11r0000_sim_BeamToVDMu.1208_001.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s11r0000_sim_BeamToVDMu.1208_002.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s11r0000_sim_BeamToVDMu.1208_003.root"],
        "1809":["/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s12r0000_sim_BeamToVD1809.1209_000.root"],
        "Neutrals":["/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_000.root",
                    "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_001.root",
                    "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_002.root",
                    "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_003.root",
                    "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_004.root",
                    "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_005.root",
                    "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_006.root",
                    "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_007.root",
                    "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_008.root",
                    "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s13r0000_sim_NeutralsToVD.1216_009.root"]
    },
    "stm_mod2":
    {
        "Ele" :["/exp/mu2e/data/users/yongyiwu/stm_mod2/datasets/MDC2020/rootfiles/stm_mod2.MDC2020s10r0000_sim_BeamToVDEle.1210_000.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod2/datasets/MDC2020/rootfiles/stm_mod2.MDC2020s10r0000_sim_BeamToVDEle.1210_001.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod2/datasets/MDC2020/rootfiles/stm_mod2.MDC2020s10r0000_sim_BeamToVDEle.1210_002.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod2/datasets/MDC2020/rootfiles/stm_mod2.MDC2020s10r0000_sim_BeamToVDEle.1210_003.root"],
        "Mu"  :["/exp/mu2e/data/users/yongyiwu/stm_mod2/datasets/MDC2020/rootfiles/stm_mod2.MDC2020s11r0000_sim_BeamToVDMu.1211_000.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod2/datasets/MDC2020/rootfiles/stm_mod2.MDC2020s11r0000_sim_BeamToVDMu.1211_001.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod2/datasets/MDC2020/rootfiles/stm_mod2.MDC2020s11r0000_sim_BeamToVDMu.1211_002.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod2/datasets/MDC2020/rootfiles/stm_mod2.MDC2020s11r0000_sim_BeamToVDMu.1211_003.root"],
        "1809":["/exp/mu2e/data/users/yongyiwu/stm_mod2/datasets/MDC2020/rootfiles/stm_mod2.MDC2020s12r0000_sim_BeamToVD1809.1212_000.root"],
    },
    "stm_mod3":
    {
        "Ele" :["/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s10r0000_sim_BeamToVDEle.1213_000.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s10r0000_sim_BeamToVDEle.1213_001.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s10r0000_sim_BeamToVDEle.1213_002.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s10r0000_sim_BeamToVDEle.1213_003.root"],
        "Mu"  :["/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s11r0000_sim_BeamToVDMu.1214_000.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s11r0000_sim_BeamToVDMu.1214_001.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s11r0000_sim_BeamToVDMu.1214_002.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s11r0000_sim_BeamToVDMu.1214_003.root"],
        "1809":["/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s12r0000_sim_BeamToVD1809.1215_000.root"]
    }
}

neutron_genealogy_root_files = {
    "stm_mod1":
    {
        "Ele" :["/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s10r0000_sim_BeamToVDEle.1207_000.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s10r0000_sim_BeamToVDEle.1207_001.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s10r0000_sim_BeamToVDEle.1207_002.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s10r0000_sim_BeamToVDEle.1207_003.NeutronBackTrace.root"],
        "Mu"  :["/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s11r0000_sim_BeamToVDMu.1208_000.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s11r0000_sim_BeamToVDMu.1208_001.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s11r0000_sim_BeamToVDMu.1208_002.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s11r0000_sim_BeamToVDMu.1208_003.NeutronBackTrace.root"],
        "1809":["/exp/mu2e/data/users/yongyiwu/stm_mod1/datasets/MDC2020/rootfiles/stm_mod1.MDC2020s12r0000_sim_BeamToVD1809.1209_000.NeutronBackTrace.root"]
    },
    "stm_mod3":
    {
        "Ele" :["/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s10r0000_sim_BeamToVDEle.1213_000.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s10r0000_sim_BeamToVDEle.1213_001.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s10r0000_sim_BeamToVDEle.1213_002.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s10r0000_sim_BeamToVDEle.1213_003.NeutronBackTrace.root"],
        "Mu"  :["/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s11r0000_sim_BeamToVDMu.1214_000.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s11r0000_sim_BeamToVDMu.1214_001.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s11r0000_sim_BeamToVDMu.1214_002.NeutronBackTrace.root",
                "/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s11r0000_sim_BeamToVDMu.1214_003.NeutronBackTrace.root"],
        "1809":["/exp/mu2e/data/users/yongyiwu/stm_mod3/datasets/MDC2020/rootfiles/stm_mod3.MDC2020s12r0000_sim_BeamToVD1809.1215_000.NeutronBackTrace.root"]
    }
}

input_event_count = {
    "stm_pawel":
    {
        "Ele" :[9677*2000000,
                5943*2000000],
        "Mu"  :[2928*2000000,
                9702*2000000,
                10000*322800],
        "1809":[100*1547943]
    },
    "stm_baseline":
    {
        "Ele" :[1000*1000000,
                1000*1000000,
                1000*1000000,
                1000*1000000],
        "Mu"  :[1000*1080700,
                1000*1080700,
                1000*1080700,
                1000*1080700],
        "1809":[100*1547943]
    },
    "stm_mod1":
    {
        "Ele" :[1000*1000000,
                1000*1000000,
                1000*1000000,
                1000*1000000],
        "Mu"  :[1000*1080700,
                1000*1080700,
                1000*1080700,
                1000*1080700],
        "1809":[100*1547943],
        "Neutrals":[881*1335926,
                    938*1335926,
                    887*1335926,
                    908*1335926,
                    929*1335926,
                    921*1335926,
                    894*1335926,
                    901*1335926,
                    893*1335926,
                    927*1335926]
    },
    "stm_mod2":
    {
        "Ele" :[1000*1000000,
                1000*1000000,
                1000*1000000,
                1000*1000000],
        "Mu"  :[1000*1080700,
                1000*1080700,
                1000*1080700,
                1000*1080700],
        "1809":[100*1547943]
    },
    "stm_mod3":
    {
        "Ele" :[1000*1000000,
                1000*1000000,
                1000*1000000,
                1000*1000000],
        "Mu"  :[1000*1080700,
                1000*1080700,
                1000*1080700,
                1000*1080700],
        "1809":[100*1547943]
    }
}

goal_POT = float(1e13/10.) #1/10 super-cycle event count

last_stage_output = {
    "Ele"     : 32925729.,
    "Mu"      :   869305.,
    "1809"    :  1432535.,
    "Neutrals":133592516.
}
last_stage_POT = {
    "Ele" :float(2e8),
    "Mu"  :float(2e8),
    "1809":float(4e8)/869305.*float(2e8),
    "Neutrals":float(2e8)
}

def getrootlist(geometry, simtype):
    return root_file[geometry][simtype]

def getrootlist_neutrongenealogy(geometry, simtype):
    return neutron_genealogy_root_files[geometry][simtype]

def getinputevent(geometry, simtype):
    return sum(root_file[geometry][simtype])
