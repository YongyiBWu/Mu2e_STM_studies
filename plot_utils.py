from __future__ import print_function
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from array import array

import constants
import pdgid

def single_hist(df, item, title, rangemin = None, rangemax = None, stepsize = None): 
    # df of a particle from a tag
    # item = ['time', 'x', 'y', 'E', 'r']
    # title[0]: title, title[1]: xtitle, title[2]: ytitle

    fig, ax = plt.subplots(figsize=(4,3))

    if rangemin is None:
        if item in ['x','y']:
            rangemin = -110.
        elif item in ['time','E','r']:
            rangemin = 0.
        elif item == 'pz':
            rangemin = -50.
        elif item == 'startz':
            rangemin = 3000.
        else:
            rangemin = np.min(df[item])
    if rangemax is None:
        if item in ['x','y','r']:
            rangemax = 110.
        elif item == 'E':
            #rangemax = 2000.
            rangemax = 10000.
        elif item == 'time':
            rangemax = 5000.
        elif item == 'pz':
            rangemax = 20000.
        elif item == 'startz':
            rangemax = 7000.    
        else:
            rangemax = np.max(df[item])
    if stepsize is None:
        if item in ['x','y']:
            stepsize = 5.
        elif item == 'r':
            stepsize = 2.5
        elif item == 'E':
            stepsize = 50.
        elif item == 'time':
            stepsize = 50.
        elif item == 'pz':
            stepsize = 50.
        elif item == 'startz':
            stepsize = 40.
        else:
            stepsize = (rangemax-rangemin)/50.
    nbins = int((rangemax-rangemin)/stepsize)
    bins = [float(rangemin)+stepsize*float(i) for i in range(nbins+1)]
    
    ax.hist(df[item], bins, histtype='step')
    ax.set_title(title[0])
    ax.set_xlabel(title[1])
    ax.set_ylabel(title[2])

    entries = len(df[item])
    mean = np.mean(df[item])
    std = np.std(df[item])
    underflow = (df[item] < rangemin).sum()
    overflow  = (df[item] > rangemax).sum()
    textstr = (
        f"Entries:   {entries}\n"
        f"Mean:      {mean:.3f}\n"
        f"Std Dev:   {std:.3f}\n"
        f"Underflow: {underflow}\n"
        f"Overflow:  {overflow}"
    )
    ax.text(
        1.5, 0.97, textstr,
        transform=ax.transAxes,
        fontsize=10,
        va='top', ha='right',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )

    return fig, ax
    
def draw_stm_projection(ax):
    circles = [
        ((0., constants.VD101_Y), constants.VD101_R),     # (center), radius
        ((constants.hole1_X, constants.hole1_Y), constants.hole1_R),
        ((constants.hole1_X, constants.hole1_Y), constants.hole1_r),
        ((constants.hole2_X, constants.hole2_Y), constants.hole2_R),
        ((constants.hole2_X, constants.hole2_Y), constants.hole2_r)
    ]
    for center, radius in circles:
        cx, cy = center
        circ = Circle((cx, cy), radius,
                      edgecolor='cyan',
                      facecolor='none',
                      linewidth=1)
        ax.add_patch(circ)
    ax.set_aspect('equal')
    return

def particle_space_dist(df, title, forcemax = None):
    # xbins = np.linspace(-110., 110., 45)
    xbins = np.linspace(-510.+constants.VD101_X, 510.+constants.VD101_X, 103)
    # ybins = np.linspace(-110., 110., 45)
    ybins = np.linspace(-510., 510., 103)
        
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(8, 10, figure=fig,
                  wspace=0.05, hspace=0.05) 
    
    ax2d = fig.add_subplot(gs[2:8, 2:8]) # 2D heatmap in bottom-left
    axX = fig.add_subplot(gs[0:2, 2:8], sharex=ax2d) # x projection on top
    axY = fig.add_subplot(gs[2:8, 8:10], sharey=ax2d) # y projection on right
    cax = fig.add_subplot(gs[2:8, 0])   # colorbar in its own slot

    H, xe, ye = np.histogram2d(
        df["x"], df["y"],
        #df["x"]-constants.VD101_X, df["y"],
        bins=[xbins, ybins],
        weights=df["weight"]
    )
    mesh = ax2d.pcolormesh(xbins, ybins, H.T, cmap="inferno", vmax=forcemax)
    
    fig.colorbar(mesh, cax=cax, label="Weighted counts", location="left", pad=0.2, orientation='vertical')

    axX.hist(df["x"], bins=xbins, color="black", histtype='step', weights=df["weight"])
    #axX.hist(df["x"]-constants.VD101_X, bins=xbins, color="black", histtype='step', weights=df["weight"])
    axX.set_ylabel("Counts/10 mm")
    axX.tick_params(axis="x", labelbottom=False)

    axY.hist(df["y"], bins=ybins, orientation="horizontal", color="black", histtype='step', weights=df["weight"])
    axY.set_xlabel("Counts/10 mm")
    axY.tick_params(axis="y", labelleft=False)

    ax2d.set_xlabel("x [mm]")
    ax2d.set_ylabel("y [mm]")

    fig.suptitle(title, y=0.93)
    #draw_stm_projection(ax2d)
    
    return fig, ax2d, axX, axY

def spectra_stacked_sgnl_bkgd(df, title, forcemax = None, weighted = True):
    bins_main = np.linspace(0., 2000., 201) # 10keV
    bins_347  = np.linspace(320., 350., 7) # 5keV
    bins_844  = np.linspace(830., 860., 7)
    bins_1809 = np.linspace(1795., 1825., 7)
    mybins = [bins_main, bins_347, bins_844, bins_1809]

    fig = plt.figure(figsize=(9, 9 if weighted else 6))
    gs = GridSpec(3 if weighted else 2, 3, figure=fig,
                  wspace=0.12, hspace=0.25) 

    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 2], sharey=ax1)
    ax = [ax0, ax1, ax2, ax3]
    if weighted:
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1], sharey=ax4)
        ax6 = fig.add_subplot(gs[2, 2], sharey=ax4)
        ax += [ax4, ax5, ax6]

    df_bkgd = df.query("tag=='Ele'").reset_index()
    df_sgnl = df.query("tag!='Ele'").reset_index()

    for i in range(4):
        tmax = forcemax
        if (forcemax != None) and (i!=0):
            tmax/=10
        ax[i].hist([df_bkgd['E'], df_sgnl['E']], bins=mybins[i], histtype="barstacked",
                   weights=[df_bkgd['weight'], df_sgnl['weight']] if weighted else None,
                   label=['background', 'signal'])
        ax[i].set_xlim(mybins[i][0],mybins[i][-1])
        ax[i].set_xlabel('E [keV]')
    if weighted:
        #cuts = ["(time > 300) & (time < 700)", "(time > 492) & (time < 1330)", "(time > 500) & (time < 1600)"]
        #textstr = ["300 ns< t< 700 ns", "492 ns< t< 1330 ns", "500 ns< t< 1600 ns"]
        cuts = ["(time > 270) & (time < 700)", "(time > 462) & (time < 1330)", "(time > 470) & (time < 1600)"]
        textstr = ["270 ns< t< 700 ns", "462 ns< t< 1330 ns", "470 ns< t< 1600 ns"]
        for j in range(3):
            plot_content = [df_bkgd.query(cuts[j]).reset_index(drop=True)['E'], df_sgnl.query(cuts[j]).reset_index(drop=True)['E']]
            myweights = [df_bkgd.query(cuts[j]).reset_index(drop=True)['weight'], df_sgnl.query(cuts[j]).reset_index(drop=True)['weight']]
            ax[4+j].hist(plot_content, bins=mybins[j+1], histtype="barstacked",
                         weights=myweights,
                         label=['background', 'signal'])
            ax[4+j].set_ylim(0.,tmax)
            ax[4+j].set_xlim(mybins[j+1][0],mybins[j+1][-1])
            ax[4+j].set_xlabel('E [keV]')
            ax[4+j].text(0.97, 0.97, textstr[j],
                         transform=ax[4+j].transAxes,
                         fontsize=10,
                         va='top', ha='right',
                         bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
                        )
                                   
    ax[0].set_ylabel('weighted counts/10 keV')
    ax[1].set_ylabel('weighted counts/5 keV')
    ax[2].tick_params(axis="y", labelleft=False)
    ax[3].tick_params(axis="y", labelleft=False)
    if weighted:
        ax[4].set_ylabel('weighted counts/5 keV')
        ax[5].tick_params(axis="y", labelleft=False)
        ax[6].tick_params(axis="y", labelleft=False)
    ax[0].legend()
    fig.suptitle(title, y=0.93)
    
    return fig, ax
        
def spectra_stacked_particle(df, title, forcemax = None, weighted = True):
    bins = np.linspace(0., 2000., 201) # 10keV

    fig, ax = plt.subplots(figsize=(9,3))

    df_stacks = []
    list_of_pdgId = sorted(df['pdgId'].unique(), key=abs)
    for pdgId in list_of_pdgId:
        mydf = df.query("pdgId==%i"%(pdgId)).reset_index()
        df_stacks.append(mydf)

    ax.hist([tdf['E'] for tdf in df_stacks], 
            bins=bins, 
            histtype="barstacked",
            weights=[tdf['weight'] for tdf in df_stacks]if weighted else None,
            label=[pdgid.pdgid_dict[tid] for tid in list_of_pdgId])

    ax.set_ylim(0.,forcemax)
    ax.set_xlim(bins[0],bins[-1])
    ax.set_xlabel('E [keV]')
    ax.set_ylabel('weighted counts/10 keV')
    ax.legend()
    fig.suptitle(title)
    
    return fig, ax

def twod_hist(df, xitem, yitem, title, forcemax = None):
    mybins = {"time":np.linspace(0., 500., 26),
              "E":np.linspace(0., 5000., 41),
              "r":np.linspace(0., 100., 41)}
    axtitle = {"time":"time / 50ns",
               "E":"E / 50 keV",
               "r":"r / 2.5 mm"}
    
    xbins = mybins[xitem]
    ybins = mybins[yitem]

    df_bkgd = df.query("tag=='Ele'").reset_index()
    df_sgnl = df.query("tag!='Ele'").reset_index()
        
    fig = plt.figure(figsize=(4, 6))
    gs = GridSpec(2, 1, figure=fig,
                  wspace=0.05, hspace=0.2) 
    
    ax1 = fig.add_subplot(gs[0, 0]) # bkgd
    ax2 = fig.add_subplot(gs[1, 0]) # sgnl
    ax = [ax1, ax2]

    H1, xe1, ye1 = np.histogram2d(
        df_bkgd[xitem], df_bkgd[yitem],
        bins=[xbins, ybins]
    )
    H2, xe2, ye2 = np.histogram2d(
        df_sgnl[xitem], df_sgnl[yitem],
        bins=[xbins, ybins],
        weights=df_sgnl["weight"]
    )
    mesh1 = ax1.pcolormesh(xbins, ybins, H1.T, cmap="inferno", vmax=forcemax)
    mesh2 = ax2.pcolormesh(xbins, ybins, H2.T, cmap="inferno", vmax=forcemax)
    
    fig.colorbar(mesh1, ax=ax1, label="unweighted counts", location="right", pad=0.05, orientation='vertical')
    fig.colorbar(mesh2, ax=ax2, label="unweighted counts", location="right", pad=0.05, orientation='vertical')

    ax1.set_xlabel(axtitle[xitem])
    ax1.set_ylabel("bkgd "+axtitle[yitem])
    ax2.set_xlabel(axtitle[xitem])
    ax2.set_ylabel("sgnl "+axtitle[yitem])

    fig.suptitle(title, y=0.93)
    
    return fig, ax

def draw_Mu2e_topview(ax):
    rectangles = [ #(lower-left x, y), width, height
        ((constants.ps_cryo_start_Z, constants.production_target_X-constants.ps_cryo_OR), constants.ps_cryo_len, constants.ps_cryo_OR*2), # PS_out
        ((constants.ps_cryo_start_Z, constants.production_target_X-constants.ps_cryo_IR), constants.ps_cryo_len, constants.ps_cryo_IR*2), # PS_in
        ((constants.ds_cryo_start_Z, constants.stopping_target_X-constants.ds_cryo_OR), constants.ds_cryo_len, constants.ds_cryo_OR*2), # DS_out
        ((constants.ds_cryo_start_Z, constants.stopping_target_X-constants.ds_cryo_IR), constants.ds_cryo_len, constants.ds_cryo_IR*2), # DS_in
        ((constants.IFB_start_Z, constants.stopping_target_X-constants.IFB_OR), constants.IFB_len, constants.IFB_OR*2), # IFB
        ((constants.stopping_target_start_Z, constants.stopping_target_X-constants.stopping_target_OR), constants.stopping_target_len, constants.stopping_target_OR*2), # stopping_target
        ((constants.trk_start_Z, constants.stopping_target_X-constants.trk_OR), constants.trl_len, constants.trk_OR*2), # trk
        ((constants.calo_start_Z, constants.stopping_target_X-constants.calo_OR), constants.calo_len, constants.calo_OR*2), # calo
        ((constants.VD101_Z, constants.stopping_target_X-1000.), 910., 1400.) # STM_area
    ]
    for corner, width, height in rectangles:
        x, y = corner
        rect = Rectangle((x, y),  width, height,
                         edgecolor='grey',
                         facecolor='none',
                         linewidth=1)
        ax.add_patch(rect)
    ax.set_ylim(-5500., 5500.)
    #ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_aspect('equal')
    ax.tick_params(axis='x', which='both',
                   bottom=False, top=False, labelbottom=False)
    return

def draw_Mu2e_sideview(ax):
    rectangles = [ #(lower-left x, y), width, height
        ((constants.ps_cryo_start_Z, -constants.ps_cryo_OR), constants.ps_cryo_len, constants.ps_cryo_OR*2), # PS_out
        ((constants.ps_cryo_start_Z, -constants.ps_cryo_IR), constants.ps_cryo_len, constants.ps_cryo_IR*2), # PS_in
        ((constants.ds_cryo_start_Z, -constants.ds_cryo_OR), constants.ds_cryo_len, constants.ds_cryo_OR*2), # DS_out
        ((constants.ds_cryo_start_Z, -constants.ds_cryo_IR), constants.ds_cryo_len, constants.ds_cryo_IR*2), # DS_in
        ((constants.IFB_start_Z, -constants.IFB_OR), constants.IFB_len, constants.IFB_OR*2), # IFB
        ((constants.stopping_target_start_Z, -constants.stopping_target_OR), constants.stopping_target_len, constants.stopping_target_OR*2), # stopping_target
        ((constants.trk_start_Z, -constants.trk_OR), constants.trl_len, constants.trk_OR*2), # trk
        ((constants.calo_start_Z, -constants.calo_OR), constants.calo_len, constants.calo_OR*2), # calo
        ((constants.VD101_Z, -254.), 910., 508.) # STM_area
    ]
    for corner, width, height in rectangles:
        x, y = corner
        rect = Rectangle((x, y),  width, height,
                         edgecolor='grey',
                         facecolor='none',
                         linewidth=1)
        ax.add_patch(rect)
    ax.set_ylim(-3500., 3500.)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    return

def draw_neutron_event_backtrace(dfn_, dfg_, tag, fileno, index):
    # grab single neutron entry
    # dfn_ = df_neutron.query("tag==@tag and fileno==@fileno and index==@index").reset_index().iloc[0]
    # grab genealogy
    # dfg_ = df_genealogy.query("tag==@tag and fileno==@fileno and index==@index").reset_index()
    #display(dfg_)
    
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(5, 1, figure=fig,
                  wspace=0.05, hspace=0.00) 
    
    ax_top  = fig.add_subplot(gs[0:3]) # top view
    ax_side = fig.add_subplot(gs[3:], sharex=ax_top) # side view

    source_id = [dfn_['pdgId']]
    
    neutron_track_top  = Line2D([dfn_['startz'], dfn_['z']], [dfn_['startx'], dfn_['x']], linewidth=2, linestyle=':', color=pdgid.pdgid_color_dict[source_id[0]])
    neutron_track_side = Line2D([dfn_['startz'], dfn_['z']], [dfn_['starty'], dfn_['y']], linewidth=2, linestyle=':', color=pdgid.pdgid_color_dict[source_id[0]])
    last_point = (dfn_['startx'],dfn_['starty'],dfn_['startz'])
    ax_top.add_line(neutron_track_top)
    ax_side.add_line(neutron_track_side)
    ax_top.scatter(dfn_['startz'], dfn_['startx'], c=pdgid.pdgid_color_dict[source_id[0]], s=5)#s=dfn_['E']/100.)
    ax_side.scatter(dfn_['startz'], dfn_['starty'], c=pdgid.pdgid_color_dict[source_id[0]], s=5)#s=dfn_['E']/100.)
    
    legend_dummy = Line2D([0], [0], linewidth=2, linestyle=':', color=pdgid.pdgid_color_dict[source_id[0]], markerfacecolor=pdgid.pdgid_color_dict[source_id[0]], markersize=0, label=pdgid.pdgid_dict[source_id[0]])
    legend_handles= []
    legend_handles.append(legend_dummy)

    for ii in range(len(dfg_)):
        this_entry = dfg_.iloc[ii]
        this_pdgid = this_entry['pdgId']
        try:
            this_color = pdgid.pdgid_color_dict[this_pdgid]
        except:
            this_color = 'yellow'
        if not this_pdgid in source_id:
            source_id.append(this_pdgid)
            try:
                legend_dummy = Line2D([0], [0], linewidth=2, linestyle=':', color=this_color, markerfacecolor=this_color, markersize=0, label=pdgid.pdgid_dict[this_pdgid])
            except:
                legend_dummy = Line2D([0], [0], linewidth=2, linestyle=':', color=this_color, markerfacecolor=this_color, markersize=0, label=str(int(this_pdgid)))
            legend_handles.append(legend_dummy)
        this_track_top  = Line2D([this_entry['startz'], last_point[2]], [this_entry['startx'], last_point[0]], linewidth=2, color=this_color, linestyle=':')
        this_track_side = Line2D([this_entry['startz'], last_point[2]], [this_entry['starty'], last_point[1]], linewidth=2, color=this_color, linestyle=':')
        ax_top.add_line(this_track_top)
        ax_side.add_line(this_track_side)
        ax_top.scatter(this_entry['startz'], this_entry['startx'], c=this_color, s=5) #, s=this_entry['E']/100.)
        ax_side.scatter(this_entry['startz'], this_entry['starty'], c=this_color, s=5) #, s=this_entry['E']/100.)
        last_point = (this_entry['startx'],this_entry['starty'],this_entry['startz'])

    ax_top.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5))

    draw_Mu2e_topview(ax_top)
    draw_Mu2e_sideview(ax_side)

    fig.suptitle("Back trace of "+tag+" %03i "%(fileno)+pdgid.pdgid_dict[dfn_['pdgId']]+" event %i"%(index), y=0.93)

    return fig, ax_top, ax_side

def draw_particle_event_backtrace(dfp_, df_genealogy, tag, fileno, index):
    return draw_neutron_event_backtrace(dfp_, df_genealogy, tag, fileno, index)

def draw_neutron_source_all(df_neutron, df_genealogy, title):
    df_source = pd.DataFrame()
    
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(5, 1, figure=fig,
                  wspace=0.05, hspace=0.0) 
    
    ax_top  = fig.add_subplot(gs[0:3]) # top view
    ax_side = fig.add_subplot(gs[3:], sharex=ax_top) # side view

    source_id = []
    source_cnt = []
    legend_handles = []

    for ii in range(len(df_neutron)):
        dfn_ = df_neutron.iloc[ii]
        tag = dfn_['tag']
        fileno = dfn_['fileno']
        index = dfn_['index']
        dfg_ = df_genealogy.query("tag==@tag and fileno==@fileno and index==@index").reset_index(drop=True)
        neutron_start_point = (dfn_['startx'],dfn_['starty'],dfn_['startz'])
        neutron_start_time = dfn_['starttime']
        neutron_initial_creation = dfn_['creationCode']
        E_at_VD = dfn_['E']
        startpx, startpy, startpz = dfn_['startpx'], dfn_['startpy'], dfn_['startpz']

        for jj in range(len(dfg_)):
            dfg_entry = dfg_.iloc[jj]
            if dfg_entry['pdgId']==dfn_['pdgId']: # neutron
                neutron_start_point = (dfg_entry['startx'],dfg_entry['starty'],dfg_entry['startz'])
                neutron_start_time = dfg_entry['starttime']
                neutron_initial_creation = dfg_entry['creationCode']
                startpx, startpy, startpz = dfg_entry['startpx'], dfg_entry['startpy'], dfg_entry['startpz']
            else:
                this_pdgid = dfg_entry['pdgId']
                parent_end_time = dfg_entry['endtime']
                break
                
        try:
            this_color = pdgid.pdgid_color_dict[this_pdgid]
        except:
            this_color = 'yellow'

        if this_pdgid in source_id:
            source_cnt[source_id.index(this_pdgid)] += 1
        else:
            source_id.append(this_pdgid)
            source_cnt.append(1)
            try:
                legend_dummy = Line2D([0], [0], marker='o', color=this_color, markerfacecolor=this_color, markersize=5, label=pdgid.pdgid_dict[this_pdgid])
            except:
                legend_dummy = Line2D([0], [0], marker='o', color=this_color, markerfacecolor=this_color, markersize=5, label=str(this_pdgid))

            legend_handles.append(legend_dummy)

        ax_top.scatter(neutron_start_point[2], neutron_start_point[0], c=this_color, s=5)#, s=this_entry['E']/100.)
        ax_side.scatter(neutron_start_point[2], neutron_start_point[1], c=this_color, s=5)#, s=this_entry['E']/100.)

        dfs_ = pd.DataFrame()
        dfs_['tag'] = [tag] # Mu, Ele, 1809, N0
        dfs_['fileno'] = [fileno]
        dfs_['index'] = [index] 
        dfs_['starttime'] = [neutron_start_time]
        dfs_['parentendtime'] = [parent_end_time]    
        dfs_['creationCode'] = [neutron_initial_creation]
        dfs_['parentId'] = [this_pdgid]
        dfs_['x'] = [neutron_start_point[0]]
        dfs_['y'] = [neutron_start_point[1]]
        dfs_['z'] = [neutron_start_point[2]]
        dfs_['EatVD'] = [E_at_VD]
        dfs_['startE'] = [np.sqrt(startpx*startpx+startpy*startpy+startpz*startpz+939565**2)-939565]
        df_source = pd.concat([df_source,dfs_], ignore_index=True)
                    
    draw_Mu2e_topview(ax_top)
    draw_Mu2e_sideview(ax_side)

    ax_top.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(title, y=0.93)
    
    print("source count")
    for i, v in enumerate(source_id):
        try:
            print(pdgid.pdgid_dict[v], ": ", source_cnt[i])
        except:
            print(v, ": ", source_cnt[i])

    return df_source, fig, ax_top, ax_side
    
def draw_particle_source_all(df_source, title):
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(5, 1, figure=fig,
                  wspace=0.05, hspace=0.0) 
    
    ax_top  = fig.add_subplot(gs[0:3]) # top view
    ax_side = fig.add_subplot(gs[3:], sharex=ax_top) # side view

    print('------------------------------------------------------------------------------')
    source_id = np.sort(df_source['parentpdgId'].unique())
    for myid in source_id:
        dfs_ = df_source.query("parentpdgId==@myid").reset_index(drop=True)

        try:
            this_parent_name = pdgid.pdgid_dict[myid]
            this_color = pdgid.pdgid_color_dict[myid]
        except:
            this_parent_name = str(myid)
            this_color = 'yellow'
        
        print("Parent pdgId: ", myid, this_parent_name)
        print("count: ", len(dfs_))
        print(dfs_['creationCode'].value_counts())
        print('******')
        
        ax_top.scatter(dfs_['z'], dfs_['x'], c=this_color, label=this_parent_name, s=5)#, s=this_entry['E']/100.)
        ax_side.scatter(dfs_['z'], dfs_['y'], c=this_color, label=this_parent_name, s=5)#, s=this_entry['E']/100.)

    draw_Mu2e_topview(ax_top)
    draw_Mu2e_sideview(ax_side)

    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title, y=0.93)

    return fig, ax_top, ax_side

def draw_Mu2e_faceview(ax, hallway=True, ds=True):
    # Overlay the x-y (beam face) geometry on ax, as seen looking downstream.
    #   hallway -- the hall cross section
    #   ds      -- the DS cryostat bore, centred on the beamline at x = stopping_target_X
    if hallway:
        ax.add_patch(Rectangle((constants.hallway_X_min, constants.hallway_Y_min),
                               constants.hallway_X_max - constants.hallway_X_min,
                               constants.hallway_Y_max - constants.hallway_Y_min,
                               edgecolor='grey', facecolor='none', linewidth=1))
    if ds:
        # outer and inner wall of the cryostat, concentric on the beamline
        for r in (constants.ds_cryo_OR, constants.ds_cryo_IR):
            ax.add_patch(Circle((constants.stopping_target_X, constants.stopping_target_Y),
                                r, edgecolor='grey', facecolor='none', linewidth=1))
    return


def find_z_crossings(dfe_, z0):
    # Find where each trajectory in dfe_ crosses the plane z = z0.
    #
    # A trajectory is a polyline through its stored points, so a crossing is a segment
    # whose endpoints straddle z0. kE (and x, y, t) at the crossing are linearly
    # interpolated between those two points. A particle that turns around can cross the
    # same plane more than once; every crossing is returned.
    #
    # Returns a DataFrame with one row per crossing, carrying tag/fileno/run/subRun/
    # event/simId/matched/hasTrajectory through so the result can still be split by tag
    # (or by event, or by hasTrajectory) later.
    #   x, y, t, kE  -- interpolated at z = z0
    #   downstream   -- True if the crossing segment runs with dz > 0
    #
    # NOTE the interpolation is only as good as the stored points. Entries with
    # hasTrajectory == False hold just the start and end position, so their "crossing"
    # is a point on the straight line between those two -- not a real tracked path.
    # draw_kE_at_z draws those at 50% alpha for that reason.
    carry = ['tag', 'fileno', 'run', 'subRun', 'event',
             'simId', 'pdgId', 'matched', 'hasTrajectory']
    rows = []
    for ii in range(len(dfe_)):
        e_ = dfe_.iloc[ii]
        z = np.asarray(e_['z'], dtype=float)
        if len(z) < 2:
            continue
        x  = np.asarray(e_['x'], dtype=float)
        y  = np.asarray(e_['y'], dtype=float)
        t  = np.asarray(e_['t'], dtype=float)
        kE = np.asarray(e_['kE'], dtype=float)

        dz_prev = z[:-1] - z0
        dz_next = z[1:] - z0
        # A segment crosses if its endpoints are on opposite sides. Half-open in dz_prev
        # (strictly < or >) so a point sitting exactly on the plane is claimed only by
        # the segment leaving it, not also by the one arriving -- otherwise one physical
        # crossing would be counted twice.
        straddle = ((dz_prev < 0) & (dz_next >= 0)) | ((dz_prev > 0) & (dz_next <= 0))
        for jj in np.nonzero(straddle)[0]:
            span = z[jj+1] - z[jj]
            if span == 0:
                continue      # segment lies in the plane; it has no crossing direction
            f = (z0 - z[jj]) / span
            row = {c: e_[c] for c in carry if c in e_.index}
            row.update({
                'x'         : x[jj] + f*(x[jj+1] - x[jj]),
                'y'         : y[jj] + f*(y[jj+1] - y[jj]),
                't'         : t[jj] + f*(t[jj+1] - t[jj]),
                'kE'        : kE[jj] + f*(kE[jj+1] - kE[jj]),
                # Points are stored in G4 step order (the module pushes traj.points() in
                # sequence), so array order is time order and the sign of dz along the
                # segment is the direction of travel. dt is checked rather than assumed:
                # if the pair is out of time order, flip the sense.
                'downstream': bool(span > 0) if t[jj+1] >= t[jj] else bool(span < 0),
            })
            rows.append(row)
    return pd.DataFrame(rows)


kE_floor = 0.1    # MeV; anything at or below this sizes as if it were exactly this

def _kE_marker_size(v, lo, hi, smin, smax):
    # Marker area proportional to log10(kE), mapped onto [smin, smax] over [lo, hi].
    # kE spans many decades, so the area tracks the exponent rather than the value.
    # Values below kE_floor (and any zero/negative, which have no log) are pinned to it,
    # so a thermal-energy particle still gets the smallest marker rather than vanishing.
    vv = np.clip(np.asarray(v, dtype=float), max(lo, kE_floor), hi)
    frac = np.log10(vv/lo) / np.log10(hi/lo)
    return smin + frac*(smax - smin)


def draw_kE_at_z(dfe_, z0, title = None, tags = None, kEmin = None, kEmax = None,
                 smin = 8., smax = 300., spectrum_bins = 50, notraj_alpha = 0.5,
                 down_alpha = 0.5, xlim = (-8000., 2000.), ylim = (-3500., 5000.),
                 geometry = True, ylog_min = 10):
    # Particles crossing the plane z = z0: where they cross, and their kE spectrum.
    #
    # Left panel  -- kE spectrum per PDG ID. Solid = MCTraj, dashed = NoTraj. x is
    #                always log; y goes log only if the tallest bar reaches ylog_min.
    # Right panel -- x vs y at the crossing, the beam face seen looking downstream.
    #   colour  : PDG ID (pdgid.pdgid_color_dict)
    #   size    : area proportional to log10(kE), between smin and smax
    #   filled  : crossing downstream (dz > 0), drawn at down_alpha so overlapping
    #             markers stay readable;  outline only : upstream
    #   alpha   : down_alpha for both, scaled by notraj_alpha again when
    #             hasTrajectory == False (those hold only start/end, so the crossing is
    #             interpolated on a straight line and is not a tracked path)
    #
    # xlim/ylim default to the beam-face region around the hallway cross section.
    # geometry overlays the hallway outline and the DS cryostat bore (draw_Mu2e_faceview).
    #
    # dfe_ is any frame of trajectory entries (the whole df_traj, or one event).
    # tags: restrict to these tags before plotting. The returned crossings frame keeps
    # its 'tag' column either way, so callers can also split the result themselves.
    if tags is not None and len(dfe_) and 'tag' in dfe_.columns:
        dfe_ = dfe_[dfe_['tag'].isin(tags)]

    dfx_ = find_z_crossings(dfe_, z0)

    fig = plt.figure(figsize=(15, 6))
    # spectrum on the left, beam face on the right; right=0.84 reserves the strip the
    # two legends occupy, outboard of ax_face
    gs = GridSpec(1, 2, figure=fig, wspace=0.25, right=0.84)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_face = fig.add_subplot(gs[0, 1])

    if not len(dfx_):
        ax_spec.text(0.5, 0.5, "no trajectory crosses z = %.1f mm" % z0,
                     ha='center', va='center', transform=ax_spec.transAxes)
        if title is not None:
            fig.suptitle(title, y=0.98)
        return fig, ax_face, ax_spec, dfx_

    kE = dfx_['kE'].values
    positive = kE[kE > 0]
    lo = kEmin if kEmin is not None else (positive.min() if len(positive) else kE_floor)
    hi = kEmax if kEmax is not None else (positive.max() if len(positive) else 1.0)
    # do not let the scale run below the floor: everything there sizes the same anyway,
    # and the decades underneath would otherwise eat the whole marker-size range
    lo = max(lo, kE_floor)
    if hi <= lo:
        hi = lo*10.

    bins = np.logspace(np.log10(lo), np.log10(hi), spectrum_bins+1)

    legend_handles = []
    peak = 0          # tallest histogram bar, decides whether the y axis goes log
    for pdg in np.sort(dfx_['pdgId'].unique()):
        d_ = dfx_[dfx_['pdgId'] == pdg]
        try:
            this_color = pdgid.pdgid_color_dict[pdg]
        except KeyError:
            this_color = 'yellow'
        try:
            this_label = pdgid.pdgid_dict[pdg]
        except KeyError:
            this_label = str(int(pdg))

        # four combinations: down/up x real trajectory / start-end only.
        # NoTraj is drawn fainter still, so the two effects multiply.
        for real in (True, False):
            amul = 1.0 if real else notraj_alpha
            sub = d_[d_['hasTrajectory'].astype(bool) == real]
            if not len(sub):
                continue
            down = sub[sub['downstream'].astype(bool)]
            up   = sub[~sub['downstream'].astype(bool)]
            # downstream: filled, but semi-transparent so overlapping markers show through
            if len(down):
                ax_face.scatter(down['x'], down['y'],
                                s=_kE_marker_size(down['kE'].values, lo, hi, smin, smax),
                                facecolors=this_color, edgecolors='none',
                                alpha=down_alpha*amul)
            # upstream: outline only, no fill; same alpha as the downstream markers
            if len(up):
                ax_face.scatter(up['x'], up['y'],
                                s=_kE_marker_size(up['kE'].values, lo, hi, smin, smax),
                                facecolors='none', edgecolors=this_color,
                                linewidths=1.0, alpha=down_alpha*amul)

        # swatch drawn the way the downstream markers are: filled, translucent, no rim.
        # The label carries this PDG's crossing counts as (total/u upstream/d downstream),
        # so the colours key both panels and no separate stats box is needed.
        n_down = int(d_['downstream'].astype(bool).sum())
        n_up = len(d_) - n_down
        legend_handles.append(Line2D([0], [0], linestyle='none', marker='o',
                                     markerfacecolor=this_color, markeredgecolor='none',
                                     alpha=down_alpha, markersize=8,
                                     label="%s (%i/u%i/d%i)"
                                           % (this_label, len(d_), n_up, n_down)))

        # spectrum: log-spaced bins; solid for real trajectories, dashed for start/end
        for real, style in ((True, '-'), (False, '--')):
            k_ = d_[d_['hasTrajectory'].astype(bool) == real]['kE'].values
            k_ = k_[k_ > 0]
            if len(k_):
                counts, _, _ = ax_spec.hist(k_, bins=bins, histtype='step',
                                            color=this_color, linestyle=style,
                                            alpha=1.0 if real else notraj_alpha)
                # track the tallest bar, to decide on a log y axis below
                peak = max(peak, counts.max() if len(counts) else 0)

    # style keys, appended after the particle colours; these mirror how the markers
    # are actually drawn -- filled+translucent downstream, outline-only upstream
    legend_handles.append(Line2D([0], [0], linestyle='none', marker='o', color='grey',
                                 markerfacecolor='grey', markeredgecolor='none',
                                 alpha=down_alpha, markersize=6, label='downstream'))
    legend_handles.append(Line2D([0], [0], linestyle='none', marker='o',
                                 markerfacecolor='none', markeredgecolor='grey',
                                 alpha=down_alpha, markersize=6, label='upstream'))
    legend_handles.append(Line2D([0], [0], linestyle='-', color='grey', label='MCTraj'))
    legend_handles.append(Line2D([0], [0], linestyle='--', color='grey',
                                 alpha=notraj_alpha, label='NoTraj'))
    # both legends hang off ax_face, the right-hand panel, so they sit outboard of the
    # whole figure rather than between the two plots. The title carries the grand total;
    # each particle label carries its own (total/upstream/downstream).
    n_down_all = int(dfx_['downstream'].astype(bool).sum())
    leg = ax_face.legend(handles=legend_handles, loc="upper left",
                         bbox_to_anchor=(1.02, 1.0), fontsize=8,
                         title="%i crossings (u%i/d%i)"
                               % (len(dfx_), len(dfx_) - n_down_all, n_down_all),
                         title_fontsize=8)

    # second legend keying marker area to kE: low, geometric mid, high. The smallest
    # sample is the floor whenever the data reaches it, so label it as an upper bound --
    # every particle at or below kE_floor is drawn at that one size.
    ksamples = [lo, np.sqrt(lo*hi), hi]
    size_handles = [
        Line2D([0], [0], linestyle='none', marker='o',
               markerfacecolor='grey', markeredgecolor='none', alpha=down_alpha,
               # Line2D markersize is a diameter in points; scatter s is an area
               markersize=np.sqrt(_kE_marker_size(k, lo, hi, smin, smax)),
               label=("<= %.3g MeV" % k) if k <= kE_floor else ("%.3g MeV" % k))
        for k in ksamples
    ]
    ax_face.add_artist(leg)   # keep the first legend when the second is attached
    ax_face.legend(handles=size_handles, loc="lower left", bbox_to_anchor=(1.02, 0.0),
                   fontsize=8, labelspacing=1.4,
                   title="kE (floor %.3g MeV)" % kE_floor, title_fontsize=8)

    if geometry:
        draw_Mu2e_faceview(ax_face)

    ax_face.set_xlabel("x [mm]")
    ax_face.set_ylabel("y [mm]")
    ax_face.set_aspect('equal')
    # limits after the patches, so the overlay cannot autoscale the view
    if xlim is not None:
        ax_face.set_xlim(*xlim)
    if ylim is not None:
        ax_face.set_ylim(*ylim)
    ax_face.set_title("position at z = %.1f mm  (area ~ log10 kE)" % z0, fontsize=10)

    ax_spec.set_xscale('log')
    # a log y axis over a handful of counts is mostly empty decades and odd minor ticks,
    # so only use it once the tallest bar makes it worth having
    if peak >= ylog_min:
        ax_spec.set_yscale('log')
    ax_spec.set_xlabel("kE [MeV]")
    ax_spec.set_ylabel("count")
    ax_spec.set_title("kE spectrum at z = %.1f mm" % z0, fontsize=10)


    if title is not None:
        fig.suptitle(title, y=0.98)

    return fig, ax_face, ax_spec, dfx_


def draw_particle_tracer_event(dfe_, title = None, markstart = True):
    # Event display for the TTree written by Offline/STMMC/src/ParticleTracer_module.cc,
    # read in by portROOT2pd_particletracer.PortToDF. dfe_ holds the trajectory entries to
    # draw, one row per SimParticle -- normally one matched (VD) particle plus its
    # genealogy chain back to the primary, as returned by
    # portROOT2pd_particletracer.getGenealogy.
    #
    # Solid line: entry with a stored MCTrajectory (hasTrajectory==True), drawn through
    #             all of its points.
    # Dashed line: entry with no stored MCTrajectory (hasTrajectory==False), so only the
    #             SimParticle start and end positions are available.
    fig = plt.figure(figsize=(15, 6))
    # right=0.88 leaves room for the legend, which is anchored outside the axes below.
    # The figure is saved without bbox_inches='tight' (so every PDF page keeps the same
    # width), and without this margin the legend would be clipped off the page.
    gs = GridSpec(5, 1, figure=fig,
                  wspace=0.05, hspace=0.00, right=0.88)

    ax_top  = fig.add_subplot(gs[0:3]) # top view
    ax_side = fig.add_subplot(gs[3:], sharex=ax_top) # side view

    source_id = []
    legend_handles = []

    for ii in range(len(dfe_)):
        this_entry = dfe_.iloc[ii]
        this_pdgid = this_entry['pdgId']
        try:
            this_color = pdgid.pdgid_color_dict[this_pdgid]
        except:
            this_color = 'yellow'
        # solid for a real trajectory, dashed for start/end positions only
        this_style = '-' if this_entry['hasTrajectory'] else '--'

        if not this_pdgid in source_id:
            source_id.append(this_pdgid)
            try:
                this_label = pdgid.pdgid_dict[this_pdgid]
            except:
                this_label = str(int(this_pdgid))
            legend_dummy = Line2D([0], [0], linewidth=2, linestyle='-', color=this_color,
                                  markerfacecolor=this_color, markersize=0, label=this_label)
            legend_handles.append(legend_dummy)

        x, y, z = this_entry['x'], this_entry['y'], this_entry['z']
        # the matched (VD) particle gets a thicker line than its ancestors
        this_width = 2 if this_entry['matched'] else 1.2

        ax_top.add_line(Line2D(z, x, linewidth=this_width, linestyle=this_style, color=this_color))
        ax_side.add_line(Line2D(z, y, linewidth=this_width, linestyle=this_style, color=this_color))
        if markstart and len(z):
            ax_top.scatter(z[0], x[0], c=this_color, s=5)
            ax_side.scatter(z[0], y[0], c=this_color, s=5)

    # explain the two line styles alongside the particle colours
    legend_handles.append(Line2D([0], [0], linewidth=2, linestyle='-', color='grey',
                                 label='MCTraj'))
    legend_handles.append(Line2D([0], [0], linewidth=2, linestyle='--', color='grey',
                                 label='NoTraj'))

    ax_top.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5))

    draw_Mu2e_topview(ax_top)
    draw_Mu2e_sideview(ax_side)

    if title is not None:
        fig.suptitle(title, y=0.93)

    return fig, ax_top, ax_side