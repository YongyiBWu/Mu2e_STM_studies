import numpy as np

XYmaxbincount = 7500
spectramaxbincount = 60000

VD101_X =  -3904.000 # in hall coord
VD101_Y =      0.416
VD101_Z =  40185.390
VD101_R =  100.

hole1_X = -40.64 # w.r.t. beam center
hole1_Y =   0.
hole1_R = np.sqrt(150./np.pi)
hole1_r = np.sqrt(50./np.pi)

hole2_X =  40.64
hole2_Y =   0.
hole2_R = np.sqrt(150./np.pi)
hole2_r = np.sqrt(50./np.pi)

# following info from docdb-2066, 26586. Only giving a rough sketch
production_target_X = 3904.
production_target_Y = 0.
production_target_Z = -6164.5
ps_cryo_start_Z = -7943.
ps_cryo_end_Z   = -7943.+4500.
ps_cryo_len = ps_cryo_end_Z-ps_cryo_start_Z
ps_cryo_OR = 1300.
ps_cryo_IR = 750.

stopping_target_X = -3904.
stopping_target_Y = 0.
stopping_target_start_Z = 5471.
stopping_target_end_Z   = 6271.
stopping_target_len = stopping_target_end_Z-stopping_target_start_Z
stopping_target_OR = 75.
ds_cryo_start_Z = 3239.
ds_cryo_end_Z   = 13989.
ds_cryo_len = ds_cryo_end_Z-ds_cryo_start_Z
ds_cryo_OR = 1335.
ds_cryo_IR = 950.
trk_start_Z = 8410.
trk_end_Z   = 11660.
trl_len = trk_end_Z-trk_start_Z
trk_OR = 850.
calo_start_Z = 11740.
calo_end_Z   = 13190.
calo_len = calo_end_Z-calo_start_Z
calo_OR = 650.
IFB_start_Z = 13190.
IFB_end_Z   = 17316.
IFB_len = IFB_end_Z-IFB_start_Z
IFB_OR = 950.
