import configparser, subprocess, shutil, json
import sys, string, glob, os, time, re, array

def count_log_filtered_events(log_files, filter_keyword, line_keyword = 'TrigReport', field_index = 3):
    total_count = 0
    for log_path in log_files:
        with open(log_path, 'r') as my_log:
            # Scan through each line in the log file
            in_right_section = False
            for line in my_log:
                # First need to go to "Module summary" section
                if 'TrigReport ---------- Module summary ' in line:
                    in_right_section = True
                if in_right_section and line_keyword in line and filter_keyword in line:
                    # Extract the information from deliminated spaces
                    fields = line.split()
                    if len(fields) > field_index:
                        total_count += int(fields[field_index])
                # Exit the section if we reach a blank line
                if in_right_section and line.strip() == '':
                    in_right_section = False
    return total_count
                    
# For nucap - s1_sim_MuBeamStopVD job (start with MuBeamCat, filter stopped muons, go to VD)
# count the number of stopped muons from the log files
def count_mubeam_stopped_muon():
    log_dir = ['/exp/mu2e/data/users/yongyiwu/grid_logs/MDC2020pmu20pb0s00r0000.s1_sim_MuBeamStopVD/obs_no_final_VD_filter_000/',
               '/exp/mu2e/data/users/yongyiwu/grid_logs/MDC2020pmu20pb0s00r0000.s1_sim_MuBeamStopVD/obs_no_final_VD_filter_001/']
    log_files = []
    for ld in log_dir:
        log_files.extend(glob.glob(os.path.join(ld, 'log.*_001218_*.log')))
    total_stopped_muons = count_log_filtered_events(log_files, 'tgtStopFilter')
    #total_events_on_VD = count_log_filtered_events(log_files, 'filterVirtualDetectorSteps')
    print(f"Total file number: {len(log_files)}")
    print(f"Total stopped muons in MuBeamCat for s1_sim_MuBeamStopVD: {total_stopped_muons}")
    #print(f"Total events on Virtual Detector for s1_sim_MuBeamStopVD: {total_events_on_VD}")
    return total_stopped_muons#, total_events_on_VD

if __name__ == "__main__":
    total_stopped_muons = count_mubeam_stopped_muon()
    print(f"Total stopped muons in MuBeamCat for s1_sim_MuBeamStopVD: {total_stopped_muons}")