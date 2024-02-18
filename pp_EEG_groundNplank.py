# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:54:14 2024

@author: seinj
"""

import pyxdf
import mne

### import Ground

# load xdf file
fname = "C:/Users/seinj/Teaching/Recordings_2024_EEG/source-data/pb_07_oddball_ground.xdf"
streams, header = pyxdf.load_xdf(fname)

# assign EEG data to a data variable
data = streams[0]["time_series"].T
data = data[:64]  # subselect
assert data.shape[0] == 64

# construct info 
sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(64, sfreq, ["eeg"]*64)

# construct raw data in MNE python format
rawGround = mne.io.RawArray(data, info)

# convert from microvolts to volts
rawGround.apply_function(lambda x: x*1e-6, picks="eeg")

# event stream definition
eventStream = streams[1]

# first time stamp in eeg stream
first_samp = streams[0]["time_stamps"][0]
print('first time stamp correction {}'.format(first_samp))

# event onsets
onsets = eventStream["time_stamps"] - first_samp

# create descriptions            
descriptions = [item for sub in eventStream["time_series"] for item in sub]
for i in range(len(descriptions)):
     if 'stimulus:normal' in descriptions[i]:
         descriptions[i] = 'normalG'
     elif 'stimulus:odd' in descriptions[i]:
         descriptions[i] = 'oddG'

    
rawGround.annotations.append(onsets, [0] * len(onsets), descriptions)

### import plank

fname = "C:/Users/seinj/Teaching/Recordings_2024_EEG/source-data/pb_07_oddball_plank.xdf"
streams, header = pyxdf.load_xdf(fname)

# assign EEG data to a data variable
data = streams[0]["time_series"].T
data = data[:64]  # subselect
assert data.shape[0] == 64

# construct info 
sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(64, sfreq, ["eeg"]*64)

# construct raw data in MNE python format
rawPlank = mne.io.RawArray(data, info)

# convert from microvolts to volts
rawPlank.apply_function(lambda x: x*1e-6, picks="eeg")

# event stream definition
eventStream = streams[1]

# first time stamp in eeg stream
first_samp = streams[0]["time_stamps"][0]
print('first time stamp correction {}'.format(first_samp))

# event onsets
onsets = eventStream["time_stamps"] - first_samp

# create descriptions            
descriptions = [item for sub in eventStream["time_series"] for item in sub]
for i in range(len(descriptions)):
     if 'stimulus:normal' in descriptions[i]:
         descriptions[i] = 'normalP'
     elif 'stimulus:odd' in descriptions[i]:
         descriptions[i] = 'oddP'
    
rawPlank.annotations.append(onsets, [0] * len(onsets), descriptions)

####

# Convert annotations to events for 'stimulus:normal'
eventsNormalG, eventIdsNormalG = mne.events_from_annotations(rawGround, event_id={"normalG": 1}, regexp='normalG', use_rounding=True, chunk_duration=None, verbose=None)
eventsNormalP, eventIdsNormalP = mne.events_from_annotations(rawPlank, event_id={"normalP": 2}, regexp='normalP', use_rounding=True, chunk_duration=None, verbose=None)

# convert annotations to events
eventsOddG, eventIdsOddG = mne.events_from_annotations(rawGround, event_id={"oddG": 3}, regexp='oddG', use_rounding=True, chunk_duration=None, verbose=None)                            
eventsOddP, eventIdsOddP = mne.events_from_annotations(rawPlank, event_id={"oddP": 4}, regexp='oddP', use_rounding=True, chunk_duration=None, verbose=None)                            

# Create mne.Epochs object for 'stimulus:normal'
epochsNormalG = mne.Epochs(rawGround, eventsNormalG, event_id=eventIdsNormalG, tmin=-0.3, tmax=0.5, baseline=None)
epochsNormalP = mne.Epochs(rawPlank, eventsNormalP, event_id=eventIdsNormalP, tmin=-0.3, tmax=0.5, baseline=None)

# Create mne.Epochs object for 'stimulus:odd'
epochsOddG = mne.Epochs(rawGround, eventsOddG, event_id=eventIdsOddG, tmin=-0.3, tmax=0.5, baseline=None)
epochsOddP = mne.Epochs(rawPlank, eventsOddP, event_id=eventIdsOddP, tmin=-0.3, tmax=0.5, baseline=None)

# Combine all four mne.Epochs objects
epochsCombined = mne.concatenate_epochs([epochsNormalG, epochsNormalP, epochsOddG, epochsOddP])

# avearge signals around each type of events
evokedNormG     = epochsCombined["normalG"].average()
evokedNormP     = epochsCombined["normalP"].average()
evokedOddG      = epochsCombined["oddG"].average()
evokedOddP      = epochsCombined["oddP"].average()

# covnert averaged evoked responses into a dictionary object
conds = ("normalG", "normalP", "oddG", "oddP")
evks = dict(zip(conds, [evokedNormG, evokedNormP, evokedOddG, evokedOddP]))

# combine all channels using mean, median, global field potential (gfp) methods
def custom_func(x):
     return x.max(axis=1)

for combine in ("mean", "median", "gfp", custom_func):
     mne.viz.plot_compare_evokeds(evks, picks="eeg")#, combine=combine)

# visualize the combined evoked responses    
mne.viz.plot_compare_evokeds(
    evks,
    picks = ['20'],
    colors=dict(oddG=1, oddP=1, normalG=0, normalP=0),
    linestyles=dict(normalG="solid",oddG="solid", oddP="dashed", normalP="dashed"),
    time_unit="ms",
)    