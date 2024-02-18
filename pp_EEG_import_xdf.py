# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:54:14 2024

@author: seinj
"""

import pyxdf
import mne
from copy import deepcopy

#import numpy as np

# load xdf file
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
raw = mne.io.RawArray(data, info)

# convert from microvolts to volts
raw.apply_function(lambda x: x*1e-6, picks="eeg")

# visualize (MNE default scaling 20e-6 volts)
#raw.plot()

# # Iterate over streams, find and display markers
# for stream in streams:
#     y = stream['time_series']
#     if isinstance(y, list):
#         # list of strings, draw one vertical line for each marker
#         for timestamp, marker in zip(stream['time_stamps'], y):
#             print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
            

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
         descriptions[i] = 'normal'
     elif 'stimulus:odd' in descriptions[i]:
         descriptions[i] = 'odd'

# Print the updated list
for desc in descriptions:
    print(desc)
    
raw.annotations.append(onsets, [0] * len(onsets), descriptions)


# plot overlay of events 
#raw.plot()

# convert annotations to events
eventsOdd, event_idsOdd = mne.events_from_annotations(raw, event_id={"odd": 2}, regexp='odd', use_rounding=True, chunk_duration=None, verbose=None)                            

# Convert annotations to events for 'stimulus:normal'
eventsNormal, event_idsNormal = mne.events_from_annotations(raw, event_id='auto', regexp='normal', use_rounding=True, chunk_duration=None, verbose=None)

# Create mne.Epochs object for 'stimulus:odd'
epochsOdd = mne.Epochs(raw, eventsOdd, event_id=event_idsOdd, tmin=-0.5, tmax=1.0, baseline=None)

# Create mne.Epochs object for 'stimulus:normal'
epochsNormal = mne.Epochs(raw, eventsNormal, event_id=event_idsNormal, tmin=-0.5, tmax=1.0, baseline=None)

# Combine both mne.Epochs objects
epochsCombined = mne.concatenate_epochs([epochsOdd, epochsNormal])

# plot
epochsCombined.plot()

# avearge signals around each type of events
evokedNorm = epochsCombined["normal"].average()
evokedOdd = epochsCombined["odd"].average()

# covnert averaged evoked responses into a dictionary object
conds = ("normal", "odd")
evks = dict(zip(conds, [evokedNorm, evokedOdd]))

# combine all channels using mean, median, global field potential (gfp) methods
def custom_func(x):
    return x.max(axis=1)

for combine in ("mean", "median", "gfp", custom_func):
    mne.viz.plot_compare_evokeds(evks, picks="eeg", combine=combine)

# visualize the combined evoked responses    
mne.viz.plot_compare_evokeds(
    evks,
    colors=dict(normal=1, odd=0),
    #linestyles=dict(normal="solid", odd="dashed"),
    time_unit="ms",
)    



# bad channel detection
#original_bads = deepcopy(raw.info["bads"])
#raw.info["bads"].append("EEG 050")  # add a single channel
#bad_chan = raw.info["bads"].pop(-1)  # remove the last entry in the list
#raw.info["bads"] = original_bads  # change the whole list at once