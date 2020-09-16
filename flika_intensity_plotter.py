import numpy as np
import pandas as pd
import codecs, json
from pathlib import Path
import h5py, pickle
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

#################### START USER INPUTS ####################
working_directory = r'/home/vivek/Tobias_Group/Piezo1/HaloTag_Gabby/Track_Intensities_troubleshooting'
working_file = r'AL_55_2020-06-29-TIRFM_Diff_tdt-MEFs_B_4_trackintensities_v2.json'
frame_cutoff = 200
####################

#################### END OF USER INPUTS ####################
# Load the JSON as a dataframe with frame cutoff limits
filename = Path(working_directory) / working_file
obj_text = codecs.open(filename, "r", encoding="utf-8").read()
pts = json.loads(obj_text)
txy_pts = np.array(pts["txy_pts"])
tracks = [np.array(track) for track in pts["tracks"]]
txy_intensities = np.array(pts["txy_intensities"])
txyi_pts = pd.DataFrame(columns=['TrackID', 'Frame', 'X', 'Y', 'Intensity'])
for trackIndex, _ in enumerate(tracks):
    track = tracks[trackIndex]
    pts = txy_pts[track, :]
    if len(pts) >= frame_cutoff:
        intensity = txy_intensities[track]
        indivTrack = pd.DataFrame(pts, columns=['Frame', 'X', 'Y'])
        indivTrack['Intensity'] = intensity
        indivTrack['TrackID'] = trackIndex
        txyi_pts = pd.concat([txyi_pts, indivTrack])
# Adjust Frame values to be Time
txyi_pts.Frame *= 0.100
txyi_pts.rename(columns={'Frame': 'Time'}, inplace=True)
# Define the plotting function
def plot_v2(data, minMax=[]):
    # Make two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=([10, 5]))
    # Create the intensity line plot
    g = sns.lineplot(
        x="Time", y="Intensity",
        data=data, linewidth=2, label=f'Track: {data.TrackID[0]}\nLength: {len(data)}',
        ax=axes[0])
    if minMax != []:
        axes[0].set_ylim(minMax)
    axes[0].set_xlim([-5, 65])
    axes[0].set_title('Intensity over Time')
    # Create the trajectory plot to localize the track in the video
    g1 = sns.lineplot(
        x="X", y="Y",
        data=data, linewidth=5, label=f'Track: {data.TrackID[0]}\nLength: {len(data)}',
        ax=axes[1])
    axes[1].set_xlim([0, 1023])
    axes[1].set_ylim([1023, 0])
    axes[1].set_title('Position in Video')
    return plt.gcf(), axes
# Loop over the stuff you want to plot
trackIDList = txyi_pts.TrackID.unique()
yMaxVal = int(txyi_pts.Intensity.max()*1.10)
yMinVal = int(txyi_pts.Intensity.min()*0.90)
yaxisMinMax = [yMinVal, yMaxVal]
for trackCount, eachTrackID in enumerate(trackIDList):
    indivTrack = txyi_pts.loc[txyi_pts['TrackID'] == eachTrackID]
    plot_v2(indivTrack, yaxisMinMax)
    plt.suptitle(f'Trajectory {trackCount+1} out of {len(trackIDList)}')
    plt.subplots_adjust(top=0.85)
    plt.tight_layout(pad=3.0)

    plt.show()