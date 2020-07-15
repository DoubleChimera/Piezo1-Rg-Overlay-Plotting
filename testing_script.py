import flika_JSON_IO as flikaIO
import feature_calculator as fcalc
import numpy as np
import os
import pandas as pd
import pickle

working_dir = r'/home/vivek/Documents/Python Programs/piezo1/Piezo1_Rg_Overlays/JSON_example_file/'

working_file = 'GB_221_Halo_B2.json'

experiment_name = 'GB_221_Halo_B2'

minfrm = 200

filePath = os.path.join(working_dir, working_file)

GB_221_B2_tracks = flikaIO.json_to_pandas(filePath, experiment_name, minfrm)
GB_221_B2_tracks['ID'] = GB_221_B2_tracks['ID'].astype(int)


Rg_lastFrame = fcalc.radGyr_lastFrame(GB_221_B2_tracks)

# Setup the empty column, start it with NaN values
GB_221_B2_tracks['Rg'] = np.nan
# Calc Rg for the last frame (whole trajectory)
# Add it as a new column to the dataframe
for trackID in np.arange(0, len(Rg_lastFrame), 1):
    GB_221_B2_tracks.loc[GB_221_B2_tracks.ID == trackID, 'Rg'] = Rg_lastFrame[trackID]

GB_221_B2_tracks.to_pickle(os.path.join(working_dir, 'GB_221_Halo_B2_Rg.pkl'))