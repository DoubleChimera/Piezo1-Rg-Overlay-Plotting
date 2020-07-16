import flika_JSON_IO as flikaIO
import feature_calculator as fcalc
import numpy as np
import os
import pandas as pd
import pickle

working_dir = r'/home/vivek/Documents/Python Programs/piezo1/Piezo1_Rg_Overlays/JSON_example_file/'

working_file = 'GB_tracks_threshold5_242_2020_06_23_HaloTag_Kera_A_1_1.json'

experiment_name = 'GB_tracks_threshold5_242_2020_06_23_HaloTag_Kera_A_1'

output_fileName = 'GB_tracks_threshold5_242_2020_06_23_HaloTag_Kera_A_1'

minfrm = 200

filePath = os.path.join(working_dir, working_file)

GB_242_A1_tracks = flikaIO.json_to_pandas(filePath, experiment_name, minfrm)
GB_242_A1_tracks['ID'] = GB_242_A1_tracks['ID'].astype(int)


Rg_lastFrame = fcalc.radGyr_lastFrame(GB_242_A1_tracks)

# Setup the empty column, start it with NaN values
GB_242_A1_tracks['Rg'] = np.nan
# Calc Rg for the last frame (whole trajectory)
# Add it as a new column to the dataframe
for trackID in np.arange(0, len(Rg_lastFrame), 1):
    GB_242_A1_tracks.loc[GB_242_A1_tracks.ID == trackID, 'Rg'] = Rg_lastFrame[trackID]
# Output the complete dataframe as a pickled file.
GB_242_A1_tracks.to_pickle(os.path.join(working_dir, output_fileName + '.pkl'))