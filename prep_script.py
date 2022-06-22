import flika_JSON_IO as flikaIO
import feature_calculator as fcalc
import numpy as np
import os
import pandas as pd
import pickle


working_dir = r'/home/vivek/Tobias_Group/Piezo1/Cell Overlays for Piezo1 Paper1'

working_file = 'AL_42_2020-02-28-TIRFM_Diff_tdt-mNSPCs_B_21.json'

# This is just a text label for the dataframe that will be generated
experiment_name = 'AL_42_2020-02-28-TIRFM_Diff_tdt-mNSPCs_B_21'

# This is a text label for the output file, it will have an extension added to it.
# {filename}.pkl
output_fileName = 'AL_42_2020-02-28-TIRFM_Diff_tdt-mNSPCs_B_21'

# Minimum number of frames per track
minfrm = 200

filePath = os.path.join(working_dir, working_file)

GB_242_A1_tracks = flikaIO.json_to_pandas(filePath, experiment_name, minfrm)
GB_242_A1_tracks['ID'] = GB_242_A1_tracks['ID'].astype(int)


Rg_lastFrame = fcalc.radGyr_lastFrame(GB_242_A1_tracks)

# Setup the empty column, start it with NaN values
GB_242_A1_tracks['Rg'] = np.nan
# Calc Rg for the last frame (whole trajectory)
# Add it as a new column to the dataframe, every frame has the same final value in it
for trackID in np.arange(0, len(Rg_lastFrame), 1):
    GB_242_A1_tracks.loc[GB_242_A1_tracks.ID == trackID, 'Rg'] = Rg_lastFrame[trackID]
# Output the complete dataframe as a pickled file.
GB_242_A1_tracks.to_pickle(os.path.join(working_dir, output_fileName + '.pkl'))