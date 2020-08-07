# Splits and pickles JSONs

from pathlib import Path
import flika_JSON_IO as flikaIO
import feature_calculator as fcalc
import numpy as np
import pandas as pd
import pickle

def split_Rg_pickle(working_dir, cutoff):
    # Generate the output directory inside the working directory
    Path((Path(working_dir) / f'{cutoff}Frame_Split_JSONs')).mkdir(parents=True, exist_ok=True)
    output_dir = Path(working_dir) / f'{cutoff}Frame_Split_JSONs'
    # Make a list of all the JSONs in the working_dir
    JSON_filelist = Path(working_dir).rglob('*.json')
    # Loop over that file list
    total_files = len(list(Path(working_dir).rglob('*.json')))
    for index, file in enumerate(JSON_filelist):
        print(f'Processing Experiment:   {index+1}   out of   {total_files} ...')
        experiment_name = str(file.stem)
        initial_txy_pts, initial_tracks = flikaIO.open_tracks(file)
        initial_lst, initial_lstnan, initial_trackOrigins = flikaIO.gen_indiv_tracks(cutoff, initial_tracks, initial_txy_pts)
        split_tracks = flikaIO.JSONsplitTracks(initial_txy_pts, initial_tracks, cutoff)
        split_pts = flikaIO.genJSONstyleDict(split_tracks, experiment_name, output_dir)
        txy_pts = np.array(split_pts["txy_pts"])
        tracks = [np.array(track) for track in split_pts["tracks"]]
        lst, lstnan, trackOrigins = flikaIO.gen_indiv_tracks(cutoff - 1, tracks, txy_pts)
        track_array = []
        for trackID, track in enumerate(lstnan):
            track_df = pd.DataFrame(track)
            track_df[len(track_df.columns)] = int(trackID)
            track_array.append(track_df.values)
        combined_tracks = np.vstack(track_array)
        finalTracks_df = pd.DataFrame(combined_tracks)
        finalTracks_df.columns = ['Frame', 'X', 'Y', 'ID']
        finalTracks_df['Exp_Name'] = str(experiment_name)
        finalTracks_df['ID'] = finalTracks_df['ID'].astype(int)
        Rg_lastFrame = fcalc.radGyr_lastFrame(finalTracks_df)
        # Setup the empty column, start it with NaN values
        finalTracks_df['Rg'] = np.nan
        # Calc Rg for the last frame (whole trajectory)
        # Add it as a new column to the dataframe, every frame has the same final value in it
        for trackID in np.arange(0, len(Rg_lastFrame), 1):
            finalTracks_df.loc[finalTracks_df.ID == trackID, 'Rg'] = Rg_lastFrame[trackID]
        # Output the complete dataframe as a pickled file.
        outfile_name = experiment_name + '.pkl'
        outfile = Path(output_dir) / outfile_name
        finalTracks_df.to_pickle(outfile)


if __name__ == '__main__':

    # Directory with all the JSONs to split
    working_dir = r'/home/vivek/Tobias_Group/Piezo1/HaloTag_Gabby/Testing_Plotter/'
    # Frame cutoff for track length
    cutoff = 200

    # The output directory will be generated in the working directory.

    split_Rg_pickle(working_dir, cutoff)

    # Note: there is no return value since this function outputs a compressed file for each JSON