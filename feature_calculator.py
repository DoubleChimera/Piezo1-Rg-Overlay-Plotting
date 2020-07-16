import numpy as np
import pandas as pd

def radGyr_tensor(points):
    """
    Calculate the gyration tensor
    points : numpy array of shape N x 2
    (will work for N x X ; where X is any integer )
    """
    center = points.mean(0)
    #normalized points
    normed_points = points - center[None, :]
    return np.einsum('im,in->mn', normed_points, normed_points)/len(points)

def radGyr_allFrames(tracks_df):
    """Takes a dataframe with columns 'Frame', 'X', 'Y', 'ID' 'Exp_Name' and returns a list of Rg values

    Args:
        tracks_df ([pandas dataframe]): [dataframe with ['Frame, 'X', 'Y', 'ID', 'Exp_Name']]

    Returns:
        [List]: [List of Rg values for each trajectory, calculated up to each frame]
    """
    # Get number of tracks
    trackID_list = list(tracks_df.ID.unique())
    # Loop over each track, and over each frame of each track
    Rg_list = []
    for index, trackID in enumerate(trackID_list):
        track_length = len(tracks_df.loc[tracks_df['ID'] == trackID])
        for frame in range(0, track_length, 1):
            mask = ((tracks_df['ID'] == trackID) & (tracks_df['Frame'] <= frame))
            points_array = np.array(tracks_df.loc[mask][['X', 'Y']].dropna())
            temp_radGyr_tens = radGyr_tensor(points_array)
            Rg_list.append(np.sqrt(np.sum(temp_radGyr_tens.diagonal())))
    return Rg_list

def radGyr_oneFrame(tracks_df, frameValue):
    """Takes a dataframe and calculate Rg up to a specific frame

    Args:
        tracks_df ([pandas dataframe]): [dataframe with ['Frame, 'X', 'Y', 'ID', 'Exp_Name']]
        frameValue ([int]): [frame number indexed from 0, on which to calc Rg]
                            [A frameValue of -1 will calculate the last frame, i.e. whole trajectory]

    Returns:
        [List]: [List of Rg values for each trajectory, upto each frame]
    """
    # Get number of tracks
    trackID_list = list(tracks_df.ID.unique())
    # Loop over each track, calc Rg out to the chosen frame
    Rg_frame = frameValue
    Rg_list_oneFrame = []
    for index, trackID in enumerate(trackID_list):
        track_length = len(tracks_df.loc[tracks_df['ID'] == trackID])
        mask = ((tracks_df['ID'] == trackID) & (tracks_df['Frame'] <= Rg_frame))
        points_array = np.array(tracks_df.loc[mask][['X', 'Y']].dropna())
        temp_radGyr_tens = radGyr_tensor(points_array)
        Rg_list_oneFrame.append(np.sqrt(np.sum(temp_radGyr_tens.diagonal())))
    return Rg_list_oneFrame

def radGyr_lastFrame(tracks_df):
    """Calculates Rg for the last frame of a trajectory, arguably this is the 'real' Rg value

    Args:
        tracks_df ([pandas dataframe]): [dataframe with columns ['Frame', 'X', 'Y', 'ID', 'Exp_Name']]

    Returns:
        [List]: [List of Rg values calculated up to last frame in each trajectory]
    """
    # Get number of tracks
    trackID_list = list(tracks_df.ID.unique())
    Rg_list_lastFrame = []
    for index, trackID in enumerate(trackID_list):
        track_length = len(tracks_df.loc[tracks_df['ID'] == trackID])
        mask = ((tracks_df['ID'] == trackID) & (tracks_df['Frame'] <= (track_length - 1)))
        points_array = np.array(tracks_df.loc[mask][['X', 'Y']].dropna())
        temp_radGyr_tens = radGyr_tensor(points_array)
        Rg_list_lastFrame.append(np.sqrt(np.sum(temp_radGyr_tens.diagonal())))
    return Rg_list_lastFrame