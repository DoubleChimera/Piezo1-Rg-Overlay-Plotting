# Takes in a directory of .pkl (pickled) files. These are JSONs that have been preprocessed by a minimum frame cutoff.
# Usually the minimum frame cutoff is 200 frames, but this can be varied.
# This script processes those files to produce:
#    1. Histograms of Rg Values
#    2. Kernel Density of Rg Values, and Mean Kernel Density
#    3. Cumulative Distribution of Rg Values, and Mean Cumulative Distribution
#    4. Box-Plots

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

output_dir = r'/home/vivek/Tobias_Group/Piezo1/HaloTag_Gabby/Track_Stats_July31/4uM_pickled_files/Control/'
autoSavePlot = False

working_dir = r'/home/vivek/Tobias_Group/Piezo1/HaloTag_Gabby/Track_Stats_July31/4uM_pickled_files/Control/'

all_files = glob.glob(working_dir + "/*.pkl")
df_list = []
for filename in all_files:
    df = pd.read_pickle(filename)
    df_list.append(df)

# Rg_PickleFile_list = [  f'yoda1_2uM_controls/Ctrl_242_WT_HaloKera_A1/GB_242_Halo_A_1.pkl',
#                         f'yoda1_2uM_controls/Ctrl_242_WT_HaloKera_A3/GB_242_Halo_A_3.pkl',
#                         f'yoda1_2uM_controls/Ctrl_242_WT_HaloKera_A4/GB_242_Halo_A_4.pkl',
#                         f'yoda1_2uM_treated/242_WT_HaloKera_A_Y1_1/GB_242_Halo_A_Y1_1.pkl',
#                         f'yoda1_2uM_treated/242_WT_HaloKera_A_Y1_3/GB_242_Halo_A_Y1_3.pkl',
#                         f'yoda1_2uM_treated/242_WT_HaloKera_A_Y1_4/GB_242_Halo_A_Y1_4.pkl']

# df_list = []
# for pickled_fileName in Rg_PickleFile_list:
#     df = pd.read_pickle(os.path.join(working_dir, pickled_fileName))
#     df_list.append(df)

Rg_dataframe = pd.concat(df_list)

pd.set_option("display.precision", 16)

Rg_dataframe['Frame'] = Rg_dataframe['Frame'].astype(int)

print(Rg_dataframe.dtypes)

# Get the min and max Rg values
Rg_MaxVal = math.ceil(max(Rg_dataframe['Rg']))
print(Rg_MaxVal)
# Set the width of each bin
bin_width = 0.5
# Calculate bin edges
binList = list(np.arange(0, Rg_MaxVal + 1, bin_width))

videoList = Rg_dataframe.Exp_Name.unique()

# for eachVideo in videoList:

#     total_tracks = len(Rg_dataframe.loc[((Rg_dataframe.Exp_Name == eachVideo) & (Rg_dataframe.Frame == 0))]['Rg'])

#     # plot a histogram of the Rg values for each experiment
#     fig, axes = plt.subplots(1, 1, figsize=(6, 4))
#     plt.hist(Rg_dataframe.loc[((Rg_dataframe.Exp_Name == eachVideo) & (Rg_dataframe.Frame == 0))]['Rg'],  edgecolor='black', linewidth=1, density=False, label=f'Total Tracks > 200 Frames: {total_tracks}', bins=binList)
#     plt.xlim([0, 20])
#     plt.ylim([0, 1100])
#     plt.title(f'{eachVideo} Rg Distribution')
#     fig.canvas.set_window_title(f'{eachVideo}_HaloTag_Rg')
#     plt.legend()
#     if autoSavePlot:
#         plt.savefig(output_dir + f'{eachVideo}_HaloTag_Rg.png', bbox_inches='tight')
#     plt.show()

# # ! THIS WILL NEED YOUR ATTENTION EACH TIME
# This is the plot
SummedRg = []
totalTracks = 0
for index, eachVideo in enumerate(videoList):
    lineColor = 'red'
    totalTracks += len(Rg_dataframe.loc[((Rg_dataframe.Exp_Name == eachVideo) & (Rg_dataframe.Frame == 0))]['Rg'])
    print(f'Video Number: {index}  Tracks so far: {totalTracks}')
    SummedRg += list(Rg_dataframe.loc[((Rg_dataframe.Exp_Name == eachVideo) & (Rg_dataframe.Frame == 0))]['Rg'])
    sns.distplot(Rg_dataframe.loc[((Rg_dataframe.Exp_Name == eachVideo) & (Rg_dataframe.Frame == 0))]['Rg'], hist=False, kde=True, kde_kws={'linewidth': 2}, color=lineColor)


sns.distplot(SummedRg, hist=False, kde=True, kde_kws={'linewidth': 2}, color='k', label=f'Track Count: {totalTracks}')

plot_figure_name = f'A_and_A_Y1_HaloTag_Rg_Density'

plt.ylim([0, 1.0])
plt.xlim([-3, 20])
plt.legend(prop={'size': 12})
plt.title(f'Control GB-221-230-232 33 Videos', fontsize=18)
plt.xlabel('Radius of Gyration (Rg)', fontsize=16)
plt.ylabel('Density', fontsize=16)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=1.0)
plt.tight_layout()
if autoSavePlot:
        plt.savefig(output_dir + f'Control_GB-221-230-232_33Videos' + '.png', bbox_inches='tight')
plt.show()