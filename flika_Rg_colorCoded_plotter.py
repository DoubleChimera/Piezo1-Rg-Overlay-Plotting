# Author: Vivek Tyagi   July 2020  Tobias Group UCI
# This script overlays trajectories onto a DIC .tiff image
# and color codes tracks based on their Rg and Df values (previously determined)
from PIL import Image, ImageEnhance, ImageMath, ImageOps
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pickle
import os.path
import glob
import re

# * START OF USER INPUTS
# Working directory, where all the files are located
working_dir = r'/home/vivek/Tobias_Group/Piezo1/HaloTag_Gabby/Track_Stats_July9/yoda1_greyscale_overlays/yoda1_2uM/yoda1_2uM_treated/242_WT_HaloKera_A_Y1_1/'

# Video name for plot title and output plot
analyzedFileName = 'GB_242_Halo_A_Y1_1_G200F_colored'

# DIC .tiff file name
DIC_tif_name = '242_2020_06_23_HaloTag_Kera_A+Y1_DIC_MMStack_Pos0.ome.tif'

# Which image in the DIC .tiff stack, first image is indexed as 0, second is 1, etc.
stackPosition = 0

# Pickled Dataframe to input
pickled_file_name = 'GB_242_Halo_A_Y1_1_G200F.pkl'

GB_242_A1_df = pd.read_pickle(os.path.join(working_dir, pickled_file_name))

# Automatically name and save the plot in the working directory
autoSavePlot = True

# * END OF USER INPUTS

#* ------------------------------------------------------------------------------------------------------------------------------------------------------
#* Processing Code below, shouldn't be any reason to edit
#* Except for the color bar related values
#* ------------------------------------------------------------------------------------------------------------------------------------------------------

# Set the precision of float values to 16 decimals
pd.set_option('display.precision',16)

regex = re.compile(r'\d+')

# Background DIC .tiff image with appropriate slice selected
bg_DIC_img = working_dir + DIC_tif_name
# Load the tiff stack
imStack = Image.open(bg_DIC_img)
imStack.load()

# Extract the dic frame
imStack.seek(stackPosition)
imDicFrame = imStack.copy()

# ! I don't like this fix.
# Scale the pixels by 256 and convert it to 'RGBA' format
# imDicScaled = imDicFrame.point(lambda i:i*(1./256)).convert('RGBA')
enhanced_im = imDicFrame

# imDicScaled_autoContrast = ImageOps.autocontrast(imDicFrame, cutoff = 2, ignore = 2)
# enhanced_im = ImageEnhance.Contrast(imDicScaled).enhance(0.50)
# enhanced_im = ImageEnhance.Brightness(enhanced_im).enhance(2000.0)

img_dic = enhanced_im
# img_dic = np.flipud(enhanced_im)

Rg_list = list(GB_242_A1_df.Rg.unique())

# RgDf_input = RgDf_input.astype(float)

Rg_max = max(Rg_list)
Rg_min = min(Rg_list)
cm = plt.get_cmap('jet')

# Autoscale the colorbar for the plotted tracks
cNorm  = colors.Normalize(vmin=math.floor(Rg_min), vmax=math.ceil(Rg_max))
# Manually scale the colorbar for the plotted tracks
cNorm  = colors.Normalize(vmin=0, vmax=15)

scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)


# Set up the plot with the background image
# Plot tracks on TIRFM image
subplot_kw = dict(xlim=(0, 1024), ylim=(1024, 0), autoscale_on=False)
fig, axes = plt.subplots(1, 1, subplot_kw=subplot_kw, figsize=(10, 7))
imgplot1 = axes.imshow(img_dic, cmap=plt.get_cmap('gray'))

# print rg of a track index
# print(float(RgDf_input[RgDf_input[0] == '498577'][1]))

# Plot the tracks
trackID_list = list(GB_242_A1_df.ID.unique())
for trackID in trackID_list:
    track_Rg = Rg_list[trackID]
    indivTrack_df = GB_242_A1_df.loc[GB_242_A1_df['ID'] == trackID, ['X', 'Y']]
    colorValue = scalarMap.to_rgba(track_Rg)
    plt.plot(indivTrack_df.X, indivTrack_df.Y, color=colorValue, linewidth=1)

totalTracks = len(Rg_list)
plt.suptitle(f'{analyzedFileName}')
plt.title(f'Total Tracks: {totalTracks}')

cbax = fig.add_axes([0.85, 0.12, 0.05, 0.78])
cb = mpl.colorbar.ColorbarBase(cbax, cmap=cm, norm=cNorm, orientation='vertical')
cb.set_label("Rg Values", rotation=270, labelpad=15, fontsize=14)

# Autoscale the colorbar
colorbar_Side = np.arange(math.floor(Rg_min), math.ceil(Rg_max)+1, 1)
# Manually scale the colorbar
# Note: np.arange(0, 16, 1) <-- Start: 0, End (but not including): 16, Step: 1
# So the above goes from 0 to 15 in steps of 1
# Ex. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
colorbar_Side = np.arange(0, 16, 1)

print(colorbar_Side)
cb.set_ticks(colorbar_Side)

# cb.set_ticks([0, 2, 9])
    # for index, quadrant in enumerate(quad_indexList):
#         if (trackIndex in quadrant):
#             trackData = np.loadtxt(filePath, delimiter=' ', skiprows=1, dtype=np.float32)
#             df = pd.DataFrame(trackData)
#             plt.plot(df[1], df[2], c=)

fig.tight_layout(rect=[0, 0, 1, .95])
if autoSavePlot:
    plt.savefig(working_dir + analyzedFileName + '_RgDf-Overlay.png', bbox_inches='tight')
plt.show()