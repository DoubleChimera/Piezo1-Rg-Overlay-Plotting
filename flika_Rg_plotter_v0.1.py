# Takes in a directory of .pkl (pickled) files. These are JSONs that have been preprocessed by a minimum frame cutoff.
# Usually the minimum frame cutoff is 200 frames, but this can be varied.
# This script processes those files to produce:
#    1. Histograms of Rg Values
#    2. Kernel Density of Rg Values, and Mean Kernel Density
#    3. Box-Plots
# Really gotta work on this description at some point, but you get the idea.

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns


class JSONplotter:
    def __init__(self, working_dir, output_dir, autoSavePlot):
        self.working_dir = working_dir
        self.output_dir = output_dir
        all_files = glob.glob(working_dir + "/*.pkl")
        df_list = []
        for filename in all_files:
            df = pd.read_pickle(filename)
            df_list.append(df)
        Rg_dataframe = pd.concat(df_list)
        pd.set_option("display.precision", 16)
        Rg_dataframe['Frame'] = Rg_dataframe['Frame'].astype(int)
        self.videoList = Rg_dataframe.Exp_Name.unique()
        self.Rg_dataframe = Rg_dataframe
        # things we do to the data in general to prep it for any of the functions

    def histogram(self, autoSavePlot=False):
        # Get the min and max Rg values
        Rg_MaxVal = math.ceil(max(self.Rg_dataframe['Rg']))
        # Set the width of each bin
        bin_width = 0.5
        # Calculate bin edges
        binList = list(np.arange(0, Rg_MaxVal + 1, bin_width))
        for eachVideo in self.videoList:
            total_tracks = len(self.Rg_dataframe.loc[((self.Rg_dataframe.Exp_Name == eachVideo) & (self.Rg_dataframe.Frame == 0))]['Rg'])
            # plot a histogram of the Rg values for each experiment
            fig, axes = plt.subplots(1, 1, figsize=(6, 4))
            plt.hist(self.Rg_dataframe.loc[((self.Rg_dataframe.Exp_Name == eachVideo) & (self.Rg_dataframe.Frame == 0))]['Rg'],  edgecolor='black', linewidth=1, density=False, label=f'Total Tracks > 200 Frames: {total_tracks}', bins=binList)
            plt.xlim([0, 20])
            plt.ylim([0, 1100])
            plt.title(f'{eachVideo} Rg Distribution')
            fig.canvas.set_window_title(f'{eachVideo}_HaloTag_Rg')
            plt.legend()
            if autoSavePlot:
                plt.savefig(self.output_dir + f'{eachVideo}_HaloTag_Rg.png', bbox_inches='tight')
            plt.show()

    def kernel_density(self, plotMean=True, autoSavePlot=False, plotTitle='', xRange=[0, 1.0], yRange=[-3, 20]):
        SummedRg = []
        totalTracks = 0
        for index, eachVideo in enumerate(self.videoList):
            lineColor = 'red'
            totalTracks += len(self.Rg_dataframe.loc[((self.Rg_dataframe.Exp_Name == eachVideo) & (self.Rg_dataframe.Frame == 0))]['Rg'])
            print(f'Video Number: {index}  Tracks so far: {totalTracks}')
            SummedRg += list(self.Rg_dataframe.loc[((self.Rg_dataframe.Exp_Name == eachVideo) & (self.Rg_dataframe.Frame == 0))]['Rg'])
            sns.distplot(self.Rg_dataframe.loc[((self.Rg_dataframe.Exp_Name == eachVideo) & (self.Rg_dataframe.Frame == 0))]['Rg'], hist=False, kde=True, kde_kws={'linewidth': 2}, color=lineColor)
        if plotMean:
            sns.distplot(SummedRg, hist=False, kde=True, kde_kws={'linewidth': 2}, color='k', label=f'Track Count: {totalTracks}')
        plot_figure_name = f'HaloTag_Rg_Density'
        plt.ylim(xRange)
        plt.xlim(yRange)
        plt.legend(prop={'size': 12})
        plt.title(f'{plotTitle}HaloTag_Rg_Density', fontsize=18)
        plt.xlabel('Radius of Gyration (Rg)', fontsize=16)
        plt.ylabel('Density', fontsize=16)

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=1.0)
        plt.tight_layout()
        if autoSavePlot:
                plt.savefig(self.output_dir + f'{plotTitle}HaloTag_Rg_Density' + '.png', bbox_inches='tight')
        plt.show()


def box_plot(exp_list:list, exp_labels:list, autoSavePlot=False, plotTitle='Categorical_'):
    exp_labels = pd.Series(exp_labels)
    assert len(exp_list) == len(exp_labels), "BoxPlot - VALUE_ERROR: Length of experiment list and experiment labels are not equal."
    cumulative_DF = pd.DataFrame([])
    for index, eachExp in enumerate(exp_list):
        # Add the label as a column
        eachExp.Rg_dataframe['Boxplot_Label'] = exp_labels[index]
        cumulative_DF = pd.concat([cumulative_DF, eachExp.Rg_dataframe])
    cumulative_DF = cumulative_DF.loc[(cumulative_DF.Frame == 0)]
    tempPlot = sns.boxplot(x='Boxplot_Label', y='Rg', data=cumulative_DF)
    tempPlot.set_xlabel('Experimental Categories', fontsize=16)
    tempPlot.set_ylabel('Rg', fontsize=16)
    plt.title(f'{plotTitle}HaloTag_Rg Boxplot', fontsize=18)
    plt.tight_layout()
    if autoSavePlot:
        outLabel = ""
        for eachLabel in exp_labels:
            outLabel += " " + str(eachLabel)
        plt.savefig(exp_list[0].output_dir + f'BoxPlot_Rg{outLabel}' + '.png', bbox_inches='tight')
    plt.show()

# ! This is the part to edit:
if __name__ == '__main__':
    # General inputs
    # The directories with the pickled JSON files in it.
    #       --Make as many as you need.
    #       --Here I've made two, one Control the other Yoda1
    # Control data locations
# 1. Control
    data1_pickled_JSON_files_directory = r'/home/vivek/Tobias_Group/Piezo1/HaloTag_Gabby/Testing_Plotter/'
    data1_plot_output_directory = pickled_JSON_files_directory
    # yoda1 data locations
# 2. yoda1
    data2_pickled_JSON_files_directory = r'/home/vivek/Tobias_Group/Piezo1/HaloTag_Gabby/Testing_Plotter/'
    data2_plot_output_directory = pickled_JSON_files_directory
# 3. Got more data? Use the same template as above, just rename data2 to data3

    # Do you want to automatically save plots in the output directory?
    # NOTE: since boxplots use more than one data set, the plot will be saved in the FIRST output folder (data1)
    autoSavePlot = False

    # Plot specific inputs
    # Kernel Density, show mean on plot or not
    show_kernel_mean_plot = True

    # Examples of plotting data
    # test1 is control
    # test2 is yoda1
    # First load the data
    test1 = JSONplotter(data1_pickled_JSON_files_directory, data1_plot_output_directory, autoSavePlot)
    test2 = JSONplotter(data2_pickled_JSON_files_directory, data2_plot_output_directory, autoSavePlot)

    # Then choose the plotting operations and add any missing arguments you want to add in.
    # test1.histogram(autoSavePlot)
    # test1.kernel_density(show_kernel_mean_plot, autoSavePlot, 'Control_')

    # For boxplot we need the categorical labels for each type of condition
    # exp_labels = ['Control', 'yoda1']
    # box_plot([test1, test2], exp_labels, autoSavePlot, 'Categorical ')
