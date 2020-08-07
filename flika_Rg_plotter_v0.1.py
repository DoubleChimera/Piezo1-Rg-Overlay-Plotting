# Takes in a directory of .pkl (pickled) files. These are JSONs that have been preprocessed by a minimum frame cutoff.
# Usually the minimum frame cutoff is 200 frames, but this can be varied.
# This script processes those files to produce:
#    1. Histograms of Rg Values
#    2. Kernel Density of Rg Values, and Mean Kernel Density
#    3. Box-Plots
#    4. Cumulative Distribution Function
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
        exp_length = len(eachExp.Rg_dataframe.loc[(eachExp.Rg_dataframe['Frame'] == 0)])
        eachExp.Rg_dataframe['Exp_Category'] = str(exp_labels[index]) + f'\nTracks: {exp_length}'
        cumulative_DF = pd.concat([cumulative_DF, eachExp.Rg_dataframe])
        # print(len(cumulative_DF.loc[cumulative_DF['Exp_Category'] == 'yoda1']))
    cumulative_DF = cumulative_DF.loc[(cumulative_DF.Frame == 0)]
    tempPlot = sns.boxplot(x='Exp_Category', y='Rg', data=cumulative_DF)
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

def cumulative_distribution(exp_list:list, exp_labels:list, autoSavePlot=False, plotTitle='Categorical_', binWidth=0.1):
    exp_labels = pd.Series(exp_labels)
    assert len(exp_list) == len(exp_labels), "Cumul Distrib - VALUE_ERROR: Length of experiment list and experiment labels are not equal."
    cumulative_DF = pd.DataFrame([])
    fig, axes = plt.subplots()
    for index, eachExp in enumerate(exp_list):
        # Add the label as a column
        exp_length = len(eachExp.Rg_dataframe.loc[(eachExp.Rg_dataframe['Frame'] == 0)])
        eachExp.Rg_dataframe['Exp_Category'] = str(exp_labels[index]) + f' Tracks: {exp_length}'
        mask = (eachExp.Rg_dataframe['Frame'] == 0)
        trimmed_df = eachExp.Rg_dataframe.loc[mask][['ID', 'Exp_Name', 'Rg', 'Exp_Category']]
        binValue=np.arange(min(trimmed_df['Rg']), max(trimmed_df['Rg']) + binWidth, binWidth)
        values, base = np.histogram(trimmed_df['Rg'], bins=binValue)
        cumulative = np.cumsum(values)
        normCumulative = cumulative / exp_length
        plot_label = eachExp.Rg_dataframe['Exp_Category'].iloc[0]
        axes.plot(base[:-1], normCumulative, label=plot_label)
    plt.legend(loc='lower right')
    plt.xlabel('Rg', fontsize=16)
    plt.ylabel('Percent', fontsize=16)
    plt.title(f'{plotTitle}HaloTag_Rg CDF', fontsize=18) # CDF stands for cumulative distribution function
    plt.tight_layout()
    if autoSavePlot:
        outLabel = ""
        for eachLabel in exp_labels:
            outLabel += " " + str(eachLabel)
        plt.savefig(exp_list[0].output_dir + f'CumulDistrib_Rg{outLabel}' + '.png', bbox_inches='tight')
    plt.show()

# ! This is the part to edit:
if __name__ == '__main__':
    # General inputs
    # The directories with the pickled JSON files in it.
    #       --Make as many as you need.
    #       --Here I've made two, one Control the other Yoda1
    # Control data locations
# 1. Control
    data1_pickled_JSON_files_directory = r'/home/vivek/Tobias_Group/Piezo1/HaloTag_Gabby/Testing_Plotter/200Frame_Split_JSONs/'
    data1_plot_output_directory = data1_pickled_JSON_files_directory
    # yoda1 data locations
# 2. yoda1
    data2_pickled_JSON_files_directory = r'/home/vivek/Tobias_Group/Piezo1/HaloTag_Gabby/Testing_Plotter/200Frame_Split_JSONs/'
    data2_plot_output_directory = data2_pickled_JSON_files_directory
# 3. Got more data? Use the same template as above, just rename data2 to data3

    # Do you want to automatically save plots in the output directory?
    # NOTE: since boxplots use more than one data set, the plot will be saved in the FIRST output folder (data1)
    autoSavePlot = False

    # Plot specific inputs
    # Kernel Density, show mean on plot or not
    show_kernel_mean_plot = True

    # Examples of plotting data
    # 'control' is the control data, but you can name it anything as long as you are consistent
    # 'yoda1' is the yoda1 data, but you can name it anything as long as you are consistent

    # STEP 1:
    # First load the data. Note: this does NOT plot anything.  It just loads the data into a nice format for plotting
    control = JSONplotter(data1_pickled_JSON_files_directory, data1_plot_output_directory, autoSavePlot)
    yoda1 = JSONplotter(data2_pickled_JSON_files_directory, data2_plot_output_directory, autoSavePlot)

    # STEP 2:
    # Then choose the plotting operations and add any missing arguments you want to add in.
    # Histogram
    # Function arguments: def histogram(self, autoSavePlot=False):

    # control.histogram()
    # yoda1.histogram()

    # Kernel Density
    # Function arguments: def kernel_density(self, plotMean=True, autoSavePlot=False, plotTitle='', xRange=[0, 1.0], yRange=[-3, 20]):

    # control.kernel_density(plotTitle='Control_')
    # yoda1.kernel_density(plotTitle='yoda1_')

    # Cumulative Distribution Function
    # Function Arguments def cumulative_distribution(exp_list:list, exp_labels:list, autoSavePlot=False, plotTitle='Categorical_', binWidth=0.1):
    exp_labels = ['Control', 'yoda1']
    cumulative_distribution([control, yoda1], exp_labels, autoSavePlot, 'Categorical ')

    # Boxplots
    # Function arguments: def box_plot(exp_list:list, exp_labels:list, autoSavePlot=False, plotTitle='Categorical_'):
    # NOTE: For boxplot we need the categorical labels for each type of condition
    # exp_labels = ['Control', 'yoda1']
    # box_plot([control, yoda1], exp_labels, autoSavePlot, 'Categorical ') # Trailing space in plot title is needed





    ########### Extra Snippets for Inspiration #############
    # # 1. Output all Rg values to .csv so you can import to Excel, or other program
    # # Steps: first print the dataframe of interest to see what columns and rows you actually want, otherwise you will get a TON of data
    # print(control.Rg_dataframe)
    # # Looks like all the Rg values are the same for every frame of a trajectory. They were probably calculated for the whole trajectory and inserted into every frame.
    # # So, we really only want the TrackID, the Exp_Name and the Rg value
    # # Lets make a mask to define rows we want to limit by a value
    # mask = (control.Rg_dataframe['Frame'] == 0)
    # outputDF = control.Rg_dataframe.loc[mask][['ID', 'Exp_Name', 'Rg']]
    # outputDF.to_csv(os.path.join(data1_plot_output_directory, 'control.csv'), index=False)
    # # If you want to output each experiment as its own .csv you can use list comprehension combined with a for loop
    # # The steps would be, first identify all the unique experiment names, then iterate over the dataframe using those as a mask to slice the data you want
    # # then output a .csv the same as above for each iteration over the loop.