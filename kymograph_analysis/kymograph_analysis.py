import os
import sys
import timeit
import pathlib
import datetime
import tifffile
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from kymograph_analysis_mods.processor_kymograph_analysis import ImageProcessor

# set the behavior for two types of errors: divide-by-zero and invalid arithmetic operations 
np.seterr(divide='ignore', invalid='ignore') 

# set this to zero so that warning does not pop up
plt.rcParams['figure.max_open_warning'] = 0

def convert_images(directory):
    """
    Convert all TIF files in the specified directory to standardized numpy arrays.

    Args:
        directory (str or pathlib.Path): The path to the directory containing the TIF files.

    Returns:
        dict: A dictionary where the keys are the file names and the values are the numpy arrays of the images.
    """
    input_path = pathlib.Path(directory)
    images = {}

    for file_path in input_path.glob('*.tif'):
        try:
            # Load the TIFF file into a numpy array
            image = tifffile.imread(file_path)

            # standardize image dimensions
            with tifffile.TiffFile(file_path) as tif_file:
                metadata = tif_file.imagej_metadata
            num_channels = metadata.get('channels', 1)
            image = image.reshape(num_channels, 
                                    image.shape[-2],  # rows
                                    image.shape[-1])  # cols
            
            images[file_path.name] = image
                        
        except tifffile.TiffFileError:
            print(f"Warning: Skipping '{file_path.name}', not a valid TIF file.")

    # Sort the dictionary keys alphabetically
    images = {key: images[key] for key in sorted(images)}

    return images   

####################################################################################################################################
####################################################################################################################################

def main():
    folder_path = '/Users/domchom/Desktop/136DCE_138DCE_analysis'
    plot_mean_CCFs = True
    plot_mean_peaks = True
    plot_mean_acfs = True
    plot_ind_CCFs = True
    plot_ind_peaks = False
    plot_ind_acfs = False
    fast_process = False
    line_width = 5
    group_names = ['']

    # Error Catching
    errors = []

    if line_width == '':
        line_width = 1
    try:
        line_width = int(line_width)
        if line_width % 2 == 0:
            raise ValueError("Line width must be odd")
    except ValueError:
        errors.append("Line width must be an odd number")

    if len(errors) >= 1:
        print("Error Log:")
        for count, error in enumerate(errors):
            print(count, ":", error)
        sys.exit("Please fix errors and try again.")

    # make dictionary of parameters for log file use
    log_params = {"Base Directory": folder_path,
                  "Plot Summary ACFs": plot_mean_acfs,
                "Plot Summary CCFs": plot_mean_CCFs,
                "Plot Summary Peaks": plot_mean_peaks,
                "Plot Individual CCF": plot_mean_acfs,
                "Plot Individual CCFs": plot_ind_CCFs,
                "Plot Individual Peaks": plot_ind_peaks,  
                "Line width": line_width,
                "Group Names" : group_names,
                "Files Processed": [],
                "Files Not Processed": [],
                'Plotting errors': [],
                "Group Matching Errors" : [],
                "Multiprocessing": fast_process
                }
        
    def make_log(directory, logParams):
        """
        Creates a log file in the specified directory with the given parameters.

        Args:
            directory (str): The directory in which to create the log file.
            logParams (dict): A dictionary of key-value pairs to write to the log file.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        now = datetime.datetime.now()
        logPath = os.path.join(
            directory, f"0_log-{now.strftime('%Y%m%d%H%M')}.txt")
        logFile = open(logPath, "w")
        logFile.write("\n" + now.strftime("%Y-%m-%d %H:%M") + "\n")
        for key, value in logParams.items():
            logFile.write('%s: %s\n' % (key, value))
        logFile.close()

    def plotComparisons(dataFrame: pd.DataFrame, dependent: str, independent = 'Group Name'):
        '''
        This func accepts a dataframe, the name of a dependent variable, and the name of an
        independent variable (by default, set to Group Name). It returns a figure object showing
        a box and scatter plot of the dependent variable grouped by the independent variable.
        '''
        ax = sns.boxplot(x=independent, y=dependent, data=dataFrame, palette = "Set2", showfliers = False)
        ax = sns.swarmplot(x=independent, y=dependent, data=dataFrame, color=".25")	
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
        fig = ax.get_figure()
        return fig

    file_names = [fname for fname in os.listdir(folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

    # list of groups that matched to file names
    groups_found = np.unique([group for group in group_names for file in file_names if group in file]).tolist()

    # dictionary of file names and their corresponding group names
    uniqueDic = {file : [group for group in group_names if group in file] for file in file_names}

    for file_name, matching_groups in uniqueDic.items():
        # if a file doesn't have a group name, log it but still run the script
        if len(matching_groups) == 0:
            log_params["Group Matching Errors"].append(f'{file_name} was not matched to a group')

        # if a file has multiple groups names, raise error and exit the script
        elif len(matching_groups) > 1:
            print('****** ERROR ******',
                f'\n{file_name} matched to multiple groups: {matching_groups}',
                '\nPlease fix errors and try again.',
                '\n****** ERROR ******')
            sys.exit()

    # if a group was specified but not matched to a file name, raise error and exit the script
    if len(groups_found) != len(group_names):
        print("****** ERROR ******",
            "\nOne or more groups were not matched to file names",
            f"\nGroups specified: {group_names}",
            f"\nGroups found: {groups_found}",
            "\n****** ERROR ******")
        sys.exit()

    # performance tracker
    start = timeit.default_timer()

    # create main save path
    now = datetime.datetime.now()
    os.chdir(folder_path)
    main_save_path = os.path.join(
        folder_path, f"!kymograph_processing-{now.strftime('%Y%m%d%H%M')}")

    # create directory if it doesn't exist
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    # empty list to fill with summary data for each file
    summary_list = []
    # column headers to use with summary data during conversion to dataframe
    col_headers = []

    # create a dictionary of the filename and corresponding images as mp arrays. 
    all_images = convert_images(folder_path)

    # processing movies
    with tqdm(total=len(file_names)) as pbar:
        pbar.set_description('Files processed:')
        for file_name in file_names:
            print('******'*10)
            print(f'Processing {file_name}...')

            # name without the extension
            name_wo_ext = file_name.rsplit(".", 1)[0]

            # create a subfolder within the main save path with the same name as the image file
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            if not os.path.exists(im_save_path):
                os.makedirs(im_save_path)

            processor = ImageProcessor(filename=file_name, 
                        im_save_path=im_save_path,
                        img=all_images[file_name],
                        line_width=line_width
                        )
        
            # log error and skip image if frames < 2 
            if processor.num_cols < 2:
                print(f"****** ERROR ******",
                    f"\n{file_name} has less than 2 frames",
                    "\n****** ERROR ******")
                log_params['Files Not Processed'].append(f'{file_name} has less than 2 frames')
                continue

            # if file is not skipped, log it and continue
            log_params['Files Processed'].append(f'{file_name}')
            
            # if user entered group name(s) into GUI, match the group for this file. If no match, keep set to None
            group_name = None
            if group_names != ['']:
                try:
                    group_name = [group for group in group_names if group in name_wo_ext][0]
                except IndexError:
                    pass

            # if file is not skipped, log it and continue
            log_params['Files Processed'].append(f'{file_name}')

            # calculate the population signal properties
            processor.calc_ind_peak_props()
            processor.calc_indv_ACF()
            if processor.num_channels > 1:
                processor.calc_indv_CCFs()

            # Plot the following parameters if selected
            if fast_process:
                if plot_ind_peaks:
                    ind_peak_plots = processor.plot_ind_peak_props()
                    processor.save_plots(plots=ind_peak_plots, plot_dir=os.path.join(im_save_path, 'Individual_peak_plots'))

                if plot_ind_CCFs:
                    if processor.num_channels == 1:
                            log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                    ind_ccf_plots = processor.plot_ind_ccfs()
                    processor.save_plots(plots=ind_ccf_plots, plot_dir=os.path.join(im_save_path, 'Individual_CCF_plots'))

                if plot_ind_acfs:
                    ind_acfs_plots = processor.plot_ind_acfs()
                    processor.save_plots(plots=ind_acfs_plots, plot_dir=os.path.join(im_save_path, 'Individual_ACF_plots'))
            
            else:
                if plot_ind_peaks:
                    ind_peak_plots = processor.plot_ind_peak_props()
                    ind_peak_path = os.path.join(im_save_path, 'Individual_peak_plots')
                    if not os.path.exists(ind_peak_path):
                        os.makedirs(ind_peak_path)
                    for plot_name, plot in ind_peak_plots.items():
                        plot.savefig(f'{ind_peak_path}/{plot_name}.png')

                if plot_ind_CCFs:
                    if processor.num_channels == 1:
                            log_params['Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                    ind_ccf_plots = processor.plot_ind_ccfs()
                    ind_ccf_path = os.path.join(im_save_path, 'Individual_CCF_plots')
                    if not os.path.exists(ind_ccf_path):
                        os.makedirs(ind_ccf_path)
                    for plot_name, plot in ind_ccf_plots.items():
                        plot.savefig(f'{ind_ccf_path}/{plot_name}.png')

                if plot_ind_acfs:
                    ind_acfs_plots = processor.plot_ind_acfs()
                    ind_acf_path = os.path.join(im_save_path, 'Individual_ACF_plots')
                    if not os.path.exists(ind_acf_path):
                        os.makedirs(ind_acf_path)
                    for plot_name, plot in ind_acfs_plots.items():
                        plot.savefig(f'{ind_acf_path}/{plot_name}.png')

            if plot_mean_CCFs:
                summ_ccf_plots = processor.plot_mean_CCF()
                for plot_name, plot in summ_ccf_plots.items():
                    plot.savefig(f'{im_save_path}/{plot_name}.png')

            if plot_mean_peaks:
                summ_peak_plots = processor.plot_mean_peak_props()
                for plot_name, plot in summ_peak_plots.items():
                    plot.savefig(f'{im_save_path}/{plot_name}.png')

            if plot_mean_acfs:
                summ_acf_plots = processor.plot_mean_ACF()
                for plot_name, plot in summ_acf_plots.items():
                    plot.savefig(f'{im_save_path}/{plot_name}.png')

            # Summarize the data for current image as dataframe, and save as .csv
            im_measurements_df = processor.organize_measurements()
            im_measurements_df.to_csv(f'{im_save_path}/{name_wo_ext}_measurements.csv', index=False)

            # generate summary data for current image
            im_summary_dict = processor.summarize_image(file_name = file_name, group_name = group_name)

            # populate column headers list with keys from the measurements dictionary
            for key in im_summary_dict.keys():
                if key not in col_headers:
                    col_headers.append(key)

            # append summary data to the summary list
            summary_list.append(im_summary_dict)

            # useless progress bar to force completion of previous bars
            with tqdm(total=10, miniters=1) as dummy_pbar:
                dummy_pbar.set_description('cleanup:')
                for i in range(10):
                    dummy_pbar.update(1)

            pbar.update(1)
    
        # create dataframe from summary list
        summary_df = pd.DataFrame(summary_list, columns=col_headers)

        # Get column names for all channels
        channel_cols = [col for col in summary_df.columns if 'Ch' in col and 'Mean Peak Rel Amp' in col]
        # save the summary csv file
        summary_df = summary_df.sort_values('File Name', ascending=True)

        # Normalize mean peak relative amplitude for each channel
        for col in channel_cols:
            channel_mean_rel_amp = summary_df[col]
            norm_mean_rel_amp = channel_mean_rel_amp / channel_mean_rel_amp.min()
            norm_col_name = col.replace('Mean Peak Rel Amp', 'Norm Mean Rel Amp')
            summary_df[norm_col_name] = norm_mean_rel_amp

        summary_df.to_csv(f'{main_save_path}/summary.csv', index=False)

        # if group names were entered into the gui, generate comparisons between each group
        if group_names != ['']:
            print('Generating group comparisons...')
            # make a group comparisons save path in the main save directory
            group_save_path = os.path.join(main_save_path, "0_groupComparisons")
            if not os.path.exists(group_save_path):
                os.makedirs(group_save_path)
            
            # make a list of parameters to compare
            stats_to_compare = ['Mean']
            channels_to_compare = [f'Ch {i+1}' for i in range(processor.num_channels)]
            measurements_to_compare = ['Period', 'Shift', 'Peak Width', 'Peak Max', 'Peak Min', 'Peak Amp', 'Peak Rel Amp']
            params_to_compare = []
            for channel in channels_to_compare:
                for stat in stats_to_compare:
                    for measurement in measurements_to_compare:
                        params_to_compare.append(f'{channel} {stat} {measurement}')

            # will compare the shifts if multichannel movie
            if hasattr(processor, 'channel_combos'):
                shifts_to_compare = [f'Ch{combo[0]+1}-Ch{combo[1]+1} Mean Shift' for combo in processor.channel_combos]
                params_to_compare.extend(shifts_to_compare)

            # generate and save figures for each parameter
            for param in params_to_compare:
                try:
                    fig = plotComparisons(summary_df, param)
                    fig.savefig(f'{group_save_path}/{param}.png')
                    plt.close(fig)
                except ValueError:
                    log_params['Plotting errors'].append(f'No data to compare for {param}')

            # save the means for the attributes to make them easier to work with in prism
            processor.save_means_to_csv(main_save_path, group_names, summary_df)

        end = timeit.default_timer()
        log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
        # log parameters and errors
        make_log(main_save_path, log_params)
        

if __name__ == '__main__':
    main()

print('Done with Script!')