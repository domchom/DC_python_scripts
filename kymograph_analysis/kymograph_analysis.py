import os
import sys
import timeit
import pathlib
import datetime
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from kymograph_analysis_mods.processor_kymograph_analysis import ImageProcessor

# set the behavior for two types of errors: divide-by-zero and invalid arithmetic operations 
np.seterr(divide='ignore', invalid='ignore') 

# set this to zero so that warning does not pop up
plt.rcParams['figure.max_open_warning'] = 0

def convert_images(directory):
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
    folder_path = '/Users/domchom/Desktop/kymograph_analysis_testing'
    plot_mean_CCFs = True
    plot_mean_peaks = True
    plot_mean_acfs = True
    plot_ind_CCFs = True
    plot_ind_peaks = False
    plot_ind_acfs = False
    line_width = 3

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
                "Files Processed": [],
                "Files Not Processed": [],
                'Plotting errors': []
                }
        
    ''' ** housekeeping functions ** '''
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

    def save_plots(plots, plot_dir):
        """
        Saves a dictionary of Matplotlib plots to PNG files in the specified directory.

        Parameters:
        -----------
        - plots (dict): A dictionary of Matplotlib plots, where the keys are the names of the plots and the values
                        are the actual plot objects.
        - plot_dir (str): A string representing the path to the directory where the plots should be saved. If the directory
                        doesn't exist, it will be created.
        """
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        for plot_name, plot in plots.items():
            plot.savefig(f'{plot_dir}/{plot_name}.png')

    file_names = [fname for fname in os.listdir(
        folder_path) if fname.endswith('.tif') and not fname.startswith('.')]

    ''' ** Main Workflow ** '''
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

            # Initialize the processor
            processor = ImageProcessor(filename=file_name, 
                                    im_save_path=im_save_path,
                                    img=all_images[file_name],
                                    line_width=line_width
                                    )

            # if file is not skipped, log it and continue
            log_params['Files Processed'].append(f'{file_name}')

            # calculate the population signal properties
            processor.calc_ind_peak_props()
            processor.calc_indv_ACF()
            if processor.num_channels > 1:
                processor.calc_indv_CCFs()

            # Plot the following parameters if selected
            if plot_ind_peaks:
                ind_peak_plots = processor.plot_ind_peak_props()
                save_plots(ind_peak_plots, os.path.join(im_save_path, 'Individual_peak_plots'))

            if plot_ind_CCFs:
                ind_ccf_plots = processor.plot_ind_ccfs()
                save_plots(ind_ccf_plots, os.path.join(im_save_path, 'Individual_CCF_plots'))

            if plot_ind_acfs:
                ind_acfs_plots = processor.plot_ind_acfs()
                save_plots(ind_acfs_plots, os.path.join(im_save_path, 'Individual_ACF_plots'))

            if plot_mean_CCFs:
                mean_ccf_plots = processor.plot_mean_CCF()
                save_plots(mean_ccf_plots, os.path.join(im_save_path, 'Mean_CCF_plots'))

            if plot_mean_peaks:
                mean_peak_plots = processor.plot_mean_peak_props()
                save_plots(mean_peak_plots, os.path.join(im_save_path, 'Mean_peak_plots'))

            if plot_mean_acfs:
                mean_acfs_plots = processor.plot_mean_ACF()
                save_plots(mean_acfs_plots, os.path.join(im_save_path, 'Mean_ACF_plots'))

            # Summarize the data for current image as dataframe, and save as .csv
            im_measurements_df = processor.organize_measurements()
            im_measurements_df.to_csv(
                f'{im_save_path}/{name_wo_ext}_measurements.csv', index=False)

            # generate summary data for current image
            im_summary_dict = processor.summarize_image(
                file_name=file_name)

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
        summary_df.to_csv(f'{main_save_path}/summary.csv', index=False)

        end = timeit.default_timer()
        log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
        # log parameters and errors
        make_log(main_save_path, log_params)
        print('Done with Script!')

if __name__ == '__main__':
    main()