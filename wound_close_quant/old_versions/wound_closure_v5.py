import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pyclesperanto_prototype as cle
import tifffile
import scipy 
import imageio.v2 as imageio
import os 
import glob
import logging
import napari
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#################################################
#############   USER INPUTS   ###################
#################################################

num_lines = 45 #how many line scans per frame
line_length = 150 #choose you line length
bin_num = 150 #number of data points per line
image_folder_path = '/Users/domchom/Documents/GitHub/Dom_python_scripts/wound_close_quant/movies' #path to movie of interest

# Define variables for filter and normalization parameters
window_length = 11 #for smoothing the line
polyorder = 2 #for smoothing the line
norm_min = 0 #set the min value for normalization
norm_max = 1 #set the max value for normalization

#################################################
#################################################
#################################################
#################################################
#################################################

# Set up logging
current_time = datetime.now().time()

os.chdir(image_folder_path)
if not os.path.exists(f'analysis_{current_time}'):
        os.mkdir(f'analysis_{current_time}')

#importing the images
def get_Images(folder_path):
    ''' saves all images in a dict {filepath: img as np array}'''
    all_images = {}
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if file is a TIF file
        if filename.endswith('.tif'):
            # Open the TIF file using gdal
            filepath = os.path.join(folder_path, filename)

            img = tifffile.imread(filepath)

            # standardize image dimensions
            with tifffile.TiffFile(filepath) as tif_file:
                metadata = tif_file.imagej_metadata
            num_channels = metadata.get('channels', 1)
            num_slices = metadata.get('slices', 1)
            num_frames = metadata.get('frames', 1)
            img = img.reshape(num_frames, 
                            num_slices, 
                            num_channels, 
                            img.shape[-1], #columns
                            img.shape[-2])  #rows (y)
            
            all_images[filepath] = img
    
    return all_images

def user_define_ellipse(image_path):
    '''Will open one image specified by the image path. The user will then draw an ellipse. It will
    then save the ellipse dimensions and close the napari viewer.'''
    # asking the user to identify the ring of interest
    with napari.gui_qt():
        ellipse_data = np.array([[400, 100], [400, 400], [100, 400], [100, 100]])
        viewer = napari.Viewer()
        viewer.open(image_path)

        ellipse_layer = viewer.add_shapes(
            data=[ellipse_data],
            shape_type='ellipse',
            edge_color='red',
            edge_width=2,
            face_color='transparent',
        )

        @viewer.bind_key('s')
        def save_shape(viewer):
            last_shape = viewer.layers['Shapes'].data[-1]
            if last_shape is not None:
                viewer.window.close()
                return last_shape
            else:
                print('No ellipse has been drawn yet')

        napari.run()

    return save_shape(viewer)
    
def get_center_and_ratio(shape):
    '''finds the center point of a shape. it also gives the ratio in height:width'''
    # Define the rectangle's coordinates
    print(shape)
    x1, y1 = shape[0][0] , shape[0][1]
    x2, y2 = shape[2][0] , shape[2][1]

    # Calculate the center and ratio
    center = [(x1 + x2) / 2, (y1 + y2) / 2]
    ratio = abs(y2 - y1) / abs(x2 - x1)

    return center, ratio


def create_lines(center, ellipse_ratio):
    '''creates lines in an elliptical shape with a given center. It creates
    two ellipses and iterates over the larger ellipse to find the closest points on the smaller ellipse.'''
    # Define the two ellipses
    small_ellipse_width = 20
    small_ellipse_height = small_ellipse_width * ellipse_ratio
    large_ellipse_width = line_length * 2 + small_ellipse_width
    large_ellipse_height = line_length * 2 + small_ellipse_height

    small_ellipse = Ellipse(xy=(center[0], center[1]), width=small_ellipse_width, height=small_ellipse_height, angle=0)
    large_ellipse = Ellipse(xy=(center[0], center[1]), width=large_ellipse_width, height=large_ellipse_height, angle=0)

    # Check if small ellipse is inside large ellipse
    if large_ellipse.contains_point(small_ellipse.center):
        
        # Create an array of points on the large ellipse
        theta = np.linspace(0, 2 * np.pi, num_lines)
        points = np.stack([large_ellipse.center[0] + large_ellipse.width/2*np.cos(theta), 
                        large_ellipse.center[1] + large_ellipse.height/2*np.sin(theta)], axis=1)

        # Loop over each point on the large ellipse and calculate the shortest distance to the small ellipse and line segment
        line_coords = [[np.linspace(x0, x1, bin_num), np.linspace(y0, y1, bin_num)] 
                    for i, point in enumerate(points) 
                    if (distance := np.linalg.norm(small_ellipse.center - point) - small_ellipse_width / 2) >= 0
                    for theta in [np.arctan2(point[1] - large_ellipse.center[1], point[0] - large_ellipse.center[0])]
                    for x0, y0, x1, y1 in [[point[0], point[1], 
                                            small_ellipse.center[0] + small_ellipse_width / 2 * np.cos(theta), 
                                            small_ellipse.center[1] + small_ellipse_height / 2 * np.sin(theta)]]]

        
    return line_coords


def return_line_ref_figure(img, line_coords, center):
    '''Plot the ellipses and lines. returns the fig'''
    fig, ax = plt.subplots()
    for coords in line_coords:
        ax.plot(coords[1], coords[0], 'w-', linewidth=0.5)
    ax.plot(center[1], center[0], 'ro')
    ax.set_aspect('equal', adjustable='box')
    ax.imshow(img)    
    fig.subplots_adjust(hspace=0.5)
    return fig

#perform the line scans
def calc_line_scans(img, num_frames, num_channels, line_coords):
    """
    Calculates the signal along each line in each frame of an image stack.

    Args:
        img (ndarray): A 3D numpy array of shape (num_frames, num_channels, image_shape).
        num_frames (int): The number of frames in the image stack.
        num_channels (int): The number of channels in the image stack.
        line_coords (list): A list of line coordinates, where each line is a list of x and y coordinates.

    Returns:
        list: A list of channels, frames, where each frame is a list of signals, where each signal is an array of signal values.
    """
    channels_list = []


    for c in range(num_channels): #iterate over the channels
        frames_list = [] # store all the signals for every frame

        for f in range(num_frames): #iterate over the frames
            signals_list = [] #lost to store the signal for each line for the frame
            logging.info(f'Calculating linescan: channel {c+1}, frame {f+1}')
            i = 0
            for line in line_coords: #create each line for each frame
                x, y = line[0], line[1]

                if np.all(abs(center - x[-1]) < abs(center - x[0])):
                    x = x[::-1]
                    x0, x1 = x[-1], x[0]
                else:
                    x = x
                    x0, x1 = x[0], x[-1]

                if np.all(abs(center - y[-1]) < abs(center - y[0])):
                    y = y[::-1]
                    y0, y1 = y[-1], y[0]
                else:
                   y = y
                   y0, y1 = y[0], y[-1]

                # Extract the values along the line, using cubic interpolation
                signal = scipy.ndimage.map_coordinates(img[f][0][c], np.vstack((x,y)))
                signals_list.append(signal)

                #plot_idv_lines(filename_short,img,signal,f,c,x0,x1,y0,y1,i)
                i += 1

            frames_list.append(signals_list)

        channels_list.append(frames_list)

    logging.info(f'Finished calculating line scans for {num_frames} frames and {num_channels} channels')
    
    line_values_array = np.array(channels_list)

    return line_values_array

def plot_idv_lines(filename, img, signal,f,c,x0,x1,y0,y1,i):
    logging.info(f'Plotting line scan: channel {c+1}, frame {f+1}, line{i+1}')

    if not os.path.exists(f'{image_folder_path}/analysis_{current_time}/indv_linescans'):
        os.mkdir(f'{image_folder_path}/analysis_{current_time}/indv_linescans/')

    if not os.path.exists(f'{image_folder_path}/analysis_{current_time}/indv_linescans/{filename}_indv_linescans'):
        os.mkdir(f'{image_folder_path}/analysis_{current_time}/indv_linescans/{filename}_indv_linescans')

    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=2)
    axes[0].imshow(img[f][0][c])
    axes[0].plot([y0, y1], [x0, x1], 'r-')
    axes[0].axis('image')

    axes[1].plot(signal)

    file_name = f"{filename}_channel{c}_frame{f}_line{i}.png"
    ref_fig_file_path = image_folder_path + f'/analysis_{current_time}/indv_linescans/{filename}_indv_linescans/' + file_name
    plt.close()
    fig.savefig(ref_fig_file_path)



    if not os.path.exists(f'{image_folder_path}/analysis_{current_time}/reference_images'):
        os.mkdir(f'{image_folder_path}/analysis_{current_time}/reference_images')

    fig = return_line_ref_figure(image[14][0][0], line_coords, center) #create the figure
    file_name = filename.split('.')[0]
    file_name = f"{file_name.split('/')[-1]}_ref.png"
    ref_fig_file_path = image_folder_path + f'/analysis_{current_time}/reference_images/' + file_name




def calculate_mean_values(array):
    '''
    Iterates over each frame of a np array containing (num_channels, num_frames, num_lines in frame, num_bins in line).
    Averages each corresponding pixel on the line for each frame, and then returns a np
    array that contains (num_frames, one line in frame, num_bins in line)
    '''
    mean_values_all_ch = []

    for channel in array:
        bin_num = channel.shape[-1]
        mean_values = []
        for frame in channel:
            pixel_avg_per_frame = []
            for n in range(bin_num):
                pixel_avg_per_frame.append(frame[:,n].mean())
            mean_values.append(pixel_avg_per_frame)
        mean_values_all_ch.append(mean_values)

    mean_values_all_ch = np.array(mean_values_all_ch)

    return mean_values_all_ch

# Define a function to normalize the data for a channel
def normalize_data(array_raw_values):
    norm_values = []
    for channel in array_raw_values:
        channel_values = []
        for signal in channel:
            signal_max = np.max(signal)
            norm_signal = (signal / signal_max) * (norm_max - norm_min) + norm_min
            channel_values.append(norm_signal)
        norm_values.append(channel_values)
    
    return norm_values

# Define a function to subtract background from the data for a channel
def subtract_background(array_raw_values):
    norm_values = []
    for channel in array_raw_values:
        bg_values = np.min(channel)
        channel_values = []
        for signal in channel:
            signal_bg = (signal - bg_values) * (norm_max - norm_min) / (np.max(channel) - bg_values)
            channel_values.append(signal_bg)
        norm_values.append(channel_values)
    return norm_values

#define a function to smooth data
def smooth_data(array_values):
    smoothed_data = []
    for channel in array_values:
        channel_values = []
        for signal in channel:
            signal_values = scipy.signal.savgol_filter(signal, window_length=window_length, polyorder=polyorder)
            channel_values.append(signal_values)
        smoothed_data.append(channel_values)
    return smoothed_data

#find peaks - will treat each frame like a box
def find_peaks(array):
    # make empty arrays to fill with peak measurements for each channel

    peak_widths = np.zeros(shape=(len(array), len(array[0])))
    peak_maxs = np.zeros(shape=(len(array), len(array[0])))
    peak_mins = np.zeros(shape=(len(array), len(array[0])))

    # make a dictionary to store the arrays and measurements generated by this function so they don't have to be re-calculated later
    ind_peak_props = {}

    for channel in range((len(array))):
        for frame_num in range((len(array[0]))):

            signal = scipy.signal.savgol_filter(array[channel][frame_num], window_length = window_length, polyorder = polyorder)
            peaks, _ = scipy.signal.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.65) #<--- edit last number to change the threshold

            # if peaks detected, calculate properties and return property averages. Otherwise return nans
            if len(peaks) > 0:
                proms, _, _ = scipy.signal.peak_prominences(signal, peaks)
                widths, heights, leftIndex, rightIndex = scipy.signal.peak_widths(signal, peaks, rel_height=0.5)
                mean_width = np.mean(widths, axis=0)
                mean_max = np.mean(signal[peaks], axis = 0)
                mean_min = np.mean(signal[peaks]-proms, axis = 0)
                peak_widths[channel, frame_num] = mean_width
                peak_maxs[channel, frame_num] = mean_max
                peak_mins[channel, frame_num] = mean_min

                # store the smoothed signal, peak locations, maxs, mins, and widths for each box in each channel
                ind_peak_props[f'Ch {channel} Box {frame_num}'] = {'smoothed': signal, 
                                                        'peaks': peaks,
                                                        'proms': proms, 
                                                        'heights': heights, 
                                                        'leftIndex': leftIndex, 
                                                        'rightIndex': rightIndex}
                    
            else:
                peak_widths[channel, frame_num] = np.nan
                peak_maxs[channel, frame_num] = np.nan
                peak_mins[channel, frame_num] = np.nan

                # store the smoothed signal, peak locations, maxs, mins, and widths for each box in each channel
                ind_peak_props[f'Ch {channel} Frame {frame_num}'] = {'smoothed': np.nan, 
                                                        'peaks': np.nan,
                                                        'proms': np.nan, 
                                                        'heights': np.nan, 
                                                        'leftIndex': np.nan, 
                                                        'rightIndex': np.nan}

    peak_amps = peak_maxs - peak_mins
    peak_rel_amps = peak_amps / peak_mins

    return ind_peak_props, peak_amps, peak_rel_amps

def cross_correlation(signal1, signal2):
    peaks1, _ = scipy.signal.find_peaks(signal1, prominence=(np.max(signal1)-np.min(signal1))*0.20)
    peaks2, _ = scipy.signal.find_peaks(signal2, prominence=(np.max(signal2)-np.min(signal2))*0.20)

    if len(peaks1) > 0 and len(peaks2) > 0:
        corr_signal1 = signal1 - signal1.mean()
        corr_signal2 = signal2 - signal2.mean()
        cc_curve = np.correlate(corr_signal1, corr_signal2, mode='full')
        #smooth the curve
        cc_curve = scipy.signal.savgol_filter(cc_curve, window_length=11, polyorder=2)
        # normalize the curve
        num_frames = len(signal1)
        cc_curve = cc_curve / (num_frames * signal1.std() * signal2.std())
        #find peaks
        peaks, _ = scipy.signal.find_peaks(cc_curve, prominence=0.01)
        # absolute difference between each peak and zero
        peaks_abs = abs(peaks - cc_curve.shape[0] // 2)
        # if peaks were identified, pick the one closest to the center
        if len(peaks) > 1:
            delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
            delayIndex = peaks[delay]
            delay_frames = delayIndex - cc_curve.shape[0] // 2
        # otherwise, return nans for both period and autocorrelation curve
        else:
            delay_frames = np.nan
            cc_curve = np.full((bin_num * 2 - 1), np.nan)
    else:
        delay_frames = np.nan
        cc_curve = np.full((bin_num * 2 - 1), np.nan)

    return delay_frames, cc_curve


def return_figure(ch1: np.ndarray, ch2: np.ndarray, ccf_curve: np.ndarray, ch1_name: str, ch2_name: str, shift: int):
    '''
    Space saving function to generate individual plots with variable input. returns a figure object.
    '''
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(ch1, color = 'tab:cyan', label = ch1_name)
    ax1.plot(ch2, color = 'tab:red', label = ch2_name)
    ax1.set_xlabel('distance (pixels)')
    ax1.set_ylabel('Mean box px value')
    ax1.legend(loc='upper right', fontsize = 'small', ncol = 1)
    ax2.plot(np.arange(-line_length + 1, line_length ), ccf_curve)
    ax2.set_ylabel('Crosscorrelation')
    
    if not shift == np.nan:
        color = 'red'
        ax2.axvline(x = shift, alpha = 0.5, c = color, linestyle = '--')
        if shift < 1:
            ax2.set_xlabel(f'{ch1_name} leads by {int(abs(shift))} pixels')
        elif shift > 1:
            ax2.set_xlabel(f'{ch2_name} leads by {int(abs(shift))} pixels')
        else:
            ax2.set_xlabel('no shift detected')
    else:
        ax2.set_xlabel(f'No peaks identified')
    
    fig.subplots_adjust(hspace=0.5)
    return fig


# Define function to create a linescan plot for a single frame
def create_linescan_plot(frame, ch1, ch2, y_max, smooth = None):
    plt.style.use('dark_background')
    fig, (ax1) = plt.subplots(1, 1)
    ax1.set_ylim([0, y_max])

    if smooth == True:
        signal1 = scipy.signal.savgol_filter(ch1[frame], window_length=11, polyorder=2)
        signal2 = scipy.signal.savgol_filter(ch2[frame], window_length=11, polyorder=2)
    else:
        signal1 = ch1[frame]
        signal2 = ch2[frame]

    ax1.plot(signal1, color='cyan', label='Ch1')
    ax1.plot(signal2, color='red', label='Ch2')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
    
    return fig

# Define function to create linescan movie
def create_linescan_movie(ch1, ch2, filename, smooth = None):

    y_max = np.max(np.maximum(ch1, ch2))
    if not os.path.exists(f'{image_folder_path}/analysis_{current_time}/linescan_movies'):
        os.mkdir(f'{image_folder_path}/analysis_{current_time}/linescan_movies')
    for i in range(len(ch1)):
        fig = create_linescan_plot(i, ch1, ch2, y_max, smooth)
        plt.savefig(f'{image_folder_path}/analysis_{current_time}/linescan_movies/frame_{i:04d}.png')
        plt.close(fig)
    with imageio.get_writer(f'{image_folder_path}/analysis_{current_time}/linescan_movies/{filename}.mp4', mode='I') as writer:
        for i in range(len(ch1)):
            image = imageio.imread(f'{image_folder_path}/analysis_{current_time}/linescan_movies/frame_{i:04d}.png')
            writer.append_data(image)
    file_list = glob.glob(os.path.join(f'{image_folder_path}/analysis_{current_time}/linescan_movies/', "*.png"))
    for file_path in file_list:
        os.remove(file_path)

def save_reference_image(image, line_coords, center, current_time, image_folder_path):

    if not os.path.exists(f'{image_folder_path}/analysis_{current_time}/reference_images'):
        os.mkdir(f'{image_folder_path}/analysis_{current_time}/reference_images')

    fig = return_line_ref_figure(image[15][0][0], line_coords, center) #create the figure
    file_name = filename.split('.')[0]
    file_name = f"{file_name.split('/')[-1]}_ref.png"
    ref_fig_file_path = image_folder_path + f'/analysis_{current_time}/reference_images/' + file_name
    plt.close()
    fig.savefig(ref_fig_file_path)

def save_mean_frame_linescans(filename, array, ccfs, shifts):
 # Create a new directory to store the figures
    if not os.path.exists(f'{image_folder_path}/analysis_{current_time}/mean_frame_linescans'):
        os.mkdir(f'{image_folder_path}/analysis_{current_time}/mean_frame_linescans')

    if not os.path.exists(f'{image_folder_path}/analysis_{current_time}/mean_frame_linescans/{filename}'):
        os.mkdir(f'{image_folder_path}/analysis_{current_time}/mean_frame_linescans/{filename}')

    for frame in range(len(array[0])):
        fig = return_figure(array[0][frame], array[1][frame], ccfs[0][frame], "rGBD", "wGBD", shifts[0][frame])
        file_name = f"indv_frame_scan_{frame + 1}.png"
        file_path = os.path.join(f'{image_folder_path}/analysis_{current_time}/mean_frame_linescans/{filename}', file_name)
        plt. close()
        fig.savefig(file_path)


#######################################################
#######################################################
#######################################################
#######################################################

#get all images in folder
all_images = get_Images(image_folder_path)

#dict for saving the ellipse center and ratio
all_ellipse_center_ratio = {}

#loop over all images to find center and ratio
for key in all_images:
    ellipse = user_define_ellipse(key) #the user will create the ellipse in napari for the given image
    all_ellipse_center_ratio[key] = ellipse

for filename in all_images:
    filename_short = filename.split('/')[-1]
    ellipse = all_ellipse_center_ratio[filename]
    center, ratio = get_center_and_ratio(ellipse) #save center, ratio
    image = all_images[filename] #get the image np array
    line_coords =  create_lines(center, ratio) #create the lines 
    
    save_reference_image(image, line_coords, center, current_time, image_folder_path)

    line_values_array = calc_line_scans(image, len(image), len(image[0][0]), line_coords)

    # Calculate the mean values for each channel
    raw_mean_values = calculate_mean_values(line_values_array)

    # Normalize the data for each channel
    norm_mean_values = normalize_data(raw_mean_values)

    # Subtract background from the data for each channel
    mean_values_norm_bg = subtract_background(raw_mean_values)

    #smooth all data
    raw_values_smooth = smooth_data(raw_mean_values)
    norm_mean_values_smooth = smooth_data(norm_mean_values)
    mean_values_norm_bg_smooth = smooth_data(mean_values_norm_bg)

    #find the peaks
    ind_peak_props, peak_amps, peak_rel_amps = find_peaks(raw_values_smooth)

    # make a list of unique channel combinations to calculate CCF for
    channels = list(range(len(raw_values_smooth)))
    channel_combos = []
    for i in range(len(raw_values_smooth)):
        for j in channels[i + 1:]:
            channel_combos.append([channels[i], j])
    num_combos = len(channel_combos)

    #calc shifts anc cross-corelation
    shifts = np.zeros(shape=(num_combos, len(raw_values_smooth[0])))
    ccfs = np.zeros(shape=(num_combos, len(raw_values_smooth[0]), bin_num*2-1))

    for combo_number, combo in enumerate(channel_combos):
        for frame_num in range(len(raw_values_smooth[0])):
            delay_frames, cc_curve = cross_correlation(raw_values_smooth[0][frame_num], raw_values_smooth[1][frame_num])
            shifts[combo_number, frame_num] = delay_frames
            ccfs[combo_number, frame_num] = cc_curve

    save_mean_frame_linescans(filename_short, raw_values_smooth, ccfs, shifts)

    logging.info(f'Creating linescan movie for {filename_short}')
    create_linescan_movie(raw_values_smooth[0], raw_values_smooth[1], f'{filename_short}_linescan_movie_raw', False)
    logging.info(f'Done with {filename_short}')

with open(f"/Users/domchom/Documents/GitHub/Dom_python_scripts/wound_close_quant/movies/analysis_{current_time}/log.txt", "w") as file:
    file.write(f"Date_time:{current_time}\n")
    file.write(f"Number of linescans per frame:{num_lines}\n")
    file.write(f"Line length:{line_length}\n")
    file.write(f"Number of bins per line:{bin_num}\n")
    file.write(f"Image folder path:{image_folder_path}\n")
    file.write(f"Window length for smoothing data:{window_length}\n")
    file.write(f"Polyorder for smoothing data:{polyorder}\n")
    file.write(f"Ellipse center coordinates:{center}\n")

logging.info(f'Done with Script!')













    
