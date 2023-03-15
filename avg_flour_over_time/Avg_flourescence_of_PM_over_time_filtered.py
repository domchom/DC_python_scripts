#note for dom: make sure to set the env to 3.9.7 ('base': conda)

import glob,os
import sys
import math                   
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import skimage.io as skio
from skimage.morphology import disk, binary_erosion, binary_dilation, binary_opening, binary_closing, remove_small_objects
from skimage.restoration import rolling_ball 
from skimage.measure import label
from skimage.filters import gaussian
import tkinter as tk
from tkinter import Tk
from tkinter import ttk
import pyclesperanto_prototype as cle
import sys

######################################
################# GUI ################
######################################

#store the path name outside of the tk thingy 
#not sure why this works or why it is necessary
path_name = ''
raw_plot_value = 0
normal_plot_value = 0
rolling_plot_value = 0
frames_to_roll_value = 0
pre_stim_frames_value = 0

def cancel_script():
    global path_name
    path_name = 'The script was cancelled!'
    window.destroy()
    print("You cancelled the script!")
    sys.exit()

def get_values_close_window():
    global path_name
    path_name = path.get()
    global raw_plot_value
    raw_plot_value = raw_to_plot.get()
    global normal_plot_value
    normal_plot_value = normal_to_plot.get()
    global rolling_plot_value
    rolling_plot_value = int(rolling_to_plot.get())
    global frames_to_roll_value
    frames_to_roll_value = int(frames_to_roll.get())
    global pre_stim_frames_value
    pre_stim_frames_value = int(pre_stim_frame.get())
    window.destroy()

window = tk.Tk()
window.title('User Inputs')
window.geometry('600x190')

#store the path and variables
path = tk.StringVar()
raw_to_plot = tk.IntVar()
normal_to_plot = tk.IntVar()
rolling_to_plot = tk.IntVar()
frames_to_roll = tk.StringVar()
pre_stim_frame = tk.StringVar()

#Path  label and entry
label_path =  tk.Label(text="Paste the path to the folder with your movies:").grid(row=0, column=0)
entry_path = tk.Entry(window, textvariable=path)
entry_path.grid(row=0, column=1)
entry_path.focus() #focus on this box

#checkboxes for what to plot
label_to_plot = tk.Label(text="Check the data you would like to plot:").grid(row=1, column=0)
raw_check = tk.Checkbutton(window, text = 'raw data', variable=raw_to_plot, onvalue=1, offvalue=0).grid(row=1, column=1) #assign to 1 if clicked
normal_check = tk.Checkbutton(window, text = 'normalized data', variable=normal_to_plot, onvalue=1, offvalue=0).grid(row=2, column=1) #assign to 1 if clicked
rolling_check = tk.Checkbutton(window, text = 'rolling data', variable=rolling_to_plot, onvalue=1, offvalue=0).grid(row=3, column=1) #assign to 1 if clicked

#For how many frames to roll
label_frames_to_roll = tk.Label(text="How many frames would you like to roll (Enter '0' if none):").grid(row=4, column=0)
entry_frames_to_roll = tk.Entry(window, textvariable=frames_to_roll)
entry_frames_to_roll.grid(row=4, column=1)

#prestim frames
label_pre_stim_frames = tk.Label(text="How many pre-stim frames:").grid(row=5, column=0)
entry_pre_stim_frames = tk.Entry(window, textvariable=pre_stim_frame)
entry_pre_stim_frames.grid(row=5, column=1)

cancel_button = tk.Button(text='Cancel', command=cancel_script).grid(row=6, column=0)
enter_button = tk.Button(text='Enter', command=get_values_close_window).grid(row=6, column=1)

window.mainloop()

######################################
##SET VARIABLES and ERRORS ###########
######################################

path_name = path_name.strip()

if path_name == '':
    print("ERROR: You did not enter a path_name!")
    sys.exit()
else:   
    os.chdir(path_name) #path to the movies

if rolling_plot_value == 1 and frames_to_roll_value == 0:
    print("ERROR: You cannot plot the rolling value if you do not want to calculate it!")
    sys.exit()
else:
    pass

######################################
########## PROCESSING ################
######################################
 
fileNames = [] #list for each of the file names
for file in glob.glob("*.tif"): #obtaining the name of each file that ends in ".tif" in the provided directory
    fileNames.append(file)
    fileNames.sort()
    
##### Below used to calculate the RAW average intensity for each frame in each movie 
raw = {} #dictionary to store the raw total intensity value per frame for each file
for file in fileNames: #iterate over each file in the list of file names
    movie = skio.imread(file) #read each movie
    movie = movie[:-1] #remove the last frame of each movie
    avg_pix_int_per_frame = () #tuple that will contain the average pixel intensity of each frame
    i = 0 
    
    ##### Below is used for selecting the edge of the cell (not perfect) by selecting the highest pixel values after 50 frames
    single_frame = cle.create([512, 512]);
    frame = 50;
    cle.copy_slice(movie, single_frame, frame) #creates an image of a single slice of the movie at frame = 50
    mask = single_frame > np.mean(single_frame)*2.5 #masked the image
    blurred = gaussian(mask, 5) #blurring that first mask
    edge = blurred > np.mean(blurred)*5 #masking again
    edge_eroded4 = binary_erosion(edge, disk(4)) #to help with binary erosion
    labeled_edge_eroded4 = label(edge_eroded4, connectivity=2) #for labelling the structures
    #cle.imshow(labeled_edge_eroded4, labels=True)


    edge_only = remove_small_objects(edge_eroded4, min_size=5000) #filtering out the smaller regions, like membrane folds
    #labeled_edge_only = label(edge_only, connectivity=2)
    #cle.imshow(labeled_edge_only, labels = True)
    
    for one_frame in movie: #iterate over each frame calc the average pixel intensity per frame; assign the tuple created above
        masked = one_frame * edge_only
        i += 1 #increasing frame number
        intensity = np.mean(masked) #calc the mean intensity
        avg_pix_int_per_frame = avg_pix_int_per_frame + (intensity,) #add to the tuple above
    raw[file] = avg_pix_int_per_frame #file name is the key; average pixel intensity per frame stored as tuple is the value

##### Below used to calculate the NORMALIZED average intensity for each frame in each movie 
normalized = {} #dictionary to store the normalized total intensity value per frame for each file
for profile in raw: #iterate over each movie profile and normalize to the first frame; store in new dict above
    pre_stim_value = sum(raw[profile][:pre_stim_frames_value])/pre_stim_frames_value #finding the average intensity level of the pre-stim frames for normalizing
    normal_int_profile = () #tuple to store he normalized average pixel intensity of each frame
    for raw_avg_int in raw[profile]: #iterate over the profile for each movie and normalize each frame
        norm_avg_int = raw_avg_int - pre_stim_value #normalize
        normal_int_profile = normal_int_profile + (norm_avg_int,) #new profile
    normalized[profile] = normal_int_profile #store normalized profile in new dict
        
##### Below used to calculate the ROLLING NORMALIZED average intensity for each frame in each movie 
if frames_to_roll_value == 0:
    print("You did not choose to roll any frames")
else:
    rolling_norm = {} #dictionary to contain the rolling normalized intensity profiles for each movie
    for profile in normalized: #for each intensity value in the tuple "value" of the dictinary
        rolling_int_profile = ()
        i = 0
        for norm_avg_int in normalized[profile]:
            i += 1
            rolling_total = norm_avg_int #used to calculate the rolling avg intensity 
            if i < pre_stim_frames_value + 1: #pre stim frames will not be rolled
                rolling_int_profile = rolling_int_profile + (rolling_total,)
                continue
            elif i < len(normalized[profile]) - frames_to_roll_value: # to keep the system in the correct range
                for w in range(frames_to_roll_value - 1): #for loop to add the correct number of frames to the running total
                    w += 1 
                    rolling_total = rolling_total + normalized[profile][i - 1 + w]
                rolling_average = rolling_total / frames_to_roll_value #taking the average of the selected frames
                rolling_int_profile = rolling_int_profile + (rolling_total,) #final rolling normalized average intensity values added to one tuple per movie
            else:
                break
        rolling_norm[profile] = rolling_int_profile # the tuple is added to the dictionary with the file name as the key
        

######################################
########## SAVE TO CSV ###############
######################################

##### Below creates three csv files: raw, normailzed, and rolling normalized profiles for each movie
with open('!raw_values.csv', 'w') as f:
    for key in raw.keys():
        f.write("%s,%s\n"%(key,raw[key]))

with open('!normalized_values.csv', 'w') as f:
    for key in normalized.keys():
        f.write("%s,%s\n"%(key,normalized[key]))

if frames_to_roll_value == 0:
    pass
else:
    with open(f'!rolling({roll_avg_frames})_normalized_values.csv', 'w') as f:
        for key in rolling_norm.keys():
            f.write("%s,%s\n"%(key,rolling_norm[key]))

######################################
########## PLOTTING ##################
######################################

##### Below is code for plotting the intensity over time for each profile on the same graph
def plot_profiles(dict_of_profiles):
    for movie in dict_of_profiles: #for each file in dictionary or normalized intensities
        i = 0
        One_movie_profile = {} #dictionairy to contain the normalized intensity profile for the one movie that that this for loop is currently working on
        for roll_norm_int in dict_of_profiles[movie]: #for each intensity value in the tuple "value" of the dictinary
            i += 1 #frame number
            One_movie_profile[i] = roll_norm_int #final rolling normalized average intensity value
        myList = One_movie_profile.items() #items() returns a view object. The view object contains the key-value pairs of the dictionary, as tuples in a list.
        x, y = zip(*myList) #zip(*iterables) --> A zip object yielding tuples until an input is exhausted.

        #smoothing the line 
        x = np.array(x) #converting to a numpy array
        y = np.array(y) #converting to a numpy array
        x_new = np.linspace(x.min(), x.max(), 100) #max is max, min is min, data avergaed into ______ bins
        bspline = scipy.interpolate.make_interp_spline(x, y)
        y_new = bspline(x_new)
        plt.plot(x_new, y_new, linewidth=2)

def make_plot(list_of_file_names, pre_stim_frames, what_to_plot):
    legend = ' '.join(list_of_file_names)
    legend_new = legend.replace('.tif', '')
    legend_new = legend_new.split(' ')

    plt.xlabel('Frames')
    plt.ylabel(f'pixel intensity')
    plt.title('Average flourescence over time')
    plt.legend(legend_new)
    plt.axvline(pre_stim_frames)
    plt.text(pre_stim_frames+.1,-1,'stimulation',rotation=0)
    plt.axvline(55)
    plt.text(pre_stim_frames+45.1,-1,'stop stimulation',rotation=0)
    font = {'family' : 'Times New Roman',
            'weight' : 'bold',
            'size'   : 10}

    plt.rc('font', **font)
    plt.savefig(f'!intensity_profiles_{what_to_plot}.png')

    plt.show()

plt.style.use(['dark_background'])
plt.rcParams["figure.figsize"] = (10,6)

if raw_plot_value == 1:
    plot_profiles(raw)
    make_plot(fileNames, pre_stim_frames_value, 'raw')
else:
    pass

if normal_plot_value == 1:
    plot_profiles(normalized)
    make_plot(fileNames, pre_stim_frames_value, 'normalized')
else:
    pass

if rolling_plot_value == 1:
    plot_profiles(rolling_norm)
    make_plot(fileNames, pre_stim_frames_value, 'rolling_norm')
else:
    pass
