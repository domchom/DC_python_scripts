import os
import sys
import math
import pathlib  
import fnmatch
import datetime                           
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
from tkinter import Tk
from tkinter import ttk
import skimage.io as skio
import scipy.signal as sig
from genericpath import exists           
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory
import timeit


np.seterr(divide='ignore', invalid='ignore')

'''*** Start GUI Window ***'''

#initiates Tk window
root = tk.Tk()
root.title('Select your options')
root.geometry('500x250')

#sets number of columns in the main window
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)

#defining variable types for the different widget fields
boxSizeVar = tk.IntVar()            #variable for box grid size
boxSizeVar.set(16)                  #set default value 
plotIndividualACFsVar = tk.BooleanVar()     #variable for plotting individual ACFs
plotIndividualCCFsVar = tk.BooleanVar()     #variable for plotting individual CCFs
plotIndividualPeaksVar = tk.BooleanVar()    #variable for plotting individual peaks
acfPeakPromVar = tk.DoubleVar()             #variable for peak prominance threshold   
acfPeakPromVar.set(0.1)                     #set default value
groupNamesVar = tk.StringVar()   #variable for group names list
folderPath = tk.StringVar()      #variable for path to images

#function for getting path to user's directory
def getFolderPath():
    folderSelected = askdirectory()
    folderPath.set(folderSelected)

#function for hitting cancel button or quitting
def on_quit(): 
    root.destroy() #destroys window
    sys.exit("You opted to cancel the script!")

#function for hitting start button
def on_start(): 
        root.destroy() #destroys window
    

'''widget creation'''
#file path selection widget
fileEntry = ttk.Entry(root, textvariable=folderPath)
fileEntry.grid(column=0, row=0, padx=10, sticky='E')
browseButton = ttk.Button(root, text= 'Select source directory', command=getFolderPath)
browseButton.grid(column=1, row=0, sticky='W')

#boxSize entry widget
boxSizeBox = ttk.Entry(root, width = 3, textvariable=boxSizeVar) #creates box widget
boxSizeBox.grid(column=0, row=1, padx=10, sticky='E') #places widget in frame
boxSizeBox.focus()      #focuses cursor in box
boxSizeBox.icursor(2)   #positions cursor after default input characters
ttk.Label(root, text='Enter grid box size (px)').grid(column=1, row=1, columnspan=2, padx=10, sticky='W') #create label text

#create acfpeakprom entry widget
ttk.Entry(root, width = 3, textvariable=acfPeakPromVar).grid(column=0, row=2, padx=10, sticky='E') #create the widget
ttk.Label(root, text='Enter ACF peak prominence threshold').grid(column=1, row=2, padx=10, sticky='W') #create label text

#create groupNames entry widget
ttk.Entry(root,textvariable=groupNamesVar).grid(column=0, row=3, padx=10, sticky='E') #create the widget
ttk.Label(root, text='Enter group names separated by commas').grid(column=1, row=3, padx=10, sticky='W') #create label text

#create checkbox widgets and labels
ttk.Checkbutton(root, variable=plotIndividualACFsVar).grid(column=0, row=5, sticky='E', padx=15)
ttk.Label(root, text='Plot individual ACFs').grid(column=1, row=5, columnspan=2, padx=10, sticky='W') #plot individual ACFs
ttk.Checkbutton(root, variable=plotIndividualCCFsVar).grid(column=0, row=6, sticky='E', padx=15) #plot individual CCFs
ttk.Label(root, text='Plot individual CCFs').grid(column=1, row=6, columnspan=2, padx=10, sticky='W')

ttk.Checkbutton(root, variable=plotIndividualPeaksVar).grid(column=0, row=7, sticky='E', padx=15) #plot individual peaks
ttk.Label(root, text='Plot individual peaks').grid(column=1, row=7, columnspan=2, padx=10, sticky='W')

#Creates the 'Start Analysis' button
startButton = ttk.Button(root, text='Start Analysis', command=on_start) #creates the button and bind it to close the window when clicked
startButton.grid(column=1, row=9, pady=10, sticky='W') #place it in the tk window

#Creates the 'Cancel' button
cancelButton = ttk.Button(root, text='Cancel', command=on_quit) #creates the button and bind it to on_quit function
cancelButton.grid(column=0, row=9, pady=10, sticky='E') #place it in the tk window

root.protocol("WM_DELETE_WINDOW", on_quit) #calls on_quit if the root window is x'd out.
root.mainloop() #run the script

#get the values stored in the widget
boxSizeInPx = boxSizeVar.get()
plotIndividualACFs= plotIndividualACFsVar.get()
plotIndividualCCFs = plotIndividualCCFsVar.get()
plotIndividualPeaks = plotIndividualPeaksVar.get()
acfPeakProm = acfPeakPromVar.get()
groupNames = groupNamesVar.get()
groupNames = [x.strip() for x in groupNames.split(',')] #list of group names. splits string input by commans and removes spaces
baseDirectory = folderPath.get() 

#make dictionary of parameters for log file use
logParams = {
    "Box Size(px)" : boxSizeInPx,
    "Base Directory" : baseDirectory,
    "ACF Peak Prominence" : acfPeakProm,
    "Group Names" : groupNames,
    "Plot Individual ACFs" : plotIndividualACFs,
    "Plot Individual CCFs" : plotIndividualCCFs,
    }

errors = []
if acfPeakProm > 1 :
    errors.append("The ACF peak prominence can not be greater than 1, set 'ACF peak prominence threshold' to a value between 0 and 1. More realistically, a value between 0 and 0.5")
if len(baseDirectory) < 1 :
    errors.append("You didn't enter a directory to analyze")

if len(errors) >= 1 :
    print("Error Log:")
    for count, error in enumerate(errors):
        print(count,":", error)
    sys.exit("Please fix errors and try again.") 

'''*** End GUI Window ***'''
