import os
import subprocess

import numpy as np
import pandas as pd
import mask_analysis as ma
import time

# THIS SCRIPT MUST BE RUN FROM THE .BAT FILE TO WORK
ma.__time_print__('Input path to folder with .vsi files (the script will find all files in the given folder)')
data_path = str(input('Enter Directory: '))  # get data folder path
if data_path[-1] == '\\':  # remove end \ if present
    data_path = data_path[:-1]

# define image names from microscoper pre-processing
images = [r'\Multi Fluo.tif', r'\Green Ex.tif', r'\UV.tif']

# define output path from defined path
listed_path = data_path.split('\\')
if listed_path[-2][-1] != ':':  # if parent directory is not the hard drive location
    new_folder = listed_path[-2] + ' (processed)'
    listed_path[-2] = new_folder
else:
    new_folder = listed_path[-1] + ' (processed)'
    listed_path[-1] = new_folder
out_path = '\\'.join(listed_path)

# mine files from given data path
data_files = [i for i in os.listdir(data_path) if i.split('.')[-1] == 'vsi']
data_folders = [i.replace('.vsi', '') for i in data_files]
force_pro = []  # set empty force processing list for later use
ana_method = 0  # set initial analysis method to feature recognition

# progress check the script
# check out path validity (if no such file, progress must be zero)
if not os.path.exists(out_path):
    ma.__time_print__('CREATING OUTPUT DIRECTORY')
    print(f'---| {out_path}')
    os.makedirs(out_path)

# check whether .tif files have been created from .vsi files
for i in data_folders:
    vsi_dir = data_path + r'\_' + i + '_'
    for j in images:
        if not os.path.isfile(vsi_dir + j):
            ma.__time_print__('Data files are missing. Running microscoper script to generate missing files.')
            p = subprocess.Popen(rf'microscoper -f . -k {i}', cwd=data_path)
            p.wait()
            break

# perform preprocessing of images
angle_max_fields, fixed_best_angle = False, False
out_dirs, _sb = ma.preProcessImgs(data_folders, data_path, out_path, images, force_pro, ana_method, angle_max_fields,
                                     fixed_best_angle)

redo_check = True
while redo_check:
    ma.__time_print__('Image pre-processing is done. If re-processing of any images is '
          f'desired, enter the file numbers below separated by a ",", type in a range from one image to another '
          f'separated by a "-", or press ENTER to continue.')
    for id, e in enumerate(data_folders):
        print(f'[{id}]: {e}')
    redo_img = str(input('Enter File Numbers or Continue: '))
    if not redo_img:
        redo_check = False
        pass
    else:
        redo_list = []
        for i in redo_img.replace(' ', '').split(','):
            if '-' in i:
                i_split = i.split('-')
                i_range = list(range(int(i_split[0]), int(i_split[1]) + 1))
                redo_list += i_range
            else:
                redo_list.append(int(i))
        force_pro = [e for i, e in enumerate(data_folders) if i in redo_list]

        ma.__time_print__('Which method of recognition should be employed?')
        print('[0]: Large Feature Set Recognition')
        print(' >>>   [0.1]: Specify Max Fields for Angle Determination')
        if len(redo_list) == 1:
            print(' >>>   [0.2]: Specify Angle Offset')
            print(' >>>   [0.3]: Use Legacy LFSR')
        else:
            print(' >>>   [0.2]: Use Legacy LFSR')
        print('[1]: Bright Spot Recognition')
        print('[2]: Bright Spot Recognition (Forced Square)')
        ana_method = str(input('Enter Number: '))

        if ana_method == '0.1':
            ma.__time_print__('Specify the maximum number of fields the MOFM script should '
                  f'use to find the angular offset. The default is 6. Note that going higher will inevitably cause the '
                  f'found fields to be of worse quality, but quantity may prevail here.')
            angle_max_fields = int(input('Enter Max Field Number: ')) - 1
        if ana_method == '0.2':
            ma.__time_print__('Enter the angle offset in degrees. It is currently only '
                  f'possible to enter a single angle, so it is advised to only use this for a single image at a time.')
            fixed_best_angle = np.deg2rad(float(input('Enter Angle Offset: ')))
        if ana_method == '0.3':
            angle_max_fields = 0

        # perform preprocessing of images
        out_dirs, _sb = ma.preProcessImgs(data_folders, data_path, out_path, images, force_pro, ana_method,
                                          angle_max_fields, fixed_best_angle)

# perform processing of cut images
ma.processImgs(data_folders, out_dirs, out_path, _sb)
ma.__time_print__('Image processing is done. This window can be closed now.')
