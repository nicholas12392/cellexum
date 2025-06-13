"""GUI for cellexum"""
import tkinter as tk
from tkinter import filedialog, messagebox
import time

import javabridge
import bioformats
from bioformats import logback
from PIL import Image, ImageTk, ImageOps
import os

import threading
import multiprocessing
import numpy as np
from fontTools.ttLib.tables.S_V_G_ import doc_index_entry_format_0
from matplotlib.style.core import available

import base
from base import directory_checker
from skeletons import *
import analysis
import cv2
import supports
import concurrent.futures
import shutil

graphics_folder = rf'C:\Users\nicho\OneDrive - Aarhus universitet\6SEM\Bachelor\NTSA_PROCESSING\graphics\vector'

active_frames = []  # empty list to store active frames for toggling

class FileSelection(WindowFrame):
    """Frame which allows user to select files for processing."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # define dependent variables and load widget
        self.dv_define('InputFolder', tk.StringVar(self, ''))
        self.dv_define('OutputFolder', tk.StringVar(self, ''))
        self.dv_define('SelectedFiles', JSONVar(self, value={}))
        self.load()

    def __base__(self):
        self.add(AppTitle, text='Image File Selection')
        self.add(AppLabel, text='Enter image directory:')
        _ = self.add(DirectoryEntry, sid='InputFolder')
        _.trace_add('write', self.__load_directory); _.trace_add('write', self.__post_available_channels)

        self.add(TextButton, text='Select All', command=self.__select_all)
        self.add(TextButton, text='Deselect All', command=self.__deselect_all, padx=(130, 0), prow=True)

        self.add(AppFrame, sid='FileSelectionContainer')  # construct container for files in directory

        active_frames.append(self)

    def reload(self):
        """Change the reload functionality to simply reset the input folder."""
        self['DirectoryEntryInputFolder'].set('')

    def __select_all(self):
        for elem in self.containers['AppFrameFileSelectionContainer']:
            self[elem].set(True)

    def __deselect_all(self):
        for elem in self.containers['AppFrameFileSelectionContainer']:
            self[elem].set(False)

    def __load_directory(self, *_):
        """Internal method that loads the current selected directory .vsi files."""
        selected_directory = self['DirectoryEntryInputFolder'].get()
        if selected_directory == self.dv_get('InputFolder'):
            pass  # if the chosen folder is the current folder do nothing
        elif selected_directory == '':
            self.dv_set('OutputFolder', '')  # set default output directory
            self.dv_set('InputFolder', '')  # set input directory
            self.container_drop('AppFrameFileSelectionContainer')  # remove existing files in the container

            # store input folder in cache file
            supports.json_dict_push(rf'{supports.__cache__}\settings.json', behavior='update',
                                    params={'DirectorySettings': {
                                        'InputFolder': selected_directory,
                                        'OutputFolder': selected_directory + ' (processed)'
                                    }})
            self.__update_selection()  # update selection after setting up check buttons
        else:
            """Since the PreprocessingOptions reload is triggered by InputFolder trace, the OutputFolder must be 
            changed before the InputFolder, to avoid reloading the ProcessingOptions with the wrong InputFolder."""
            self.dv_set('OutputFolder', selected_directory + ' (processed)')  # set default output directory
            self.dv_set('InputFolder', selected_directory)  # set input directory
            supports.tprint(f'Set default output folder to: {selected_directory} (processed)')
            self.container_drop('AppFrameFileSelectionContainer')  # remove existing files in the container
            file_folders = [i.removesuffix('.vsi') for i in os.listdir(selected_directory) if i.endswith('.vsi')]

            for folder in supports.sort(file_folders):
                elem = self.add(AppCheckbutton, text=folder, container='AppFrameFileSelectionContainer')
                elem.trace_add('write', self.__update_selection)

            # store input folder in cache file
            supports.json_dict_push(rf'{supports.__cache__}\settings.json', behavior='update',
                                params={'DirectorySettings': {
                                    'InputFolder': selected_directory,
                                    'OutputFolder': selected_directory + ' (processed)'
                                }})
            self.__update_selection()  # update selection after setting up check buttons

    def __update_selection(self, *_):
        """Internal method that updates the selected files dependent variable."""
        try:
            _ = {}
            for elem in self.containers['AppFrameFileSelectionContainer']:
                _[self[elem].text] = self[elem].get()
            self.dv_set('SelectedFiles', _)
        except KeyError:
            supports.tprint('There exists no .vsi files at %s' % self.dv_get('InputFolder'))

    @supports.thread_daemon
    def __post_available_channels(self, *_):
        """Internal method that posts the available color channels as a global variable."""

        saves = supports.json_dict_push(rf'{supports.__cache__}\saves.json', behavior='read')
        try:  # only iterate if no saved channels can be found for that data set
            available_channels = saves[self.dv_get('InputFolder')]['AvailableChannels']
            if not available_channels:
                raise KeyError
        except KeyError:
            self.dv_set('AvailableChannels', [])  # ensure that channels are blocked, until new are found
            _in, _out = multiprocessing.Pipe(duplex=False)  # open pipe
            process = multiprocessing.Process(target=self.channel_search, args=(self.dv_get('InputFolder'), _out),
                                              daemon=True)
            process.start()
            process.join()
            available_channels = _in.recv()
            process.close()

            update_dict = {self.dv_get('InputFolder'): {
                'AvailableChannels': available_channels,
            }}
            supports.json_dict_push(rf'{supports.__cache__}\saves.json', params=update_dict, behavior='update')
        self.dv_set('AvailableChannels', available_channels)


    @staticmethod
    def channel_search(input_folder, _out):
        if input_folder == '': return  # quit if the input folder was reset
        files = [i.removesuffix('.vsi') for i in os.listdir(input_folder) if i.endswith('.vsi')]
        javabridge.start_vm(class_path=bioformats.JARS); logback.basic_config()
        available_channels = []
        for file in files:
            try:
                metadata = bioformats.omexml.OMEXML(bioformats.get_omexml_metadata(
                    r'{}\{}.vsi'.format(input_folder, file)))
                pix = base.OMENavigator(metadata).n('Image').n('Pixels')
                for c in range(int(pix['SizeC'])):
                    try:
                        channel = pix.n('Channel', c)['Name']
                        if channel not in available_channels and channel is not None:
                            available_channels.append(channel)
                    except IndexError:
                        pass
            except Exception as e:
                supports.tprint(f'Failed to collect channel data for {file} with exit: {e!r}')
        javabridge.kill_vm()
        _out.send(available_channels)


class MultiFieldPreprocessingOptions(FieldProcessingFrame):
    def __init__(self, parent, *args, **kwargs):
        kwargs['padx'] = 0; kwargs['pady'] = 0
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

    def __base__(self):
        self.add(AppSubtitle, text='Global Options')
        self.add(AppLabel, text='Image mask:', sid='ImageMask', tooltip=True)

        # fetch masks in the __masks__ folder
        mask_data = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json', behavior='read')
        if self.parent['SelectionMenuSampleType'].get() in mask_data:
            mask_data = mask_data[self.parent['SelectionMenuSampleType'].get()]
        else:
            mask_data = {}
        _ = list(mask_data.keys()) + ['Add ...']   # set up keys

        elem = self.add(SelectionMenu, sid='ImageMask', options=_, prow=True, padx=(120, 0), width=20,
                            commands={'Add ...': self.__add_mask})
        self.dv_define('ImageMask', elem.selection)
        self.add(TextButton, text='Create Orientation Reference', sid='COR',
                     command=self.__load_orientation_reference_creator)
        elem.trace_add('write', self.__show_orc_button)
        elem.trace_add('write', self.__update_image_mask_cache)

        self.add(AppSubtitle, text='Image Settings', tooltip=True)
        self.add(TextButton, sid='ResetImageSettingsTable', text='Reset', command=self.restore_default, warning=True,
                 prow=True, padx=(170, 0), pady=(42, 5))
        self.add(AppFrame, sid='ImageSettingsTable')  # table container
        self.add(TextButton, sid='LoadUnPreprocessed', text='Select Unpreprocessed', command=self.select_missing,
                 container='AppFrameImageSettingsTable')
        self.add(AppLabel, text='Rotate', sid='Rotate', prow=True, column=1, padx=5,
                 container='AppFrameImageSettingsTable', tooltip=True)
        self.add(AppLabel, text='Align', sid='Align', prow=True, column=2, container='AppFrameImageSettingsTable',
                   padx=5, tooltip=True)
        self.add(AppLabel, text='Fields', sid='MinFields', prow=True, column=3, container='AppFrameImageSettingsTable',
                 padx=5, tooltip=True)
        self.add(AppLabel, text='Masking Method', sid='MaskingMethod', prow=True, column=4,
                   container='AppFrameImageSettingsTable', padx=5, tooltip=True)
        self.add(AppLabel, text='Mask Shift', sid='MaskShift', prow=True, column=5,
                 container='AppFrameImageSettingsTable', padx=5, tooltip=True)
        self.add(AppLabel, text='Fidelity', sid='IterativeFidelity', prow=True, column=6,
                 container='AppFrameImageSettingsTable', padx=5, tooltip=True)
        self.add(AppLabel, text='Span', sid='OnsetIntensitySpan', prow=True, column=7,
                 container='AppFrameImageSettingsTable', padx=5, tooltip=True)
        self.update_image_settings()
        self.add(TextButton, sid='LoadPreviousSettings', text='Load Previous Preprocessing', warning=True,
                     command=self.load_previous_settings, container='AppFrameImageSettingsTable')

        self.add(AppButton, text='PREPROCESS', command=self.preprocess)

        self.parent['SelectionMenuMaskChannel'].selection.trace_add('write', self.__show_orc_button)

        self.__show_orc_button()

    def __load_orientation_reference_creator(self):
        # prior to opening the OrientationReferenceCreator the MaskChannel must be saved to the Settings.json
        _ = {'CollectionPreprocessSettings': {
            'MaskChannel': self.parent['SelectionMenuMaskChannel'].get(),
            'SampleType': self.parent['SelectionMenuSampleType'].get(),
            'MaskSelection': self['SelectionMenuImageMask'].get()
        }}

        base.directory_checker(self.dv_get('OutputFolder'), clean=False)
        supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), params=_, behavior='update')

        level = TopLevelWidget(self); level.title('Orientation Reference Creator')
        level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')

        content = OrientationReferenceCreator(level.main.interior, name='OrientationReferenceCreator', tie=self)
        content.grid()

    def add_table_entry(self, file):
        _ = self.add(TextCheckbutton, text=file, sid=f':{file}', fg=supports.__cp__['fg'], font='Arial 12',
                     container='AppFrameImageSettingsTable', padx=(0, 15), tag='ImageSelection')
        _.bind('<Control-1>', self.update_all)

        _ = self.add(SelectionMenu, sid=rf'Rotate:{file}', prow=True, column=1, width=6,
                     font='Arial 12', bg=supports.__cp__['dark_bg'], container='AppFrameImageSettingsTable',
                     default=0, options=('Auto', '0', '90', '180', '270'), padx=5, group='ImageSettingsTable')
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.label_bind('<Control-1>', self.update_all)

        _ = self.add(SettingEntry, sid=rf'Align:{file}', prow=True, column=2, width=5, vartype=float,
                     container='AppFrameImageSettingsTable', default='', padx=5, group='ImageSettingsTable')
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.bind('<Control-1>', self.update_all)

        _ = self.add(SettingEntry, sid=rf'MinFields:{file}', prow=True, column=3, width=5, padx=5,
                     container='AppFrameImageSettingsTable', default=6, vartype=int, group='ImageSettingsTable')
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.bind('<Control-1>', self.update_all)

        _ = self.add(SelectionMenu, sid=rf'MaskingMethod:{file}', prow=True, column=4, width=13, padx=5,
                     container='AppFrameImageSettingsTable', default=0, options=('Calculate', 'Hybrid'),
                     group='ImageSettingsTable')
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.label_bind('<Control-1>', self.update_all)

        _ = self.add(SelectionMenu, sid=rf'MaskShift:{file}', prow=True, column=5, width=8, padx=5,
                     container='AppFrameImageSettingsTable', default=0, options=('Auto', 'None'),
                     group='ImageSettingsTable')
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.label_bind('<Control-1>', self.update_all)
        
        _ = self.add(SettingEntry, sid=f'IterativeFidelity:{file}', prow=True, column=6, width=6, padx=5,
                     container='AppFrameImageSettingsTable', default=3, group='ImageSettingsTable', vartype=int)
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.bind('<Control-1>', self.update_all)

        _ = self.add(SettingEntry, sid=f'OnsetIntensitySpan:{file}', prow=True, column=7, width=4, padx=5,
                     container='AppFrameImageSettingsTable', default=10, group='ImageSettingsTable', vartype=int)
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.bind('<Control-1>', self.update_all)

    def hide_table_entry(self, file):
        for name in (f'TextCheckbutton:{file}', f'SelectionMenuRotate:{file}', f'SettingEntryAlign:{file}',
                     f'SelectionMenuMaskingMethod:{file}', f'SettingEntryMinFields:{file}',
                     f'SelectionMenuMaskingMethod:{file}', f'SelectionMenuMaskShift:{file}',
                     f'SettingEntryIterativeFidelity:{file}', f'SettingEntryOnsetIntensitySpan:{file}'):
            if self.exists(name):
                self[name].grid_remove()

    def show_table_entry(self, file):
        for name in (f'TextCheckbutton:{file}', f'SelectionMenuRotate:{file}', f'SettingEntryAlign:{file}',
                     f'SelectionMenuMaskingMethod:{file}', f'SettingEntryMinFields:{file}',
                     f'SelectionMenuMaskingMethod:{file}', f'SelectionMenuMaskShift:{file}',
                     f'SettingEntryIterativeFidelity:{file}', f'SettingEntryOnsetIntensitySpan:{file}'):
            if self.exists(name):
                self[name].grid()
            else:
                self.add_table_entry(file)

    def __update_image_mask_cache(self, *_):
        value = self['SelectionMenuImageMask'].get()
        if value not in ('Select Option', 'Add ...'):
            supports.post_cache({'PreprocessingSettings': {'ImageMask': value}})

    def load_previous_settings(self):
        """Method that loads the settings used during latest preprocessing."""
        settings = supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), behavior='read')
        if settings['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            files = list(settings['PreprocessedParameters'].keys())
            _ = []
            for file in self.tags['ImageSelection']:  # set activity status
                name = file.split(':')[-1]
                if name in files:
                    self[file].set(True)
                    for k, j in zip(('Rotate', 'MinFields', 'Align', 'MaskShift', 'MaskingMethod', 'IterativeFidelity',
                                     'OnsetIntensitySpan'),
                                    ('SelectionMenu', 'SettingEntry', 'SettingEntry', 'SelectionMenu', 'SelectionMenu',
                                     'SettingEntry', 'SettingEntry')):
                        self[f'{j}{k}:{name}'].set(settings['PreprocessedParameters'][name]['FieldParameters'][k])
                else:
                    self[file].set(False)
            self['SelectionMenuImageMask'].set(settings['CollectionPreprocessSettings']['MaskSelection'])
            self.parent['SelectionMenuMaskChannel'].set(settings['CollectionPreprocessSettings']['MaskChannel'])
        else:
            supports.tprint('Previous settings have the wrong formatting.')

    def __show_orc_button(self, *_):
        """Internal method that shows and hides the orientation reference creator button."""
        if (self['SelectionMenuImageMask'].get() not in ('Select Option', 'Add ...') and
                self.parent['SelectionMenuMaskChannel'].get() not in ('Select Option', 'Loading')):
            self['TextButtonCOR'].grid()
        else:
            self['TextButtonCOR'].grid_remove()

    def __add_mask(self):
        level = TopLevelWidget(self); level.title('Mask Creator')
        level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')
        content = MultiFieldMaskCreator(level.main.interior, tie=self); content.grid()

    def preprocess(self):
        """Internal method that runs NTSA preprocessing according to the selected settings."""
        if self.preprocess_check() is False: return

        mask_data = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                            behavior='read')[self.parent['SelectionMenuSampleType'].get()]
        if 'OrientationReference' not in mask_data[self['SelectionMenuImageMask'].get()]:
            for tag in self.tags['ImageSelection']:
                file = tag.split(':')[-1]
                if self[f'SelectionMenuRotate:{file}'].get() == 'Auto' and self[f'TextCheckbutton:{file}'].get() is True:
                    messagebox.showerror('Error', "Cannot use Rotate 'Auto' with no defined orientation reference.",)
                    return
        else:
            if not os.path.exists(r'{}\_misc\OrientationReference.tiff'.format(self.dv_get('OutputFolder'))):
                supports.tprint('Orientation reference for mask {} is configured with {} from a different sample set.'.format(
                    self['SelectionMenuImageMask'].get(),
                    mask_data[self['SelectionMenuImageMask'].get()]['OrientationReference']['SampleName']))

        # extract settings and update settings json
        _ = {}
        for file in self.files:
            _[file] = {
                'Rotate': self[f'SelectionMenuRotate:{file}'].get(),
                'Align': self[f'SettingEntryAlign:{file}'].get(),
                'MinFields': self[f'SettingEntryMinFields:{file}'].get(),
                'MaskingMethod': self[f'SelectionMenuMaskingMethod:{file}'].get(),
                'MaskShift': self[f'SelectionMenuMaskShift:{file}'].get(),
                'IterativeFidelity': self[f'SettingEntryIterativeFidelity:{file}'].get(),
                'OnsetIntensitySpan': self[f'SettingEntryOnsetIntensitySpan:{file}'].get(),
            }

        update_dict = {
            'DirectorySettings': {
                'InputFolder': self.dv_get('InputFolder'),
                'OutputFolder': self.dv_get('OutputFolder')},
            'CollectionPreprocessSettings': {
                'MaskSelection': self['SelectionMenuImageMask'].get(),
                'MaskChannel': self.parent['SelectionMenuMaskChannel'].get(),
                'SampleType': self.parent['SelectionMenuSampleType'].get()},
            'IndividualPreprocessSettings': _
        }

        # create output folder if it does not exist
        if not os.path.isdir(self.dv_get('OutputFolder')):
            os.makedirs(self.dv_get('OutputFolder'))
            supports.tprint('Output folder created at: {}'.format(self.dv_get('OutputFolder')))

        supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), update_dict, 'update')

        supports.tprint('Started preprocessing.')
        self.preprocess_daemon()


class SingleFieldMaskCreator(FieldMaskCreator):
    """Class that constructs a mask creator for the cellexum application. In essence, this is a sub-application that
    allows for unique mask design, that can be utilized by the application."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Field Settings', pady=(0, 5))
        self.add(AppLabel, text='Field dimensions (w, h):')
        self.add(SettingEntry, sid='FieldWidth', prow=True, padx=(200, 0), vartype=int)
        self.add(AppLabel, text='x', sid='FieldDimensionTimes', prow=True, padx=(260, 0))
        self.add(SettingEntry, sid='FieldHeight', prow=True, padx=(280, 0), vartype=int)

        self.add(AppLabel, text='Field units:')
        self.add(SelectionMenu, sid='FieldUnits', prow=True, padx=(200, 0), options=('pix', 'µm', 'mm'))

        self.add(AppLabel, text='Mask name:', sid='MaskName', pady=5)
        self.add(SettingEntry, sid='MaskName', prow=True, padx=(100, 0), width=25, pady=5)
        self.add(AppButton, text='Save Mask', command=self.__save_mask, pady=(5, 0))
        self.add(AppButton, text='Cancel', command=self.cancel_window, pady=(5, 0), padx=(130, 0), prow=True)

    def __save_mask(self):
        """Internal method that saves a created mask to be used in the Cellexum application."""
        json_path = rf'{supports.__cwd__}\__misc__\masks.json'
        mask_name = self['SettingEntryMaskName'].get()

        try:  # fetch mask entries for data type
            mask_json = supports.json_dict_push(json_path, behavior='read')[self.sample_type]
        except KeyError:
            mask_json = {}

        if mask_name in mask_json:
            prompt = messagebox.askokcancel('Mask name already exists',
                                            message=f'A mask already exists with name {mask_name!r} for data type '
                                                    f'{self.sample_type!r}. Proceeding will overwrite it. Do you want '
                                                    f'to continue?')
            if prompt is not True:
                return

        # add mask data to the mask file
        _ = {self.sample_type: {mask_name: {
            'FieldWidth': self['SettingEntryFieldWidth'].get(),
            'FieldHeight': self['SettingEntryFieldWidth'].get(),
            'FieldUnits': self['SelectionMenuFieldUnits'].get()
        }}}

        supports.json_dict_push(json_path, _, behavior='update')

        # at last update visible mask options
        if mask_name not in mask_json:  # add option to available options if it does not already exist
            self.tie['SelectionMenuImageMask'].add_option(mask_name, order=-1)
        self.tie['SelectionMenuImageMask'].set(mask_name)  # set added option as selection
        self.cancel_window()  # destroy mask selector


class SingleFieldPreprocessingOptions(FieldProcessingFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

    def __base__(self):
        self.add(AppSubtitle, text='Global Options')
        self.add(AppLabel, text='Image mask:', sid='ImageMask')
        mask_data = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json', behavior='read')
        if self.parent['SelectionMenuSampleType'].get() in mask_data:
            mask_data = mask_data[self.parent['SelectionMenuSampleType'].get()]
        else:
            mask_data = {}
        _ = list(mask_data.keys()) + ['Add ...']  # set up keys

        elem = self.add(SelectionMenu, sid='ImageMask', options=_, prow=True, padx=(120, 0), width=20,
                        commands={'Add ...': self.__add_mask})
        self.dv_define('ImageMask', elem.selection)
        elem.trace_add('write', self.update_image_mask_cache)

        self.add(AppSubtitle, text='Image Settings', tooltip=True)
        self.add(TextButton, sid='ResetImageSettingsTable', text='Reset', command=self.restore_default, warning=True,
                 prow=True, padx=(170, 0), pady=(42, 5))
        self.add(AppFrame, sid='ImageSettingsTable')  # table container
        self.add(TextButton, sid='LoadUnPreprocessed', text='Select Unpreprocessed', command=self.select_missing,
                 container='AppFrameImageSettingsTable')
        self.add(AppLabel, text='Align', sid='Align', prow=True, column=1, container='AppFrameImageSettingsTable',
                 padx=5, tooltip=True)
        self.add(AppLabel, text='Masking Method', sid='MaskingMethod', prow=True, column=2,
                 container='AppFrameImageSettingsTable', padx=5, tooltip=True)
        self.add(AppLabel, text='Fidelity', sid='IterativeFidelity', prow=True, column=3,
                 container='AppFrameImageSettingsTable', padx=5, tooltip=True)
        self.update_image_settings()
        self.add(TextButton, sid='LoadPreviousSettings', text='Load Previous Preprocessing', warning=True,
                 command=self.load_previous_settings, container='AppFrameImageSettingsTable')

        self.add(AppButton, text='PREPROCESS', command=self.preprocess)

    def add_table_entry(self, file):
        self.add(TextCheckbutton, text=file, sid=f':{file}', fg=supports.__cp__['fg'], font='Arial 12',
                 container='AppFrameImageSettingsTable', padx=(0, 15), tag='ImageSelection')

        _ = self.add(SettingEntry, sid=rf'Align:{file}', prow=True, column=1, width=5, vartype=float,
                     container='AppFrameImageSettingsTable', default='', padx=5, group='ImageSettingsTable')
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.bind('<Control-1>', self.update_all)

        _ = self.add(SelectionMenu, sid=rf'MaskingMethod:{file}', prow=True, column=2, width=14, padx=5,
                     container='AppFrameImageSettingsTable', default=0, group='ImageSettingsTable',
                     options=('Contour', 'Fixed'))
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.label_bind('<Control-1>', self.update_all)

        _ = self.add(SettingEntry, sid=rf'IterativeFidelity:{file}', prow=True, column=3, width=6, vartype=int,
                     container='AppFrameImageSettingsTable', default=4, padx=5, group='ImageSettingsTable')
        _.tether(self[f'TextCheckbutton:{file}'], False, action=_.tether_action)
        _.bind('<Control-1>', self.update_all)

    def hide_table_entry(self, file):
        for name in (f'TextCheckbutton:{file}', f'SettingEntryAlign:{file}', f'SelectionMenuMaskingMethod:{file}',
                     f'SettingEntryIterativeFidelity:{file}'):
            if self.exists(name):
                self[name].grid_remove()

    def show_table_entry(self, file):
        for name in (f'TextCheckbutton:{file}', f'SettingEntryAlign:{file}', f'SelectionMenuMaskingMethod:{file}',
                     f'SettingEntryIterativeFidelity:{file}'):
            if self.exists(name):
                self[name].grid()
            else:
                self.add_table_entry(file)

    def __add_mask(self):
        level = TopLevelWidget(self); level.title('Mask Creator')
        level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')
        content = SingleFieldMaskCreator(level.main.interior, tie=self); content.grid()

    def load_previous_settings(self):
        """Method that loads the settings used during latest preprocessing."""
        settings = supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), behavior='read')
        if settings['CollectionPreprocessSettings']['SampleType'] == 'Single-Field':
            files = list(settings['PreprocessedParameters'].keys())
            _ = []
            for file in self.tags['ImageSelection']:  # set activity status
                name = file.split(':')[-1]
                if name in files:
                    self[file].set(True)
                    for k, j in zip(('Align', 'MaskingMethod', 'IterativeFidelity'),
                                    ('SettingEntry', 'SelectionMenu', 'SettingEntry')):
                        self[f'{j}{k}:{name}'].set(settings['PreprocessedParameters'][name]['FieldParameters'][k])
                else:
                    self[file].set(False)
            self['SelectionMenuImageMask'].set(settings['CollectionPreprocessSettings']['MaskSelection'])
            self.parent['SelectionMenuMaskChannel'].set(settings['CollectionPreprocessSettings']['MaskChannel'])
        else:
            supports.tprint('Previous settings have the wrong formatting.')

    def preprocess(self):
        """Internal method that runs NTSA preprocessing according to the selected settings."""
        if self.preprocess_check() is False: return

        # extract settings and update settings json
        _ = {}
        for file in self.files:
            _[file] = {
                'Align': self[f'SettingEntryAlign:{file}'].get(),
                'MaskingMethod': self[f'SelectionMenuMaskingMethod:{file}'].get(),
                'IterativeFidelity': self[f'SettingEntryIterativeFidelity:{file}'].get()
            }

        update_dict = {
            'DirectorySettings': {
                'InputFolder': self.dv_get('InputFolder'),
                'OutputFolder': self.dv_get('OutputFolder')},
            'CollectionPreprocessSettings': {
                'MaskSelection': self['SelectionMenuImageMask'].get(),
                'MaskChannel': self.parent['SelectionMenuMaskChannel'].get(),
                'SampleType': self.parent['SelectionMenuSampleType'].get()},
            'IndividualPreprocessSettings': _
        }

        # create output folder if it does not exist
        if not os.path.isdir(self.dv_get('OutputFolder')):
            os.makedirs(self.dv_get('OutputFolder'))
            supports.tprint('Output folder created at: {}'.format(self.dv_get('OutputFolder')))

        supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), update_dict, 'update')

        supports.tprint('Started preprocessing.')
        self.preprocess_daemon()


class ZeroFieldPreprocessingOptions(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        kwargs['padx'] = 0; kwargs['pady'] = 0
        super().__init__(parent, *args, **kwargs)

    def __base__(self):
        self.add(AppButton, text='PREPROCESS', command=self.preprocess)

    def preprocess(self):
        """Internal method that runs NTSA preprocessing according to the selected settings."""
        # extract settings and update settings json
        update_dict = {
            'DirectorySettings': {
                'InputFolder': self.dv_get('InputFolder'),
                'OutputFolder': self.dv_get('OutputFolder')},
            'CollectionPreprocessSettings': {
                'SampleType': self.parent['SelectionMenuSampleType'].get(),
                'MaskChannel': self.parent['SelectionMenuMaskChannel'].get()}
        }

        # create output folder if it does not exist
        if not os.path.isdir(self.dv_get('OutputFolder')):
            os.makedirs(self.dv_get('OutputFolder'))
            supports.tprint('Output folder created at: {}'.format(self.dv_get('OutputFolder')))

        supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), update_dict, 'update')

        supports.tprint('Started preprocessing.')
        self.preprocess_daemon()

    @supports.thread_daemon
    def preprocess_daemon(self):
        files = [file for file, state in self.dv_get('SelectedFiles').items() if state is True]  # get active files
        base.RawImageHandler().handle(files)  # handle all images before preprocessing

        with concurrent.futures.ProcessPoolExecutor(max_workers=supports.get_max_cpu(),
                                                    mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = {executor.submit(base.PreprocessingHandler().preprocess, file): file for file in files}
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                try:
                    supports.tprint(f'Preprocessed image {future.result()}.')
                    time.sleep(.5)  # avoid overlapping instances that may overload the GUI modules
                    self.dv_set('LatestPreprocessedFile', future.result())
                except Exception as exc:
                    supports.tprint('Failed to preprocess {} with exit: {!r}'.format(file, exc))
                    if self.dv_get('Debugger') is True:
                        raise exc
        supports.tprint('Completed all preprocessing.')


class PreprocessingOptions(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

    def __traces__(self):
        self.dv_trace('InputFolder', 'write', self.reload)
        self.dv_trace('AvailableChannels', 'write', self.__available_channel_listener)

    def __base__(self):
        self.add(AppTitle, text='Preprocessing')
        self.add(AppLabel, text='Output folder:')
        _ = self.add(DirectoryEntry, sid='OutputFolder', forbidden=(self.dv_get('InputFolder'), ))
        _.set(self.dv_get('OutputFolder'))
        _.trace_add('write', self.__on_dir_change)
        self.add(AppLabel, text='Sample type:', sid='SampleType', tooltip=True)
        _ = self.add(SelectionMenu, sid='SampleType', options=('Multi-Field', 'Single-Field', 'Zero-Field'), prow=True,
                     padx=(120, 0), width=20)
        _.trace_add('write', self.update_container_choice)
        self.add(AppLabel, text='Mask channel:', sid='MaskChannel', tooltip=True)

        _ = self.dv_get('AvailableChannels')
        elem = self.add(SelectionMenu, sid='MaskChannel', options=_, prow=True, padx=(120, 0), width=20)
        self.add(LoadingCircle, size=24, width=6, bg=supports.__cp__['bg'], aai=4, stepsize=.7, sid='MaskChannel',
                       prow=True, padx=(310, 0), pady=(0, 5))
        if not _:
            elem.disable('Loading')
            self['LoadingCircleMaskChannel'].start()

        self.add(MultiFieldPreprocessingOptions, sid='Container')
        self.add(SingleFieldPreprocessingOptions, sid='Container')
        self.add(ZeroFieldPreprocessingOptions, sid='Container')

        self.update_container_choice()

    def update_container_choice(self, *_):
        value = self['SelectionMenuSampleType'].get()
        for st in ('Multi', 'Single', 'Zero'):
            self[f'{st}FieldPreprocessingOptionsContainer'].grid_remove()

        if value != 'Select Option':  # update cache on option change
            supports.post_cache({'PreprocessingSettings': {'SampleType': value}})

            value = value.replace('-', '')
            self[f'{value}PreprocessingOptionsContainer'].load()
            if value != 'ZeroField':
                self[f'{value}PreprocessingOptionsContainer'].update_image_settings()  # prepare image settings for swap
            self[f'{value}PreprocessingOptionsContainer'].grid()

    def __available_channel_listener(self, *_):
        if self['SelectionMenuMaskChannel']['state'] == 'disabled':
            self['SelectionMenuMaskChannel'].enable()
            self['LoadingCircleMaskChannel'].stop()
        elif not self.dv_get('AvailableChannels'):
            self['SelectionMenuMaskChannel'].disable()
            self['LoadingCircleMaskChannel'].start()
        self['SelectionMenuMaskChannel'].update_options(self.dv_get('AvailableChannels'))

    def __on_dir_change(self, *_):
        """Internal method that updates local cache for the output folder."""
        self.dv_set('OutputFolder', self['DirectoryEntryOutputFolder'].get())
        supports.json_dict_push(rf'{supports.__cache__}\settings.json',
                            {'DirectorySettings': {'OutputFolder': self.dv_get('OutputFolder')}})


class MaskGallery(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.mask_folder = None
        self.settings = None
        self.selected_files = None

    def __traces__(self):
        self.dv_trace('LatestPreprocessedFile', 'write', self.__auto_load_image)
        self.dv_trace('InputFolder', 'write', self.clear)
        self.dv_trace('OutputFolder', 'write', self.clear)

    def __base__(self):
        self.mask_folder = mask_folder = r'{}\_masks for manual control'.format(self.dv_get('OutputFolder'))
        self.selected_files = [k for k, v in self.dv_get('SelectedFiles').items() if v is True]

        self.add(AppTitle, text='Mask Gallery')
        self.add(AppFrame, sid='ImageContainer')

        # construct placeholder containers
        for file in supports.sort(self.selected_files):
            self.add(AppFrame, sid=f':{file}:Container', container='AppFrameImageContainer', pady=(0, 10))

        # load existing masks upon initial tab load
        if os.path.exists(mask_folder):
            for file in supports.sort(self.selected_files):
                if os.path.isfile(rf'{mask_folder}/{file}.png'):
                    self.load_image(file)

    def __auto_load_image(self, *_):
        """Internal method that automatically loads the latest preprocessed mask image into the gallery. The image is
        only loaded if the __base__ has been called prior to calling this method."""
        if self.is_empty() is False:
            image = self.dv_get('LatestPreprocessedFile')
            self.load_image(image)

    @supports.thread_daemon
    def load_image(self, image):
        self.settings = supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')),
                                                behavior='read')
        self.container_drop(f'AppFrame:{image}:Container')  # drop image container contents
        self.container_drop(f'AppFrame:{image}:Metadata')  # drop metadata container contents
        mask_path = rf'{self.mask_folder}\{image}.png'

        # place image in container and store to memory
        _ = self.add(ZoomImageFrame, sid=f':{image}', container=f'AppFrame:{image}:Container')
        _.scroll_scalar = (.75, .75)
        _.set_image(path=mask_path)

        # place metadata in a container next to the image
        self.add(AppFrame, sid=f':{image}:Metadata', container=f'AppFrame:{image}:Container',
                 prow=True, column=1, sticky='n', padx=(20, 0))
        self.add(AppSubtitle, text=image, sid=f':{image}', container=f'AppFrame:{image}:Metadata',
                 pady=(0, 15))

        try:
            md = self.settings['PreprocessedParameters'][image]
            _l1 = ('Quality: Tier {Level:.2f} with Area Deviation {AreaError:.1f}%, and Ratio Deviation '
                   '{RatioError:.3f}%\n\n').format(**md['QualityParameters'])
            _l1 += 'Field Size: ({}, {}), Align Angle: {}°, Scale: {} pix:µm\n\n'.format(
                int(round(md['FieldParameters']['Width'], 0)),
                int(round(md['FieldParameters']['Height'], 0)), round(md['FieldParameters']['Align'], 3),
                round(md['FieldParameters']['ScaleBar'], 3))
            if self.settings['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
                _l1 += ('Masked with {MaskingMethod!r} and Rotated {Rotate}° with Certainty '
                        '{RotationCertainty:.1f} and Matrix Uniformity {ComparedMatrixUniformity:.3f}').format(
                    **md['FieldParameters'])
            elif self.settings['CollectionPreprocessSettings']['SampleType'] == 'Single-Field':
                _l1 += 'Masked with {MaskingMethod!r} and Threshold Parameter {TP}'.format(
                    **md['FieldParameters'])

            self.add(AppLabel, text=_l1, sid=f':{image}', container=f'AppFrame:{image}:Metadata', justify='left',
                     overwrite=True)
        except KeyError as e:
            if self.dv_get('Debugger') is True:
                supports.tprint(f'Parameter load-in for image mask {image} failed with {e!r}.')

        self.add(TextButton, text='View Mask', function='open_image', data=mask_path, overwrite=True,
                 sid=f'LowRes:{image}', container=f'AppFrame:{image}:Metadata')
        hr_path = r'{}\{}\StructureMask.tiff'.format(self.dv_get('OutputFolder'), image)
        self.add(TextButton, text='View Hi-Res Mask', function='open_image', data=hr_path, overwrite=True,
                 sid=f'HiRes:{image}', container=f'AppFrame:{image}:Metadata')

    def load(self):
        if not self.is_empty():
            if len(self.selected_files) != len(self.dv_get('SelectedFiles')):
                self.reload()
        else:
            super().load()


class ProcessingOptions(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.__files = None

    def __traces__(self):
        self.dv_trace('AvailableChannels', 'write', self.__update_channel_selection_menu)
        self.dv_trace('SelectedFiles', 'write', self.update_counting_settings)
        self.dv_trace('InputFolder', 'write', self.trace_reload)

    def trace_reload(self):
        self.after(10, super().reload)  # avoid simultaneous triggering of the processing and preprocessing options

    def __base__(self):
        self.add(AppTitle, text='Processing')

        # global cell count settings
        self.add(AppSubtitle, text='General Processing Settings')

        self.add(AppLabel, text='Edge exclusion distance (µm):', sid='EdgeExclusionDistance', tooltip=True)
        self.add(SettingEntry, sid='EdgeExclusionDistance', vartype=int, default=50, prow=True, column=0, padx=(220, 0))

        self.add(AppLabel, text='Seeding density (c/cm²):', sid='SeedingDensity')
        self.add(SettingEntry, sid='SeedingDensity', vartype=int, prow=True, column=0, padx=(220, 0))

        self.add(AppLabel, text='Cell type:', sid='CellType', tooltip=True)
        _ = ('Fibroblast', 'MC3T3', 'Add ...')
        self.add(SelectionMenu, sid='CellType', options=_, default=0, prow=True, column=0, padx=(75, 0))

        # nuclei processing settings
        _ = self.add(AppCheckbutton, text='Nuclei Processing')
        self.add(AppFrame, sid='NucleiProcessingSettings', padx=(22, 0), pady=(5, 20))
        self.add(AppLabel, text='Nuclei channel:', sid='NucleiChannel', tooltip=True, padx=(0, 10),
                 container='AppFrameNucleiProcessingSettings')
        self.add(SelectionMenu, sid='NucleiChannel', options=self.dv_get('AvailableChannels'), prow=True, column=1,
                 container='AppFrameNucleiProcessingSettings')
        _.trace_add('write', self.__set_up_np_settings)

        # morphology processing settings
        _ = self.add(AppCheckbutton, text='Morphology Processing')
        self.add(AppFrame, sid='MorphologyProcessingSettings', padx=(22, 0), pady=(5, 20))
        self.add(AppLabel, text='Morphology channel:', sid='MorphologyChannel', tooltip=True, padx=(0, 10),
                 container='AppFrameMorphologyProcessingSettings')
        self.add(SelectionMenu, sid='MorphologyChannel', options=self.dv_get('AvailableChannels'), column=1, prow=True,
                 container='AppFrameMorphologyProcessingSettings')

        self.add(AppLabel, text='Threshold intensity:', sid=f'ThresholdIntensity', tooltip=True, padx=(0, 10),
                 container='AppFrameMorphologyProcessingSettings')
        self.add(SettingEntry, sid='ThresholdIntensity', vartype=int, default=0, prow=True, column=1,
                 container='AppFrameMorphologyProcessingSettings')
        _.trace_add('write', self.__set_up_mp_settings)

        # cell counting settings
        self.add(AppSubtitle, text='Cell Counting Settings', tooltip=True)
        self.add(TextButton, sid='ResetCellCountingTable', text='Reset', command=self.restore_default, warning=True,
                 prow=True, padx=(250, 0), pady=(42, 5))
        self.add(AppFrame, sid='CellCountingTable')  # table container
        self.add(TextButton, sid='LoadUnprocessed', text='Select Unprocessed', command=self.select_missing,
                 container='AppFrameCellCountingTable')
        self.add(AppLabel, text='Counting Method', sid='CountingMethod', prow=True, column=1, padx=5, tooltip=True,
                   container='AppFrameCellCountingTable')
        self.add(AppLabel, text='Slice Size', sid='SliceSize', prow=True, column=2, tooltip=True,
                 container='AppFrameCellCountingTable', padx=5)
        self.add(AppLabel, text='Cycles', sid='Cycles', prow=True, column=3, container='AppFrameCellCountingTable',
                 padx=5, tooltip=True)
        self.add(AppLabel, text='Filter', sid='Filter', prow=True, column=4, container='AppFrameCellCountingTable',
                 padx=5, tooltip=True)
        self.add(AppLabel, text='Step', sid='Step', prow=True, column=5, container='AppFrameCellCountingTable',
                 padx=5, tooltip=True)
        self.update_counting_settings()
        self.add(TextButton, sid='LoadPreviousSettings', text='Load Previous Processing', warning=True,
                 command=self.load_previous_processing, container='AppFrameCellCountingTable')

        self.add(AppButton, text='PROCESS', command=self.process)

        self.__update_counting_table_options()

    def __set_up_np_settings(self, *_):
        if self['AppCheckbuttonNucleiProcessing'].get() is True:
            self['AppFrameNucleiProcessingSettings'].grid()
        else:
            self['AppFrameNucleiProcessingSettings'].grid_remove()

    def __set_up_mp_settings(self, *_):
        if self['AppCheckbuttonMorphologyProcessing'].get() is True:
            self['AppFrameMorphologyProcessingSettings'].grid()
        else:
            self['AppFrameMorphologyProcessingSettings'].grid_remove()

    def select_missing(self):
        """Attempt to determine which files are yet to be processed and selected those files for processing."""
        for f in self.dv_get('SelectedFiles'):
            if os.path.isfile(r'{}\{}\data.json'.format(self.dv_get('OutputFolder'), f)):
                self[f'TextCheckbutton:{f}'].set(False)
            elif self.exists(f'TextCheckbutton:{f}'):
                self[f'TextCheckbutton:{f}'].set(True)

    def __update_channel_selection_menu(self, *_):
        self['SelectionMenuNucleiChannel'].update_options(self.dv_get('AvailableChannels'))
        self['SelectionMenuMorphologyChannel'].update_options(self.dv_get('AvailableChannels'))

    def update_counting_settings(self, *_):
        """Internal method that updates setting table."""

        _ = []
        for file, state in self.dv_get('SelectedFiles').items():
            file_dir = r'{}\{}'.format(self.dv_get('OutputFolder'), file)
            if os.path.isdir(file_dir) and os.path.isfile(rf'{file_dir}\maskdata.json'):
                self.show_table_entry(file)  # ensure that all files exist in the widget space
                if state is True:
                    _.append(file)
            else:
                self.hide_table_entry(file)
        self.__files = _

    def add_table_entry(self, file):
        self.add(TextCheckbutton, text=file, sid=f':{file}', fg=supports.__cp__['fg'], font='Arial 12',
                 container='AppFrameCellCountingTable', padx=(0, 15), tag='ImageSelection')

        elem = self.add(SelectionMenu, sid=rf'CountingMethod:{file}', prow=True, column=1, width=14,
                        font='Arial 12', bg=supports.__cp__['dark_bg'], container='AppFrameCellCountingTable',
                        default=0, options=('CCA', 'Black-Out', 'Classic', 'Hough'), padx=5,
                        group='CellCountingTableSettings')
        elem.tether(self[f'TextCheckbutton:{file}'], False, action=elem.tether_action)
        elem.trace_add('write', self.__update_counting_table_options)
        elem.label_bind('<Control-1>', self.__update_all)

        elem = self.add(SettingEntry, sid=rf'SliceSize:{file}', prow=True, column=2, width=8, padx=5,
                        container='AppFrameCellCountingTable', default='', vartype=int,
                        group='CellCountingTableSettings')
        elem.tether(self[f'TextCheckbutton:{file}'], False, action=elem.tether_action)
        elem.bind('<Control-1>', self.__update_all)

        elem = self.add(SettingEntry, sid=rf'Cycles:{file}', prow=True, column=3, width=6, padx=5,
                        container='AppFrameCellCountingTable', default=1, vartype=int, tag='Black-Out',
                        group='CellCountingTableSettings')
        elem.tether(self[f'TextCheckbutton:{file}'], False, action=elem.tether_action)
        elem.bind('<Control-1>', self.__update_all)

        elem = self.add(SettingEntry, sid=rf'Filter:{file}', prow=True, column=4, width=6, padx=5,
                        container='AppFrameCellCountingTable', default=2.1, vartype=float, tag='Black-Out',
                        group='CellCountingTableSettings')
        elem.tether(self[f'TextCheckbutton:{file}'], False, action=elem.tether_action)
        elem.bind('<Control-1>', self.__update_all)

        elem = self.add(SettingEntry, sid=rf'Step:{file}', prow=True, column=5, width=6, padx=5,
                        container='AppFrameCellCountingTable', default=.1, vartype=float, tag='Black-Out',
                        group='CellCountingTableSettings')
        elem.tether(self[f'TextCheckbutton:{file}'], False, action=elem.tether_action)
        elem.bind('<Control-1>', self.__update_all)

    def hide_table_entry(self, file):
        for name in (f'TextCheckbutton:{file}', f'SelectionMenuCountingMethod:{file}',
                     f'SettingEntrySliceSize:{file}', f'SettingEntryCycles:{file}',
                     f'SettingEntryFilter:{file}', f'SettingEntryStep:{file}'):
            if self.exists(name):
                self[name].grid_remove()

    def show_table_entry(self, file):
        for name in (f'TextCheckbutton:{file}', f'SelectionMenuCountingMethod:{file}',
                     f'SettingEntrySliceSize:{file}', f'SettingEntryCycles:{file}',
                     f'SettingEntryFilter:{file}', f'SettingEntryStep:{file}'):
            if self.exists(name):
                self[name].grid()
            else:
                self.add_table_entry(file)

    def load_previous_processing(self):
        """Method that loads the settings used during latest preprocessing."""
        settings = supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), behavior='read')
        files = list(settings['IndividualProcessSettings'].keys())
        _ = []
        for file in self.tags['ImageSelection']:  # set activity status
            name = file.split(':')[-1]
            if name in files:
                self[file].set(True)
                for k, j in zip(('CountingMethod', 'SliceSize', 'Cycles', 'Filter', 'Step'),
                                ('SelectionMenu', 'SettingEntry', 'SettingEntry', 'SettingEntry', 'SettingEntry')):
                    self[f'{j}{k}:{name}'].set(settings['IndividualProcessSettings'][name][k])
            else:
                self[file].set(False)
        self['SettingEntryEdgeExclusionDistance'].set(settings['CollectionProcessSettings']['EdgeProximity'])
        self['SelectionMenuCellType'].set(settings['CollectionProcessSettings']['CellType'])
        self['SelectionMenuNucleiChannel'].set(settings['CollectionProcessSettings']['NucleiChannel'])
        self['SelectionMenuMorphologyChannel'].set(settings['CollectionProcessSettings']['MorphologyChannel'])
        self['SettingEntryThresholdIntensity'].set(settings['CollectionProcessSettings']['ThresholdIntensity'])
        self['SettingEntrySeedingDensity'].set(settings['CollectionProcessSettings']['SeedingDensity'])

    def __update_counting_table_options(self, *_):
        """Internal method that updates cell counting settings depending on selected counting method."""

        non_bo, no_slice = 0, 0
        for file in self.__files:
            if self[rf'SelectionMenuCountingMethod:{file}'].get() != 'Black-Out':
                non_bo += 1
                for option in self.tags['Black-Out']:
                    if option.split(':')[-1] == file:
                        self[option].grid_remove()
                if self[rf'SelectionMenuCountingMethod:{file}'].get() != 'CCA':
                    no_slice += 1
                    self[rf'SettingEntrySliceSize:{file}'].grid_remove()
            else:
                for option in self.tags['Black-Out']:
                    if option.split(':')[-1] == file:
                        self[option].grid()
                self[rf'SettingEntrySliceSize:{file}'].grid()

        for option in ('Cycles', 'Filter', 'Step'):
            if non_bo == len(self.__files):
                self[f'AppLabel{option}'].grid_remove()
            else:
                self[f'AppLabel{option}'].grid()

        if no_slice == len(self.__files):
            self[f'AppLabelSliceSize'].grid_remove()
        else:
            self[f'AppLabelSliceSize'].grid()

    def __update_all(self, e):
        """Internal method that updates all cell counting methods."""
        last_click = self.dv_get('LastClick')
        stype = last_click.split(':')[0]
        value = self[last_click].get()
        for file in self.__files:
            self[rf'{stype}:{file}'].set(value)

    def restore_default(self):
        for option in self.groups['CellCountingTableSettings']:
            self[option].set(self[option].default)

    def process(self):
        # generate settings entry
        _ = {}
        for file in self.__files:
            _[file] = {
                'CountingMethod': self[f'SelectionMenuCountingMethod:{file}'].get(),
                'SliceSize': self[f'SettingEntrySliceSize:{file}'].get(),
                'Cycles': self[f'SettingEntryCycles:{file}'].get(),
                'Filter': self[f'SettingEntryFilter:{file}'].get(),
                'Step': self[f'SettingEntryStep:{file}'].get(),
            }

        update_dict = {
            'CollectionProcessSettings': {
                'EdgeProximity': self['SettingEntryEdgeExclusionDistance'].get(),
                'CellType': self['SelectionMenuCellType'].get(),
                'ThresholdIntensity': self['SettingEntryThresholdIntensity'].get(),
                'SeedingDensity': self['SettingEntrySeedingDensity'].get(),
            },
            'IndividualProcessSettings': _
        }

        # check what types of analysis should be performed
        if self['AppCheckbuttonNucleiProcessing'].get() is False:
            update_dict['CollectionProcessSettings']['NucleiChannel'] = None
        else:
            if self['SelectionMenuNucleiChannel'].get() == 'Select Option':
                messagebox.showerror('Missing Nuclei Channel', 'Set a color channel for nuclei processing.',
                                     parent=self)
            else:
                update_dict['CollectionProcessSettings']['NucleiChannel'] = self['SelectionMenuNucleiChannel'].get()

        if self['AppCheckbuttonMorphologyProcessing'].get() is False:
            update_dict['CollectionProcessSettings']['MorphologyChannel'] = None
        else:
            if self['SelectionMenuMorphologyChannel'].get() == 'Select Option':
                messagebox.showerror('Missing Morphology Channel', 'Set a color channel for morphology processing.',
                                     parent=self)
            else:
                update_dict['CollectionProcessSettings']['MorphologyChannel'] = self['SelectionMenuMorphologyChannel'].get()

        supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), params=update_dict,
                            behavior='update')  # update settings json

        supports.tprint('Started processing.')
        self.process_daemon()

    @supports.thread_daemon
    def process_daemon(self):
        files = [file for file in self.__files if self[f'TextCheckbutton:{file}'].get() is True]  # get active files

        with concurrent.futures.ProcessPoolExecutor(max_workers=supports.get_max_cpu(),
                                                    mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = {executor.submit(base.ProcessingHandler().process, file): file for file in files}
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                try:
                    supports.tprint(f'Processed image {future.result()}.')
                except Exception as e:
                    supports.tprint(f'Failed to process {file} with exit: {e!r}.')
                    if self.dv_get('Debugger') is True:
                        raise e
        supports.tprint('Completed all processing.')


class MultiFieldMaskCreator(FieldMaskCreator):
    """Class that constructs a mask creator for the cellexum application. In essence, this is a sub-application that
    allows for unique mask design, that can be utilized by the application."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Field Settings', pady=(0, 5))
        self.add(TextButton, text='Open Existing Mask', command=self.__open_mask, sid='OpenMask', tooltip=True)

        self.add(AppLabel, text='Field dimensions (w, h):')
        self.add(SettingEntry, sid='FieldWidth', prow=True, padx=(200, 0), vartype=int)
        self.add(AppLabel, text='x', sid='FieldDimensionTimes', prow=True, padx=(260, 0))
        self.add(SettingEntry, sid='FieldHeight', prow=True, padx=(280, 0), vartype=int)

        self.add(AppLabel, text='Field spacing (x, y):')
        self.add(SettingEntry, sid='FieldSpacingX', prow=True, padx=(200, 0), vartype=int)
        self.add(AppLabel, text='x', sid='FieldSpacingTimes', prow=True, padx=(260, 0))
        self.add(SettingEntry, sid='FieldSpacingY', prow=True, padx=(280, 0), vartype=int)

        self.add(AppLabel, text='Spacing deviation (x, y):', sid='SpacingDeviation', tooltip=True)
        self.add(SettingEntry, sid='SpacingDeviationX', prow=True, padx=(200, 0), vartype=int, default=0)
        self.add(AppLabel, text='x', sid='SpacingDeviationTimes', prow=True, padx=(260, 0))
        self.add(SettingEntry, sid='SpacingDeviationY', prow=True, padx=(280, 0), vartype=int, default=0)

        self.add(AppLabel, text='Field units:')
        self.add(SelectionMenu, sid='FieldUnits', prow=True, padx=(200, 0), options=('pix', 'µm'))

        self.add(AppSubtitle, text='Mask Settings')
        self.add(TextButton, text='Import Mask From File', command=self.__import_mask, sid='ImportMask', tooltip=True)
        self.add(AppLabel, text='Columns:', sid='Columns')
        _col = self.add(SettingEntry, sid='Columns', prow=True, padx=(100, 0), width=2, vartype=int)
        self.add(AppLabel, text='Rows:', sid='Rows')
        _row = self.add(SettingEntry, sid='Rows', prow=True, padx=(100, 0), width=2, vartype=int)

        self.add(EntryGrid, sid='Mask', vargrid=(_row.entry, _col.entry), gap=2)
        _col.trace_add('write', self['EntryGridMask'].setup_grid)
        _row.trace_add('write', self['EntryGridMask'].setup_grid)

        self.add(AppLabel, text='Mask name:', sid='MaskName', pady=5)
        self.add(SettingEntry, sid='MaskName', prow=True, padx=(100, 0), width=25, pady=5)
        self.add(AppButton, text='Save Mask', command=self.__save_mask, pady=(5, 0))
        self.add(AppButton, text='Cancel', command=self.cancel_window, pady=(5, 0), padx=(130, 0), prow=True)

    def __fetch_mask(self, **kwargs):
        """Internal method that presets the fetch_mask method."""
        path, file_name = self.fetch_mask('SettingEntryMaskName', **kwargs)
        return path, file_name

    def __import_mask(self):
        """Internal method that imports a mask file into the mask creator."""
        mask_path, _ = self.__fetch_mask(filetypes=[('Text File', '*.txt'), ('Excel File', '*.xlsx'), ('Mask File', '*.mask')])
        _df, _dict = self.import_mask(mask_path)  # import mask from path
        mrows, mcols = _df.shape  # get rows and columns
        self['SettingEntryRows'].set(mrows)
        self['SettingEntryColumns'].set(mcols)
        self['EntryGridMask'].set_grid(_dict)

    def __save_mask(self):
        """Internal method that saves a created mask to be used in the Cellexum application."""
        json_path = rf'{supports.__cwd__}\__misc__\masks.json'
        mask_name = self['SettingEntryMaskName'].get()

        try:  # fetch mask entries for data type
            mask_json = supports.json_dict_push(json_path, behavior='read')[self.sample_type]
        except KeyError:
            mask_json = {}

        if mask_name in mask_json:
            prompt = messagebox.askokcancel('Mask name already exists',
                                            message=f'A mask already exists with name {mask_name!r} for data type '
                                                    f'{self.sample_type!r}. Proceeding will overwrite it. Do you want '
                                                    f'to continue?')
            if prompt is not True:
                return

        mask_df = base.craft_dataframe(self['EntryGridMask'].get_grid())
        mask_path = rf'{supports.__cwd__}\__masks__\{mask_name}.mask'
        mask_df.to_csv(mask_path, sep='\t', index=False, encoding='utf-8', header=False)

        # add mask data to the mask file
        _ = {self.sample_type: {mask_name: {
            'Rows': self['SettingEntryRows'].get(),
            'Columns': self['SettingEntryColumns'].get(),
            'FieldWidth': self['SettingEntryFieldWidth'].get(),
            'FieldHeight': self['SettingEntryFieldWidth'].get(),
            'FieldSpacingX': self['SettingEntryFieldSpacingX'].get(),
            'FieldSpacingY': self['SettingEntryFieldSpacingY'].get(),
            'FieldUnits': self['SelectionMenuFieldUnits'].get(),
            'SpacingDeviationX': self['SettingEntrySpacingDeviationX'].get(),
            'SpacingDeviationY': self['SettingEntrySpacingDeviationY'].get()
        }}}

        supports.json_dict_push(json_path, _, behavior='update')

        # at last update visible mask options
        if mask_name not in mask_json:  # add option to available options if it does not already exist
            self.tie['SelectionMenuImageMask'].add_option(mask_name, order=-1)
        self.tie['SelectionMenuImageMask'].set(mask_name)
        self.cancel_window()

    def __open_mask(self):
        """Internal method that opens an existing mask and imports all its settings."""
        _, mask_name = self.__fetch_mask(filetypes=[('Mask File', '*.mask')], initialdir=rf'{supports.__cwd__}\__masks__')

        mask = base.load_mask_file(mask_name)
        mask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                            behavior='read')[self.sample_type][mask_name]

        # construct mask grid and preset field names in the grid
        mask_dict = base.strip_dataframe(mask)  # strip the input mask
        self['SettingEntryRows'].set(mask_settings['Rows'])
        self['SettingEntryColumns'].set(mask_settings['Columns'])
        self['EntryGridMask'].set_grid(mask_dict)  # set the mask grid to be the opened mask

        # preset all other settings in the mask creator
        self['SettingEntryMaskName'].set(mask_name)
        self['SettingEntryFieldWidth'].set(mask_settings['FieldWidth'])
        self['SettingEntryFieldHeight'].set(mask_settings['FieldHeight'])
        self['SettingEntryFieldSpacingX'].set(mask_settings['FieldSpacingX'])
        self['SettingEntryFieldSpacingY'].set(mask_settings['FieldSpacingY'])
        self['SelectionMenuFieldUnits'].set(mask_settings['FieldUnits'])

    def open_mask_bind(self, e):
        return self.__open_mask()

    def import_mask_bind(self, e):
        return self.__import_mask()

    def save_mask_bind(self, e):
        return self.__save_mask()


class OrientationReferenceCreator(PopupWindow):
    def __init__(self, parent, tie, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.tie = tie
        self.rotation = tk.IntVar(self, 0)
        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Reference Setup', pady=(0, 5))
        self.add(AppLabel, text='Reference File:', sid='SampleName', tooltip=True)
        _ = supports.sort([i.removesuffix('.vsi') for i in os.listdir(self.tie.dv_get('InputFolder')) if
                           i.endswith('.vsi')])
        _ = self.add(SelectionMenu, options=_, sid='SampleName', prow=True, padx=(150,0), width=20)
        _.trace_add('write', self.__load_reference_image_mt)

        self.add(AppLabel, text='Minimum Fields:')
        self.add(SettingEntry, sid='MinFields', prow=True, padx=(150, 0), vartype=int, default=6,
                     width=20)
        self.add(AppLabel, text='Mask Shift:')
        self.add(SelectionMenu, options=('None', 'Auto'), sid='MaskShift', prow=True, padx=(150, 0),
                     default=0, width=20)
        self.add(AppLabel, text='Fidelity:')
        self.add(SettingEntry, sid='Fidelity', prow=True, padx=(150, 0), width=20, default=3, vartype=int,)
        self.add(AppLabel, text='Span:')
        self.add(SettingEntry, sid='Span', prow=True, padx=(150, 0), width=20, default=10, vartype=int,)

        # construct placeholder for preview image
        self.add(AppSubtitle, text='Reference Orientation')
        _ = self.add(AppCheckbutton, text='Enhance preview image', default=True, sid='EnhancePreview')
        _.trace_add('write', self.__load_reference_image_mt)
        _ = self.add(AppCheckbutton, text='High-resolution preview', default=False, sid='HiResPreview')
        _.trace_add('write', self.__load_reference_image_mt)

        self.add(AppFrame, sid='PreviewImageContainer')
        self.add(ZoomImageFrame, sid='PreviewImage', container='AppFramePreviewImageContainer')
        self.add(ZoomImageFrame, sid='PreviewMask', container='AppFramePreviewImageContainer',
                       prow=True, column=1, padx=40)

        self.add(AppFrame, sid='ReferenceOrientationButtons')
        self.add(AppButton, text='+90', sid='Sub90', command=self.__add90, pady=(5, 0),
                     container='AppFrameReferenceOrientationButtons')
        self.add(AppLabel, textvariable=self.rotation, sid='Rotation', prow=True, column=1, padx=40, pady=(5, 0),
                   container='AppFrameReferenceOrientationButtons')
        self.add(AppButton, text='-90', sid='Add90', command=self.__sub90, prow=True, column=2,
                     container='AppFrameReferenceOrientationButtons', pady=(5, 0))

        self.add(AppButton, text='Create Mask', command=self.create_mask, sid='CreateMask', pady=(20, 0))
        self.add(LoadingCircle, size=32, width=6, bg=supports.__cp__['bg'], aai=5, stepsize=.5, sid='CreateMask',
                       prow=True, padx=(150, 0), pady=(20, 0))

        self.rotation.trace_add('write', self.__load_reference_image_mt)

    def save(self):
        """Method that saves the configuration."""

        scale_bar = supports.json_dict_push(r'{}\{}\metadata.json'.format(
            self.tie.dv_get('OutputFolder'), self['SelectionMenuSampleName'].get()),
        behavior='read')['ImageData']['ScaleBarRMS']

        update_dict = {self.tie.parent['SelectionMenuSampleType'].get(): {self.tie.dv_get('ImageMask'): {
            'OrientationReference': {
                'SampleName': self['SelectionMenuSampleName'].get(),
                'MinFields': self['SettingEntryMinFields'].get(),
                'MaskingMethod': 'Calculate',
                'MaskShift': self['SelectionMenuMaskShift'].get(),
                'ScaleBar': scale_bar,
                'Rotate': self.rotation.get(),
                'RotateMethod': 'Manual',
                'Align': None,
                'IterativeFidelity': self['SettingEntryFidelity'].get(),
                'OnsetIntensitySpan': self['SettingEntrySpan'].get()
        }}}}

        supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json', params=update_dict, behavior='update')

    def __add90(self):
        _ = self.rotation.get()
        self.rotation.set(_ + 90)

    def __sub90(self):
        _ = self.rotation.get()
        self.rotation.set(_ - 90)

    def __load_reference_image_mt(self, *_):
        """Internal wrapper for multi-threading."""
        if self['SelectionMenuSampleName'].get() != 'Select Option':
            return threading.Thread(target=self.__load_reference_image, daemon=True).start()

    def __load_reference_image(self, *_):
        # check existence of image layers
        self['LoadingCircleCreateMask'].start()
        file = self['SelectionMenuSampleName'].get(); channel = self.tie.parent['SelectionMenuMaskChannel'].get()
        channel_path = r'{}\{}\{}.tif'.format(self.tie.dv_get('OutputFolder'), file, channel)
        dump_path = r'{}\_misc'.format(self.tie.dv_get('OutputFolder'))

        if not os.path.isdir(dump_path):
            os.makedirs(dump_path)

        if not os.path.isfile(channel_path):
            supports.tprint('Image layers for selected image have not yet been generated. Generating image layers.')
            base.RawImageHandler().handle([file])

        _ = 1.8 if self['AppCheckbuttonEnhancePreview'].get() is True else 1
        if self['AppCheckbuttonHiResPreview'].get() is True:
            self['ZoomImageFramePreviewImage'].scroll_scalar = (1, 1)
            self['ZoomImageFramePreviewImage'].set_image(path=channel_path, brighten=_, rotate=self.rotation.get())
        else:
            self['ZoomImageFramePreviewImage'].scroll_scalar = (.75, .75)
            lr_path = r'{}\{}_preview.tiff'.format(dump_path, file)
            if not os.path.isfile(lr_path):  # check if preview file exists in folder
                img = cv2.imread(channel_path)
                img = base.criterion_resize(img)
                cv2.imwrite(lr_path, img)

            self['ZoomImageFramePreviewImage'].set_image(path=lr_path, brighten=_, rotate=self.rotation.get())
        self.__load_mask_image(brighten=_)
        self['LoadingCircleCreateMask'].stop()

    def __load_mask_image(self, **kwargs):
        """Internal method that loads the mask image."""
        if self['AppCheckbuttonHiResPreview'].get() is False:
            self['ZoomImageFramePreviewMask'].scroll_scalar = (.75, .75)
            load_path = r'{}\_misc\OrientationReference_preview.tiff'.format(self.tie.dv_get('OutputFolder'))
        else:
            self['ZoomImageFramePreviewMask'].scroll_scalar = (1, 1)
            load_path = r'{}\_misc\OrientationReference.tiff'.format(self.tie.dv_get('OutputFolder'))

        if os.path.isfile(load_path):
            self['ZoomImageFramePreviewMask'].set_image(path=load_path, **kwargs)

    @supports.thread_daemon
    def create_mask(self):
        self['LoadingCircleCreateMask'].start()
        self.save()  # save settings to be fetched
        analyser = analysis.ImageAnalysis()
        analyser.construct_orientation_matrix()
        analyser.create_reference_mask_image()

        _ = 1.8 if self['AppCheckbuttonEnhancePreview'].get() is True else 1
        self.__load_mask_image(brighten=_)
        self['LoadingCircleCreateMask'].stop()


class PresentationMaskCreator(MaskCreatorWindow):
    def __init__(self, parent, tie, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.tie = tie
        self.mask_type = None
        self.mask = None
        self.__cf_config = tie['SelectionGridMask'].get_grid_dict(invert=True)
        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='General Options')
        self.add(TextButton, text='Open Existing Presentation Mask', command=self.__open_presentation_mask)
        self.add(AppCheckbutton, text='Enable TeX', sid='UseTex', default=False)
        self.add(AppCheckbutton, text='Convert fields to TeX math', sid='TexMath', default=False)
        _ = self.add(AppCheckbutton, text='Raw string', sid='RawString', default=False)
        self.add(AppCheckbutton, text='Convert to numbers', sid='Numbers', default=False)
        _.tether(self['AppCheckbuttonTexMath'], True, action=_.tether_action)
        _.tether(self['AppCheckbuttonUseTex'], True, action=_.tether_action)

        self.add(AppSubtitle, text='Set Presentation Mask', sid='PresentationMask')
        _cpps = self.tie.sample_settings()['CollectionPreprocessSettings']  # fetch collection preprocess settings
        self.mask_type = _cpps['MaskSelection']
        self.mask = base.load_mask_file(self.mask_type)  # load mask file
        mask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                            behavior='read')[_cpps['SampleType']][_cpps['MaskSelection']]

        self.add(TextButton, text='Import Presentation Mask', command=self.__import_presentation_mask)

        # construct mask grid and preset field names in the grid
        mask_dict = base.strip_dataframe(self.mask)  # strip the input mask

        _ = self.add(EntryGrid, sid='PresentationMask', grid=(mask_settings['Rows'], mask_settings['Columns']),
                       gap=2, fkwargs={'toggle': True})
        _.set_grid(mask_dict)
        _.set_grid_state(self.__cf_config)  # disable control fields by default

        self.add(AppLabel, text='Presentation mask name:', sid='PresentationMaskName', pady=5)
        self.add(SettingEntry, sid='PresentationMaskName', prow=True, padx=(200, 0), width=25, pady=5)
        self.add(AppButton, text='Save Presentation Mask', command=self.__save_presentation_mask, pady=(5, 0))
        self.add(AppButton, text='Cancel', command=self.cancel_window, pady=(5, 0), padx=(230, 0),
                     prow=True)

    def __fetch_mask(self, **kwargs):
        """Internal method that presets the fetch_mask method."""
        path, file_name = self.fetch_mask('SettingEntryPresentationMaskName', **kwargs)
        return path, file_name

    def __open_presentation_mask(self):
        """Internal method that opens a presentation mask."""
        path = rf'{supports.__cwd__}\__masks__\presentation_masks'
        if not os.listdir(path):
            path = rf'{supports.__cwd__}\__masks__'
        mask_path, mask_name = self.__fetch_mask(filetypes=[('Mask File', '*.mask')], initialdir=path)
        mask = base.load_mask_file(mask_path, relative_path=False)
        mask_setup = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                            behavior='read')[self.mask_type][mask_name]

        # construct mask grid and preset field names in the grid
        mask_dict = base.strip_dataframe(mask)  # strip the input mask
        self['EntryGridPresentationMask'].set_grid(mask_dict)  # set mask grid
        self['EntryGridPresentationMask'].set_grid_state(mask_setup['Enabled'])
        for e in ('UseTex', 'TexMath', 'RawString'):
            self[f'AppCheckbutton{e}'].set(mask_setup[e])


    def __save_presentation_mask(self):
        """Internal method that saves the presentation mask."""

        json_path = rf'{supports.__cwd__}\__misc__\presentation_masks.json'
        mask_name = self['SettingEntryPresentationMaskName'].get()
        if mask_name == '':  # catch missing name
            messagebox.showerror('Missing Name', 'Set a name for the presentation mask before proceeding.',
                                 parent=self.parent)
        else:
            # fetch presentation mask entries for mask type
            mask_json = supports.json_dict_push(json_path, behavior='read')

            if self.mask_type in mask_json:
                mask_json = mask_json[self.mask_type]
                if mask_name in mask_json:
                    prompt = messagebox.askokcancel('Mask name already exists',
                                                    message=f'A mask already exists with name {mask_name!r} for mask '
                                                            f'type {self.mask_type!r}. Proceeding will overwrite it. '
                                                            f'Do you want to continue?',
                                                    parent=self.parent)
                    if prompt is not True:
                        return
            mask_state = self['EntryGridPresentationMask'].get_grid_state(str_keys=True)
            update_dict = {self.mask_type: {mask_name: {
                'UseTex': self['AppCheckbuttonUseTex'].get(),
                'TexMath': self['AppCheckbuttonTexMath'].get(),
                'RawString': self['AppCheckbuttonRawString'].get(),
                'Numbers': self['AppCheckbuttonNumbers'].get(),
                'Enabled': mask_state
            }}}
            supports.json_dict_push(json_path, params=update_dict, behavior='update')  # update presentation_masks.json
            mask_df = base.craft_dataframe(self['EntryGridPresentationMask'].get_grid())

            # catch non-existent directory save
            if os.path.isdir(rf'{supports.__cwd__}\__masks__\presentation_masks\{self.tie.process_mask}') is False:
                os.makedirs(rf'{supports.__cwd__}\__masks__\presentation_masks\{self.tie.process_mask}')

            mask_path = rf'{supports.__cwd__}\__masks__\presentation_masks\{self.tie.process_mask}\{mask_name}.mask'
            mask_df.to_csv(mask_path, sep='\t', index=False, encoding='utf-8', header=False)

            # at last update visible mask options
            if mask_name not in mask_json:  # add option to available options if it does not already exist
                self.tie['SelectionMenuPresentationMask'].add_option(mask_name, order=-1)
            self.tie['SelectionMenuPresentationMask'].set(mask_name)  # set added option as selection

            self.cancel_window()  # destroy mask selector

    def __import_presentation_mask(self):
        """Internal method that imports a mask file into the presentation mask creator."""
        mask_path, _ = self.__fetch_mask(filetypes=[('Text File', '*.txt'), ('Excel File', '*.xlsx'),
                                                    ('Mask File', '*.mask')])
        _df, _dict = self.import_mask(mask_path)  # import mask from path
        self['EntryGridPresentationMask'].set_grid(_dict)

    def open_mask_bind(self, e):
        return self.__open_presentation_mask()

    def import_mask_bind(self, e):
        return self.__import_presentation_mask()

    def save_mask_bind(self, e):
        return self.__save_presentation_mask()

    def cancel_window(self):
        """Update cancel window functionality to insert fallback setting."""
        fallback = self.tie['SelectionMenuPresentationMask'].previous
        if fallback == 'Add ...':
            fallback = self.tie['SelectionMenuPresentationMask'].default
        self.tie['SelectionMenuPresentationMask'].selection.set(fallback)  # set fallback option as selection
        super().cancel_window()


class FieldSortingCreator(PopupWindow):
    def __init__(self, parent, tie, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.tie = tie

        # load-in the selected presentation mask
        self.pmask = self.tie['SelectionMenuPresentationMask'].get()
        pmask_data = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json', behavior='read')
        pmask_data = pmask_data[tie.process_mask]

        if self.pmask == 'Native':
            pm_name = tie.process_mask
        else:
            pm_name = rf'presentation_masks\{tie.process_mask}\{self.pmask}'
        self.pmask_grid = base.strip_dataframe(base.load_mask_file(pm_name))
        self.pmask_fields = [base.str_to_tuple(k, int) for k, v in  # fetch active presentation mask field indices
                             pmask_data[self.pmask]['Enabled'].items() if v is True]
        pm_fields = [self.pmask_grid[k] for k in self.pmask_fields]  # fetch only active fields
        self.__fc_handler = base.FieldCodeHandler(pm_fields)  # start field code handler

        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Field Sorting', pady=(0, 5))
        self.add(AppLabel, text='Delimiters:', sid='Delimiters')
        _ = self.add(SettingEntry, sid='Delimiters', prow=True, padx=(90, 0), width=25)
        _.trace_add('write', self.update_split_examples)
        _ = self.add(AppCheckbutton, text='Split with Regular Expression', sid='UseRegex', default=False)
        _.trace_add('write', self.update_split_examples)
        _ = self.add(AppCheckbutton, text='Respect White Space', sid='RecWhiteSpace', default=False)
        _.trace_add('write', self.update_split_examples)
        self.add(AppLabel, text='Multiple Letters Separator:', sid='MultiLetterSeparator')
        _ = self.add(SettingEntry, sid='MultiLetterSeparator', prow=True, padx=(210, 0), width=7)
        _.trace_add('write', self.update_split_examples)

        for i in (1, 2, 3):
            self.add(AppLabel, text=f'Example #{i}', sid=f'SplitExample{i}', padx=(50, 0), fg=supports.__cp__['disabled_fg'])

        self.add(AppLabel, text='Sorting Indices:', sid='SortingIndices')
        _ = self.add(SettingEntry, sid='SortingIndices', prow=True, padx=(120, 0), width=7)
        _.trace_add('write', self.update_sort_examples)
        _ = self.add(AppCheckbutton, text='Reverse Order', sid='RevOrder', default=False)
        _.trace_add('write', self.update_sort_examples)

        for i in (1, 2, 3, 4, 5):
            self.add(AppLabel, text=f'Example #{i}', sid=f'SortExample{i}', padx=(50, 0), fg=supports.__cp__['disabled_fg'])

        self.add(AppLabel, text='Sorting Name:', sid='SortingName')
        self.add(SettingEntry, sid='SortingName', prow=True, padx=(110, 0), width=25)

        self.add(AppButton, text='Save Field Sorting', command=self.__save_field_sorting, pady=(5, 0))
        self.add(AppButton, text='Cancel', command=self.cancel_window, pady=(5, 0), padx=(200, 0),
                     prow=True)

    def update_split_examples(self, *_):
        """Method that updates the example list if there are delimiters."""
        if self['SettingEntryDelimiters'].get() != '':
            # fetch 3 field codes
            pm_fields = self.pmask_fields
            field_codes_indices = (0, len(pm_fields) // 2, -1)

            self.__fc_handler.split_codes(delimiters=self['SettingEntryDelimiters'].get(),
                                   regex=self['AppCheckbuttonUseRegex'].get(),
                                   white_space=self['AppCheckbuttonRecWhiteSpace'].get(),
                                   separator=self['SettingEntryMultiLetterSeparator'].get())
            for i, e in enumerate(field_codes_indices):
                self[f'AppLabelSplitExample{i + 1}']['text'] = '{!r}'.format(self.__fc_handler.sep_codes[e]
                                                                          ).removeprefix('[').removesuffix(']')

    def update_sort_examples(self, *_):
        """Method that updates the example list if there are delimiters."""
        if self['SettingEntrySortingIndices'].get() != '':
            try:
                _idx = tuple(map(int, self['SettingEntrySortingIndices'].get().split(',')))
                self.__fc_handler.sort_codes(indices=_idx,
                                             reverse_order=self['AppCheckbuttonRevOrder'].get())
                for i in range(5):
                    self[f'AppLabelSortExample{i + 1}']['text'] = '{!r}'.format(self.__fc_handler.sor_codes[i]
                                                                              ).removeprefix('[').removesuffix(']')
            except ValueError:
                pass

    def __save_field_sorting(self):
        """Internal method that saves the field sorting to the field sortings json."""

        json_path = rf'{supports.__cwd__}\__misc__\field_sortings.json'
        sorting_name = self['SettingEntrySortingName'].get()
        if sorting_name == '':  # catch missing name
            messagebox.showerror('Missing Name', 'Set a name for the field sorting before proceeding.',
                                 parent=self.parent)
        else:
            # fetch existing field sorting entries
            sorting_json = supports.json_dict_push(json_path, behavior='read')

            if self.pmask in sorting_json:
                sorting_json = sorting_json[self.pmask]
                if sorting_name in sorting_json:
                    prompt = messagebox.askokcancel('Field sorting name already exists',
                                                    message=f'A field sorting already exists with name {sorting_name!r} '
                                                            f'for mask type {self.pmask!r}. Proceeding will '
                                                            f'overwrite it. Do you want to continue?',
                                                    parent=self.parent)
                    if prompt is not True:
                        return

            _idx = tuple(map(int, self['SettingEntrySortingIndices'].get().split(',')))
            update_dict = {self.tie.process_mask: {self.pmask: {sorting_name: {
                'Delimiters': self['SettingEntryDelimiters'].get(),
                'UseRegex': self['AppCheckbuttonUseRegex'].get(),
                'RecWhiteSpace': self['AppCheckbuttonRecWhiteSpace'].get(),
                'MultiLetterSeparator': self['SettingEntryMultiLetterSeparator'].get(),
                'SortingIndices': _idx,
                'RevOrder': self['AppCheckbuttonRevOrder'].get()
            }}}}

            supports.json_dict_push(json_path, params=update_dict, behavior='update')  # update presentation_masks.json

            # at last update visible mask options
            if sorting_name not in sorting_json:  # add option to available options if it does not already exist
                self.tie['SelectionMenuFieldSorting'].add_option(sorting_name, order=-1)
            self.tie['SelectionMenuFieldSorting'].set(sorting_name)  # set added option as selection

            self.cancel_window()  # destroy sorting creator

    def cancel_window(self):
        """Update cancel window functionality to insert fallback setting."""
        # define fallback functionality to prevent the 'Add ...' selection to ever be selected as the mask
        current = self.tie['SelectionMenuFieldSorting'].get(); previous = self.tie['SelectionMenuFieldSorting'].previous
        if current == 'Add ...':
            if previous == 'Add ...':
                self.tie['SelectionMenuFieldSorting'].set(self.tie['SelectionMenuFieldSorting'].default)
            else:
                self.tie['SelectionMenuFieldSorting'].set(previous)
        super().cancel_window()


class GroupNameEntry(AppEntry):
    def __init__(self, parent, frame, *args, **kwargs):
        kwargs['font'] = 'Arial 12'; kwargs['width'] = 30
        super().__init__(parent, *args, **kwargs)

        self.frame = frame
        self.trace_add('write', self.__color_change)
        self.__color_change()

    def __color_change(self, *_):
        if self.get() == '':
            self['bg'] = supports.highlight(supports.__cp__['dark_bg'], -30)


class GroupEditEntry(AppEntry):
    def __init__(self, parent, ties, frame, *args, **kwargs):
        self.ties = ties
        self.frame = frame
        kwargs['font'] = 'Arial 12 bold'; kwargs['width'] = 30
        super().__init__(parent, *args, **kwargs)

        self.trace_add('write', self.__update_ties)
        self.bind('<Control-1>', self.__disable_ties)

    def __update_ties(self, *_):
        self.frame.update_suppress.set(True)  # suppress trace triggering while updating from GroupEditEntry
        for tie in self.ties:
            self.frame[f'GroupNameEntry:{tie}'].set(self.get())
        self.frame.update_suppress.set(False)
        self.frame.update_group_list()

        if len(self.frame.groups.get()) != len(self.frame.tags['EditGroup']):
            self.frame.load_edit_groups_column()

    def __disable_ties(self, e):
        for tie in self.ties:
            self.frame[f'GroupNameEntry:{tie}'].get_tethered('target').set(False)


class MultiGroupCreator(PopupWindow):
    def __init__(self, parent, tie, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.tie = tie
        self.__mg_handler = base.MultiGroupHandler(list(self.tie.groups.get()))
        self.tags['RenameMultiGroups'] = []; self.mg_dg_names = None
        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Multi-Group Setup', pady=(0, 5))
        self.add(AppLabel, text='Grouping type:', sid='GroupingType')
        self.add(SelectionMenu, options=('Index', 'Manual'), sid='GroupingType', padx=(120, 0), prow=True,
                        default=0, width=7)

        # index grouping
        self.add(AppFrame, sid='IndexContainer')
        _ = self.add(AppCheckbutton, text='White space delimiting', sid='WhiteSpaceDelimit', default=False,
                     container='AppFrameIndexContainer')
        self.add(AppLabel, text='Delimiters:', sid='Delimiters', container='AppFrameIndexContainer')
        elem = self.add(SettingEntry, sid='Delimiters', prow=True, padx=(90, 0), width=25,
                        container='AppFrameIndexContainer')
        elem.tether_action['selection'] = ' '
        elem.tether(_, True, elem.tether_action)
        elem.trace_add('write', self.update_groups)

        elem = self.add(AppCheckbutton, text='Split with Regular Expression', sid='UseRegex', default=False,
                        container='AppFrameIndexContainer')
        elem.tether_action['selection'] = False
        elem.tether(_, True, elem.tether_action)
        elem.trace_add('write', self.update_groups)

        elem = self.add(AppCheckbutton, text='Respect White Space', sid='RecWhiteSpace', default=False,
                        container='AppFrameIndexContainer')
        elem.tether_action['selection'] = True
        elem.tether(_, True, elem.tether_action)
        elem.trace_add('write', self.update_groups)

        self.add(AppLabel, text='Multiple Letters Separator:', sid='MultiLetterSeparator',
                 container='AppFrameIndexContainer')
        elem = self.add(SettingEntry, sid='MultiLetterSeparator', prow=True, padx=(200, 0), width=7,
                        container='AppFrameIndexContainer')
        elem.tether_action['selection'] = ''
        elem.tether(_, True, elem.tether_action)
        elem.trace_add('write', self.update_groups)

        self.add(AppLabel, text='Group index:', sid='GroupIndex', container='AppFrameIndexContainer')
        elem = self.add(SettingEntry, sid='GroupIndex', prow=True, padx=(100, 0), width=7, vartype=int,
                 container='AppFrameIndexContainer')
        elem.trace_add('write', self.update_groups)

        # manual grouping
        self.add(AppFrame, sid='ManualContainer')
        self.add(AppLabel, text='Groups:', sid='NumberGroups', container='AppFrameManualContainer')

        self['SelectionMenuGroupingType'].trace_add('write', self.__toggle_grouping_type)
        self.__toggle_grouping_type()

        # edit groups
        self.add(AppSubtitle, text='Edit')
        self.add(AppFrame, sid='EditContainer')
        _ = self.add(AppCheckbutton, text='Rename multi-groups', sid='RenameMGs', default=False,
                 container='AppFrameEditContainer')
        self.add(AppFrame, sid='RenameMGContainer', padx=(22, 0), pady=(5, 20), container='AppFrameEditContainer')
        _.trace_add('write', self.__toggle_rename_table)
        self['AppFrameRenameMGContainer'].grid_remove()

        _ = self.add(AppCheckbutton, text='Bulk edit group names', sid='BulkGroupEdit', default=False,
                 container='AppFrameEditContainer')
        self.add(AppFrame, sid='BGEContainer', padx=(22, 0), pady=(5, 20), container='AppFrameEditContainer')
        self.add(AppLabel, text='Delimiter:', sid='BGEDelimiter', container='AppFrameBGEContainer')
        elem = self.add(SettingEntry, sid='BGEDelimiter', prow=True, padx=(10, 0), column=1, width=1,
                 container='AppFrameBGEContainer')
        elem.trace_add('write', self.__edit_group_names)

        self.add(AppLabel, text='Target:', sid='BGETarget', container='AppFrameBGEContainer')
        elem = self.add(SettingEntry, sid='BGETarget', prow=True, padx=(10, 0), column=1, width=20,
                 container='AppFrameBGEContainer')
        elem.trace_add('write', self.__edit_group_names)

        self.add(AppLabel, text='Replacement:', sid='BGEReplacement', container='AppFrameBGEContainer')
        elem = self.add(SettingEntry, sid='BGEReplacement', prow=True, padx=(10, 0), column=1, width=20,
                 container='AppFrameBGEContainer')
        elem.trace_add('write', self.__edit_group_names)

        _.trace_add('write', self.__toggle_bulk_edit_table)
        self['AppFrameBGEContainer'].grid_remove()

        # group examples
        self.add(AppSubtitle, text='Groups')
        self.add(AppFrame, sid='GroupContainer')

        self.add(AppLabel, text='Multi-Group Name:', sid='MultiGroupName')
        self.add(SettingEntry, sid='MultiGroupName', prow=True, padx=(170, 0), width=25)

        self.add(AppButton, text='Save Multi-Group Setup', command=self.__save_multi_group_setup, pady=(5, 0))
        self.add(AppButton, text='Cancel', command=self.cancel_window, pady=(5, 0), padx=(240, 0), prow=True)

    def __edit_group_names(self, *_):
        if self.__mg_handler.multi_groups is not None:
            delimiter = self['SettingEntryBGEDelimiter'].get()
            target_string = self['SettingEntryBGETarget'].get()
            replacement_string = self['SettingEntryBGEReplacement'].get()
            if delimiter == '':
                targets = [target_string]; replacements = [replacement_string]
            else:
                targets = target_string.split(delimiter)
                if delimiter in replacement_string:
                    replacements = replacement_string.split(delimiter)
                else:
                    replacements = [replacement_string] * len(targets)

            _ = {}
            for mg, g in self.__mg_handler.multi_groups.items():
                _[mg] = {}
                for group, display in g.items():
                    for t, r in zip(targets, replacements):
                        display = display.replace(t, r)
                    _[mg][group] = display
                    self[f'AppLabel{mg}:{group}:IGN']['text'] = display
            self.mg_dg_names = _

    def __toggle_bulk_edit_table(self, *_):
        if self['AppCheckbuttonBulkGroupEdit'].get() is True and self.__mg_handler.multi_groups is not None:
            self['AppFrameBGEContainer'].grid()
        else:
            self['AppFrameBGEContainer'].grid_remove()
            if self.__mg_handler.multi_groups is not None:
                for mg, g in self.__mg_handler.multi_groups.items():
                    for group, display in g.items():
                        try:  # a KeyError can arise if this is triggered simultaneously with a new index definition
                            self[f'AppLabel{mg}:{group}:IGN']['text'] = display
                        except KeyError:
                            pass
            self.mg_dg_names = self.__mg_handler.multi_groups

    def __toggle_rename_table(self, *_):
        if self['AppCheckbuttonRenameMGs'].get() is True and self.__mg_handler.multi_groups is not None:
            self['AppFrameRenameMGContainer'].grid()
            if self.__mg_handler.multi_groups is not None:
                self.container_drop('AppFrameRenameMGContainer'); self.tags['RenameMultiGroups'] = []
                for mg in self.__mg_handler.multi_groups.keys():
                    self.add(AppLabel, text=mg, sid=f'MultiGroup:{mg}', container='AppFrameRenameMGContainer',
                             pady=(2, 5), font='Arial 12 bold')
                    _ = self.add(SettingEntry, text=mg, sid=f'MultiGroup:{mg}', container='AppFrameRenameMGContainer',
                             prow=True, column=1, width=15, tag='RenameMultiGroups', padx=(10, 0))
                    _.trace_add('write', self.__edit_multi_group_name)
        else:
            self['AppFrameRenameMGContainer'].grid_remove()
            for k in self.tags['RenameMultiGroups']:
                mg = k.split(':')[-1]
                self[f'AppLabel{mg}']['text'] = mg

    def __edit_multi_group_name(self, *_):
        for k in self.tags['RenameMultiGroups']:
            mg = k.split(':')[-1]
            if self[k].get() != '':
                self[f'AppLabel{mg}']['text'] = self[k].get()
            else:
                self[f'AppLabel{mg}']['text'] = mg

    def __toggle_grouping_type(self, *_):
        if self['SelectionMenuGroupingType'].get() == 'Index':
            self['AppFrameManualContainer'].grid_remove(); self['AppFrameIndexContainer'].grid()
        elif self['SelectionMenuGroupingType'].get() == 'Manual':
            self['AppFrameIndexContainer'].grid_remove(); self['AppFrameManualContainer'].grid()

    def update_groups(self, *_):
        """Method that updates the group list if there are delimiters."""
        if self['SettingEntryDelimiters'].get() != '' and self['SettingEntryGroupIndex'].get() != '':
            try:
                self.container_drop('AppFrameGroupContainer')
                self.__mg_handler.split_codes(delimiters=self['SettingEntryDelimiters'].get(),
                                       regex=self['AppCheckbuttonUseRegex'].get(),
                                       white_space=self['AppCheckbuttonRecWhiteSpace'].get(),
                                       separator=self['SettingEntryMultiLetterSeparator'].get())
                self.__mg_handler.group_codes(index=self['SettingEntryGroupIndex'].get())

                self.add(AppLabel, text='Multi-Group', container='AppFrameGroupContainer', padx=(0, 20))
                self.add(AppLabel, text='Data Groups', prow=True, column=1, container='AppFrameGroupContainer',
                         padx=(0, 20))
                self.add(AppLabel, text='In-Group Name', prow=True, column=2, container='AppFrameGroupContainer',
                         padx=(0, 20))

                for (mg_k, mg_v), c in zip(self.__mg_handler.multi_groups.items(),
                                           supports.ColorPresets().get(len(self.__mg_handler.multi_groups),
                                                                       self.tie['SelectionMenuColorPreset'].get())):
                    c = supports.rgb_to_hex(c)
                    self.add(AppLabel, text=mg_k, container='AppFrameGroupContainer', padx=(0, 20),
                             fg=c, font='Arial 12 bold')
                    for i, (dg, ign) in enumerate(mg_v.items()):
                        prow = True if i == 0 else False
                        self.add(AppLabel, text=dg, sid=f'{mg_k}:{dg}', column=1, container='AppFrameGroupContainer',
                                 padx=(0, 20), prow=prow, fg=c, font='Arial 12 bold')
                        self.add(AppLabel, text=ign, sid=f'{mg_k}:{dg}:IGN', column=2, container='AppFrameGroupContainer',
                                 padx=(0, 20), prow=True, fg=c, font='Arial 12 bold')
            except TypeError:
                pass
        self.__toggle_rename_table()
        self.__toggle_bulk_edit_table()

    def __save_multi_group_setup(self):
        """Internal method that saves the multi-group setup to the multi-group json."""

        json_path = r'{}\_misc\multi_group.json'.format(self.tie.dv_get('OutputFolder'))
        mg_name = self['SettingEntryMultiGroupName'].get()
        if mg_name == '':  # catch missing name
            messagebox.showerror('Missing Name', 'Set a name for the multi-group setup before proceeding.',
                                 parent=self.parent)
        elif self.is_empty('AppFrameGroupContainer'):
            messagebox.showerror('Missing Setup', 'Set up the multi-group before proceeding.',
                                 parent=self.parent)
        else:
            # fetch existing multi-group entries
            if os.path.exists(json_path):
                mg_json = supports.json_dict_push(json_path, behavior='read')
            else:
                mg_json = {}

            if mg_name in mg_json:
                prompt = messagebox.askokcancel('Multi-group name already exists',
                                                message=f'A multi-group already exists with name {mg_name!r} '
                                                        f'Proceeding will overwrite it. Do you want to continue?',
                                                parent=self.parent)
                if prompt is not True:
                    return

            if self['AppCheckbuttonRenameMGs'].get() is True:
                mgs = {self[e].get():v for v, e in zip(self.mg_dg_names.values(), self.tags['RenameMultiGroups'])}
            else:
                mgs = self.mg_dg_names

            update_dict = {mg_name: {
                'Delimiters': self['SettingEntryDelimiters'].get(),
                'UseRegex': self['AppCheckbuttonUseRegex'].get(),
                'RecWhiteSpace': self['AppCheckbuttonRecWhiteSpace'].get(),
                'MultiLetterSeparator': self['SettingEntryMultiLetterSeparator'].get(),
                'GroupIndex': self['SettingEntryGroupIndex'].get(),
                'RenameMultiGroup': self['AppCheckbuttonRenameMGs'].get(),
                'BulkEditSettings': {
                    'Enabled': self['AppCheckbuttonBulkGroupEdit'].get(),
                    'Delimiter': self['SettingEntryBGEDelimiter'].get(),
                    'Target': self['SettingEntryBGETarget'].get(),
                    'Replacement': self['SettingEntryBGEReplacement'].get(),
                },
                'MultiGroup': mgs
            }}

            base.directory_checker(r'{}\_misc'.format(self.tie.dv_get('OutputFolder')), clean=False)
            supports.json_dict_push(json_path, params=update_dict, behavior='mutate')  # update presentation_masks.json

            # at last update visible mask options
            if mg_name not in mg_json:  # add option to available options if it does not already exist
                self.tie['SelectionMenuMultiGroup'].add_option(mg_name, order=-1)
            self.tie['SelectionMenuMultiGroup'].set(mg_name)  # set added option as selection

            self.cancel_window()  # destroy sorting creator

    def cancel_window(self):
        """Update cancel window functionality to insert fallback setting."""
        # define fallback functionality to prevent the 'Add ...' selection to ever be selected as the mask
        current = self.tie['SelectionMenuMultiGroup'].get(); previous = self.tie['SelectionMenuMultiGroup'].previous
        if current == 'Add ...':
            if previous == 'Add ...':
                self.tie['SelectionMenuMultiGroup'].set(self.tie['SelectionMenuMultiGroup'].default)
            else:
                self.tie['SelectionMenuMultiGroup'].set(previous)
        super().cancel_window()


class DataGroupEditor(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        kwargs['padx'] = 0; kwargs['pady'] = 0
        super().__init__(parent, *args, **kwargs)

        self.groups = JSONVar(self, value={})
        self.update_suppress = tk.BooleanVar(self, False)
        self.group_colors = {}
        self.colors = supports.ColorPresets()
        self.tags['EditGroup'] = []

        self.groups.trace_add('write', self.__groups_trace)

        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Group Data', tooltip=True)
        self.add(AppLabel, text='Color preset:', sid='ColorPreset', tooltip=True)
        _ = self.add(SelectionMenu, options=self.colors.presets, sid='ColorPreset', prow=True, padx=(100, 0),
                 default=8)
        _.trace_add('write', self.trigger_color_update)

        # set up editor table
        _ = [i.removesuffix('.vsi') for i in os.listdir(self.dv_get('InputFolder')) if i.endswith('.vsi')]
        self.files = []
        for file in _:
            if os.path.isfile(r'{}\{}\data.json'.format(self.dv_get('OutputFolder'), file)):
                self.files.append(file)

        self.add(AppFrame, sid='GroupEditorTable')
        self.add(TextButton, text='Reset', sid='Reset', container='AppFrameGroupEditorTable',
                 command=self.__reset_group_editor, padx=(0, 15))
        self.add(AppLabel, text='Group Name', prow=True, column=1,
                 container='AppFrameGroupEditorTable')
        self.add(TextButton, text='Auto-Group', sid='AutoGroup', prow=True, column=1, padx=(150, 0),
                 container='AppFrameGroupEditorTable', command=self.__auto_group, tooltip=True)
        self.add(AppLabel, text='Edit Groups', prow=True, column=2, tooltip=True,
                 container='AppFrameGroupEditorTable')

        for file in supports.sort(self.files):
            elem = self.add(TextCheckbutton, text=file, container='AppFrameGroupEditorTable', padx=(0, 15),
                            font='Arial 12')
            _ = self.add(GroupNameEntry, sid=f':{file}', prow=True, column=1, tag='GroupEntry',
                         container='AppFrameGroupEditorTable', padx=(0, 15), frame=self)
            _.tether_action['selection'] = ''
            _.trace_add('write', self.__GroupEntry_trace); _.tether(elem, False, action=_.tether_action)
        self.load_edit_groups_column()

        self.add(AppLabel, text='Multi-group setup:', sid='MultiGroup', tooltip=True, pady=(5, 5))
        # set multi-group selection menu options
        _mg_path = r'{}\_misc\multi_group.json'.format(self.dv_get('OutputFolder')); _ = ['None']
        if os.path.exists(_mg_path):
            _ += list(supports.json_dict_push(_mg_path, behavior='read'))
        _ += ['Add ...']
        self.add(SelectionMenu, options=_, sid='MultiGroup', prow=True, padx=(150, 0), pady=(5, 5), default=0, width=20,
                 commands={'Add ...': self.__add_multi_group_setup})
        self.__toggle_multi_group()

    def __toggle_multi_group(self, *_):
        if self.parent.sample_type in ('Single-Field', 'Zero-Field'):
            self['AppLabelMultiGroup'].grid(); self['SelectionMenuMultiGroup'].grid()
        else:
            self['AppLabelMultiGroup'].grid_remove(); self['SelectionMenuMultiGroup'].grid_remove()

    def load(self):
        try:  # catch missing output folder
            super().load()
        except KeyError:
            supports.tprint('No input directory defined.')

    def __groups_trace(self, *_):
        if self.parent.sample_type in ('Single-Field', 'Zero-Field'):
            self.parent.update_erg_selection_menu()

    def load_edit_groups_column(self, *_):
        if self.update_suppress.get() is False:
            onset_row = self['AppLabelGroupName'].grid_info()['row'] + 1
            groups = self.groups.get()

            if groups:
                self['AppLabelEditGroups'].grid()
                for group in self.tags['EditGroup']:
                    self.drop(group)  # drop all existing groups
                self.tags['EditGroup'] = []  # reset tags

                for n, (group, ties) in enumerate(groups.items()):  # get new groups
                    _c = self.group_colors[group]
                    c = supports.rgb_to_hex(_c)
                    fg_c = supports.highlight(c, -65) if np.mean(_c) > .5 else supports.highlight(c, 130)
                    _ = self.add(GroupEditEntry, sid=f':{group}', row=onset_row + n, column=2, ties=ties,
                                 container='AppFrameGroupEditorTable', default=group, tag='EditGroup',
                                 frame=self, bg=c, fg=fg_c, pady=(0, 4))
                    _.trace_add('write', self.update_group_list)
            else:
                for group in self.tags['EditGroup']:
                    self.drop(group)  # drop all existing groups
                self['AppLabelEditGroups'].grid_remove()

    def __GroupEntry_trace(self, *_):
        self.update_group_list()
        self.load_edit_groups_column()
        self.update_group_name_colors()

    def update_group_list(self, *_):
        groups = {}  # get unique groups
        for entry in self.tags['GroupEntry']:
            file = entry.split(':')[-1]; group = self[entry].get()
            if group != '':
                if group not in groups:
                    groups[group] = []
                groups[self[entry].get()].extend([file])

        size = len(groups.keys())
        if size > 0:
            colors = self.colors.get(size, self['SelectionMenuColorPreset'].get())
            self.group_colors = dict(zip(groups.keys(), colors))
        self.groups.set(groups)

    def update_group_name_colors(self, *_):
        for f in self.files:
            group = self[f'GroupNameEntry:{f}'].get()
            if group != '':  # catch disabled/ungrouped fields
                _c = self.group_colors[group]
                c = supports.rgb_to_hex(_c)
                fg_c = supports.highlight(c, -65) if np.mean(_c) > .5 else supports.highlight(c, 130)
                self[f'GroupNameEntry:{f}']['bg'] = c; self[f'GroupNameEntry:{f}']['fg'] = fg_c

    def trigger_color_update(self, *_):
        groups = self.groups.get().keys()
        size = len(groups)
        if size > 0:
            colors = self.colors.get(size, self['SelectionMenuColorPreset'].get())
            self.group_colors = dict(zip(groups, colors))

            self.update_group_name_colors()
            for group in self.tags['EditGroup']:
                _c = self.group_colors[self[group].get()]
                c = supports.rgb_to_hex(_c)
                fg_c = supports.highlight(c, -65) if np.mean(_c) > .5 else supports.highlight(c, 130)
                self[group]['bg'] = c; self[group]['fg'] = fg_c

    def __add_multi_group_setup(self):
        level = TopLevelWidget(self); level.title('Multi-Group Creator')
        level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')
        content = MultiGroupCreator(level.main.interior, name='MultiGroupCreator', tie=self)
        content.pack(fill='both', expand=True)

    @supports.thread_daemon
    def __reset_group_editor(self):
        for f in self.files: self[f'GroupNameEntry:{f}'].set('')

    @supports.thread_daemon
    def __auto_group(self):
        for f in self.files:
            if self[f'GroupNameEntry:{f}']['state'] in ('normal', tk.NORMAL):  # prevent overwriting disabled fields
                self[f'GroupNameEntry:{f}'].set(' '.join(f.split('_')[:-1]))


class AnalysisOptions(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.process_mask = None

    def __traces__(self):
        self.dv_trace('InputFolder', 'write', self.clear)

    def __base__(self):
        # load sample set settings
        settings = supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), behavior='read')
        self.sample_type = sample_type = settings['CollectionPreprocessSettings']['SampleType']

        self.add(AppTitle, text='Data Analysis')
        self.add(DataGroupEditor, sid='Section')
        self.add(AppSubtitle, text='Analyze Data')

        # cell count plot options
        elem = self.add(AppCheckbutton, text='Nuclei Analysis', tooltip=True)
        self.add(AppFrame, sid='NucleiAnalysisSettings', padx=(22, 0), pady=(5, 20))

        if sample_type == 'Multi-Field':
            self.process_mask = process_mask = settings['CollectionPreprocessSettings']['MaskSelection']
            mask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                                    behavior='read')[sample_type][process_mask]

            # fetch custom presentation masks if they exist
            self.add(AppLabel, text='Presentation mask:', tooltip=True, sid='PresentationMask', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings')
            pmask_data = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                                 behavior='read')
            _ = list(pmask_data[process_mask]) + ['Add ...'] if process_mask in pmask_data else ['Add ...']  # fetch keys
            if 'Native' not in _:
                _ = ['Native'] + _  # add native option
            self.add(SelectionMenu, sid='PresentationMask', options=_, prow=True, column=1, width=17,
                     container='AppFrameNucleiAnalysisSettings',
                     commands={'Add ...': self.__add_mask, 'Native': self.__add_native_mask})

            self.add(AppLabel, text='Field sorting:', sid='FieldSorting', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings', tooltip=True)
            _ = self.add(SelectionMenu, options=('None', 'Add ...'), sid='FieldSorting', prow=True, column=1, width=17,
                         commands={'Add ...': self.__add_field_sorting}, container='AppFrameNucleiAnalysisSettings')
            _.tether(self['SelectionMenuPresentationMask'], 'Select Option', _.tether_action, mode='==')
            self['SelectionMenuPresentationMask'].trace_add('write', self.__update_sorting_options)

            self.add(AppCheckbutton, text='Mean multiples', sid='MeanIndexMultiples', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings', tooltip=True, default=True)
        elif sample_type in ('Single-Field', 'Zero-Field'):
            self.add(AppLabel, text='External reference group:', sid='ExternalReferenceGroup', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings', tooltip=True)
            _ = self.add(SelectionMenu, options=('None',), sid='ExternalReferenceGroup', prow=True, column=1,
                         default=0, container='AppFrameNucleiAnalysisSettings', width=15)
            self.add(AppCheckbutton, text='Hide external reference', padx=(0, 10), sid='HideExternalReference',
                     container='AppFrameNucleiAnalysisSettings', default=True)
            _.trace_add('write', self.__set_up_her_checkbutton); self.__set_up_her_checkbutton()

            self.add(AppLabel, text='Plot bar width', sid='BarWidth', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings')
            self.add(SettingEntry, sid='BarWidth', prow=True, container='AppFrameNucleiAnalysisSettings', default=0.4,
                     vartype=float, column=1)

            self.add(AppLabel, text='Seeding score criterion:', sid='SeedingScoreCriterion', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings', tooltip=True)
            self.add(SettingEntry, sid='SeedingScoreCriterion', prow=True, container='AppFrameNucleiAnalysisSettings',
                     default=0, vartype=float, column=1)

        self.add(AppLabel, text='Data label size:', sid='LabelSize', padx=(0, 10),
                 container='AppFrameNucleiAnalysisSettings')
        self.add(SettingEntry, sid='LabelSize', prow=True, container='AppFrameNucleiAnalysisSettings', default=7,
                 vartype=float, column=1)

        self.add(AppLabel, text='Data label rotation:', sid='LabelRotation', padx=(0, 10),
                 container='AppFrameNucleiAnalysisSettings')
        self.add(SettingEntry, sid='LabelRotation', prow=True, container='AppFrameNucleiAnalysisSettings', default=70,
                 vartype=float, column=1)

        self.add(AppCheckbutton, text='Morphology evaluation', sid='MorphologyAnalysis', default=False, tooltip=True,
                 container='AppFrameNucleiAnalysisSettings')
        self.add(AppCheckbutton, text='Seeding compensation', sid='SeedingCompensation', default=False, tooltip=True,
                 container='AppFrameNucleiAnalysisSettings')
        self.add(AppCheckbutton, text='Mark seeding count', sid='MarkIdealCellCount', default=False, tooltip=True,
                 container='AppFrameNucleiAnalysisSettings')

        elem.trace_add('write', self.__set_up_ccp_settings)

        # nearest neighbour histogram options
        elem = self.add(AppCheckbutton, text='Nearest Neighbor Evaluation')
        self.add(AppFrame, sid='NearestNeighborEvaluationSettings', padx=(22, 0), pady=(5, 20))

        self.add(AppLabel, text='Maximum nearest neighbour:', sid='MaxNND', padx=(0, 10),
                 container='AppFrameNearestNeighborEvaluationSettings', tooltip=True)
        self.add(SettingEntry, width=7, sid='MaxNND', prow=True, column=1, default='',
                 container='AppFrameNearestNeighborEvaluationSettings', vartype=float)

        self.add(AppLabel, text='Statistical distribution model:', sid='DistributionModel', padx=(0, 10),
                 container='AppFrameNearestNeighborEvaluationSettings')
        _ = self.add(SelectionMenu, options=('Normal', 'Log Normal', 'Skewed Normal'), sid='DistributionModel',
                     prow=True, column=1, default=1, container='AppFrameNearestNeighborEvaluationSettings')

        self.add(AppLabel, text='Maximum largest bin reduction:', sid='ForceFit', padx=(0, 10),
                 container='AppFrameNearestNeighborEvaluationSettings', tooltip=True)
        _ = self.add(SettingEntry, width=7, sid='ForceFit', prow=True, column=1, default=0, vartype=int,
                     container='AppFrameNearestNeighborEvaluationSettings')

        self.add(AppCheckbutton, text='Zero-lock distribution model', sid='ZeroLock', padx=(0, 10), tooltip=True,
                 container='AppFrameNearestNeighborEvaluationSettings', default=False)
        _.trace_add('write', self.__toggle_zero_lock_checkbutton); self.__toggle_zero_lock_checkbutton()

        self.add(AppCheckbutton, text='Apply data groups to graphs', default=True, sid='ApplyDataGroupsToNNE',
                 container='AppFrameNearestNeighborEvaluationSettings')

        elem.trace_add('write', self.__set_up_nne_settings)

        if sample_type == 'Multi-Field':  # control fields setup
            self.add(AppSubtitle, text=f'Control Fields on {process_mask!r}', tooltip=True, sid='ControlFields')
            _ = self.add(SelectionGrid, sid='Mask', grid=(mask_settings['Rows'], mask_settings['Columns']),
                         ars=1, pady=(0, 5), mask=process_mask)
            self.add(TextButton, text='RESET', ars=-1, command=_.reset_grid, sid='ResetControlFields')
            self.add(TextButton, sid='SaveMask', text='Save control fields', command=self.__save_control_fields)
            self.add(AppLabel, text='Selected fields:', tooltip=True, pady=(7, 0), sid='SelectedFields')
            self.add(AppLabel, textvariable=_.display_fields, columnspan=3, prow=True,
                     column=0, padx=(120, 0), pady=(7, 0))

            mask_cfs = supports.json_dict_push(rf'{supports.__cache__}\mask_control_fields.json', behavior='read')
            if process_mask in mask_cfs:
                _.set_grid(mask_cfs[process_mask], True)
            else:
                supports.tprint(f'Mask {process_mask!r} has no defined control fields.')

        self.add(AppSubtitle, text='Miscellaneous Options')

        self.add(AppLabel, text='Graph style:')
        self.add(SelectionMenu, options=('Crisp', 'Crisp (No L-Frame)'), sid='GraphStyle', default=1,
                 prow=True, padx=(120, 0), width=20)

        self.add(AppLabel, text='Figure font:', tooltip=True, sid='FigureFont')
        self.add(SettingEntry, width=20, sid='FigureFont', prow=True, padx=(120, 0), column=0,
                 columnspan=3, default='Arial')
        self.add(AppLabel, text='Figure scalar: ', tooltip=True, sid='FigureScalar')
        self.add(SettingEntry, width=6, sid='FigureScalar', prow=True, padx=(120, 0), column=0,
                 columnspan=3, default=1.2, vartype=float)

        self.add(AppLabel, text='Figure dpi:')
        self.add(SettingEntry, width=6, sid='FigureDpi', prow=True, padx=(120, 0), column=0, default=600, vartype=int)
        self.add(AppCheckbutton, text='Export data to Excel', sid='ExcelExport', tooltip=True, default=False)

        self.add(AppButton, text='Analyze', command=self.__analyze)

        # change defaults before display
        if sample_type in ('Single-Field', 'Zero-Field'):
            self['SettingEntryLabelRotation'].set(0)
            self['SettingEntryLabelSize'].set(10)

        self.check_morphology_processing()  # check whether the morphology processing has been performed on widget load

    def check_morphology_processing(self):
        settings = supports.json_dict_push(r'{}\Settings.json'.format(self.dv_get('OutputFolder')), behavior='read')
        if settings['CollectionProcessSettings']['MorphologyChannel'] is not None:
            self['AppCheckbuttonMorphologyAnalysis'].set(True)  # change morph eval default if processed for it
        else:
            self['AppCheckbuttonMorphologyAnalysis'].grid_remove()

    def update_erg_selection_menu(self, *_):
        groups = list(self['DataGroupEditorSection'].groups.get().keys())
        self['SelectionMenuExternalReferenceGroup'].update_options(['None'] + groups, default='None')

    def __set_up_ccp_settings(self, *_):
        if self['AppCheckbuttonNucleiAnalysis'].get() is True:
            self['AppFrameNucleiAnalysisSettings'].grid()
        else:
            self['AppFrameNucleiAnalysisSettings'].grid_remove()

    def __set_up_nne_settings(self, *_):
        if self['AppCheckbuttonNearestNeighborEvaluation'].get() is True:
            self['AppFrameNearestNeighborEvaluationSettings'].grid()
        else:
            self['AppFrameNearestNeighborEvaluationSettings'].grid_remove()

    def __set_up_her_checkbutton(self, *_):
        if self['SelectionMenuExternalReferenceGroup'].get() != 'None':
            self['AppCheckbuttonHideExternalReference'].grid()
        else:
            self['AppCheckbuttonHideExternalReference'].grid_remove()

    def __update_sorting_options(self, *_):
        """Internal method that updates sorting options based on the selected presentation mask."""
        pmask = self['SelectionMenuPresentationMask'].get()
        field_sortings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\field_sortings.json', behavior='read')

        options = ('None', 'Add ...')
        if self.process_mask in field_sortings:
            if pmask in field_sortings[self.process_mask]:
                options = ['None'] + list(field_sortings[self.process_mask][pmask]) + ['Add ...']

        self['SelectionMenuFieldSorting'].update_options(options, commands={'Add ...': self.__add_field_sorting})

    def __add_native_mask(self):
        """Internal method that adds entries for the native mask if it does not exist."""

        # determine whether a mask should be added or not
        pmask_data = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json', behavior='read')
        add_native_mask = False
        if self.process_mask not in pmask_data:
            add_native_mask = True
        else:
            if 'Native' not in pmask_data[self.process_mask]:
                add_native_mask = True

        if add_native_mask is True:
            _ = self['SelectionGridMask'].get_grid_dict(invert=True, str_keys=True)  # fetch control field setup

            update_dict = {self.process_mask: {'Native': {
                'UseTex': False,
                'TexMath': False,
                'RawString': False,
                'Enabled': _
            }}}

            # push native mask to presentation mask dict
            supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json', behavior='update',
                                    params=update_dict)

    def __add_mask(self):
        level = TopLevelWidget(self); level.title('Presentation Mask Creator')
        level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')
        content = PresentationMaskCreator(level.main.interior, name='PresentationMaskCreator', tie=self)
        content.pack(fill='both', expand=True)

    def __add_field_sorting(self):
        level = TopLevelWidget(self); level.title('Field Sorting Creator')
        level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')
        content = FieldSortingCreator(level.main.interior, name='FieldSortingCreator', tie=self)
        content.pack(fill='both', expand=True)

    def __toggle_zero_lock_checkbutton(self, *_):
        if self['SelectionMenuDistributionModel'].get() == 'Log Normal':
            self['AppCheckbuttonZeroLock'].grid()
        else:
            self['AppCheckbuttonZeroLock'].grid_remove()

    def __save_control_fields(self):
        """Internal method that saves control fields under the selected mask."""
        _existing = supports.json_dict_push(rf'{supports.__cache__}\mask_control_fields.json', behavior='read')
        out_dir = self.dv_get('OutputFolder')
        settings = supports.json_dict_push(rf'{out_dir}\Settings.json', behavior='read')
        _mask = settings['CollectionPreprocessSettings']['MaskSelection']
        if _mask in _existing:
            prompt = messagebox.askokcancel('Warning', f'A control field selection already exists for '
                                                       f'mask {_mask!r}. Proceeding will overwrite the existing '
                                                       f'saved selection. Do you want to continue?')
            if prompt is not True:
                return
        _existing[_mask] = self['SelectionGridMask'].get_grid()  # define new mask
        supports.json_dict_push(rf'{supports.__cache__}\mask_control_fields.json', _existing, behavior='replace')

    def __analyze(self):
        """Internal method that extract the cell counting settings."""

        json_path = r'{}\Settings.json'.format(self.dv_get('OutputFolder'))  # set file path
        settings = {'CollectionAnalysisSettings': {
            'ColorPreset': self['DataGroupEditorSection']['SelectionMenuColorPreset'].get(),
            'DataGroups': list(self['DataGroupEditorSection'].groups.get().keys()),
            'AnalyzeData': {
                'NucleiAnalysis': self['AppCheckbuttonNucleiAnalysis'].get(),
                'NearestNeighbourHistogram': self['AppCheckbuttonNearestNeighborEvaluation'].get(),
            },
            'ExcelExport': self['AppCheckbuttonExcelExport'].get(),
            'FigureFont': self['SettingEntryFigureFont'].get(),
            'LabelSize': self['SettingEntryLabelSize'].get(),
            'LabelRotation': self['SettingEntryLabelRotation'].get(),
            'GraphStyle': self['SelectionMenuGraphStyle'].get(),
            'FigureScalar': self['SettingEntryFigureScalar'].get(),
            'MaxNND': self['SettingEntryMaxNND'].get(),
            'ZeroLock': self['AppCheckbuttonZeroLock'].get(),
            'DistributionModel': self['SelectionMenuDistributionModel'].get(),
            'FigureDpi': self['SettingEntryFigureDpi'].get(),
            'ApplyDataGroupsToNNE': self['AppCheckbuttonApplyDataGroupsToNNE'].get(),
            'ForceFit': self['SettingEntryForceFit'].get(),
            'MorphologyAnalysis': self['AppCheckbuttonMorphologyAnalysis'].get(),
            'SeedingCompensation': self['AppCheckbuttonSeedingCompensation'].get(),
            'MarkIdealCellCount': self['AppCheckbuttonMarkIdealCellCount'].get(),
        }, 'IndividualAnalysisSettings': {}}

        # add sample-type specific settings
        if self.sample_type == 'Multi-Field':
            if self['SelectionMenuFieldSorting'].get() == 'Select Option':
                messagebox.showerror('Missing Field Sorting', "Please select a Field Sorting from the "
                                                              "drop-down menu to continue.", parent=self.parent)
                return

            _ = {
                'ControlFields': self['SelectionGridMask'].get(),
                'FieldSorting': self['SelectionMenuFieldSorting'].get(),
                'PresentationMask': self['SelectionMenuPresentationMask'].get(),
                'MeanIndexMultiples': self['AppCheckbuttonMeanIndexMultiples'].get(),
            }
        elif self.sample_type in ('Single-Field', 'Zero-Field'):
            _ = {
                'ExternalReferenceGroup': self['SelectionMenuExternalReferenceGroup'].get(),
                'HideExternalReference': self['AppCheckbuttonHideExternalReference'].get(),
                'BarWidth': self['SettingEntryBarWidth'].get(),
                'MultiGroup': self['DataGroupEditorSection']['SelectionMenuMultiGroup'].get(),
                'SeedingScoreCriterion': self['SettingEntrySeedingScoreCriterion'].get()
            }
        settings['CollectionAnalysisSettings']['SampleTypeSettings'] = _

        for f in self['DataGroupEditorSection'].files:
            settings['IndividualAnalysisSettings'][f] = {
                'DataGroup': self['DataGroupEditorSection'][f'GroupNameEntry:{f}'].get(),
                'State': self['DataGroupEditorSection'][f'TextCheckbutton{f}'].get()
            }

        supports.json_dict_push(json_path, settings, behavior='mutate')

        multiprocessing.Process(target=self._analyze).start()

    @staticmethod
    def _analyze():
        return base.AnalysisHandler().analyze()


class ResultOverview(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # setup internal image entry
        self.__images = {}

    def __base__(self):
        self.load_settings()  # load settings before setting up the widget
        self.add(AppTitle, text='Results')
        self.add(TextButton, text='Open Folder', command=self.open_figure_folder)

    def open_figure_folder(self):
        os.startfile(r'{}\_figures'.format(self.dv_get('OutputFolder')))


class ApplicationSettings(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

    def __base__(self):

        self.defaults = defaults = supports.json_dict_push(rf'{supports.__cache__}\application.json',
                                               behavior='read')['ApplicationSettings']

        self.add(AppTitle, text='Application Settings')
        self.add(TextButton, text='Restore Defaults', command=self.restore_defaults, warning=True)
        self.add(AppFrame, sid='Column0')
        self.add(AppFrame, sid='Column1', prow=True, column=1)
        self.add(AppFrame, sid='Column2', prow=True, column=2, padx=(50, 0))

        # column 0 setup
        self.add(AppSubtitle, text='Performance', container='AppFrameColumn0')
        self.add(AppLabel, text='Maximum CPU usage:', sid='MaxRelativeCPU', container='AppFrameColumn0', tooltip=True)
        self.add(SettingEntry, sid='MaxRelativeCPU', prow=True, padx=(180, 0), vartype=float,
                 default=defaults['MaxRelativeCPU'], container='AppFrameColumn0')

        self.add(AppLabel, text='Maximum CPU count:', sid='MaxAbsoluteCPU', container='AppFrameColumn0', tooltip=True)
        self.add(SettingEntry, sid='MaxAbsoluteCPU', prow=True, padx=(180, 0), vartype=int,
                 default=defaults['MaxAbsoluteCPU'], container='AppFrameColumn0')

        self.add(AppLabel, text='Audit images resolution:', sid='AuditImageResolution', container='AppFrameColumn0',
                 tooltip=True)
        self.add(SettingEntry, sid='AuditImageResolution', prow=True, padx=(180, 0), vartype=int,
                 default=defaults['AuditImageResolution'], container='AppFrameColumn0')

        self.add(AppLabel, text='File poll frequency (ms):', sid='FilePollingFrequency', container='AppFrameColumn0',
                 tooltip=True)
        self.add(SettingEntry, sid=f'FilePollingFrequency', prow=True, padx=(180, 0), vartype=int,
                 container='AppFrameColumn0', default=defaults['FilePollingFrequency'])

        self.add(AppLabel, text='File poll time-out (s):', sid='FilePollingTimeOut', container='AppFrameColumn0',
                 tooltip=True)
        self.add(SettingEntry, sid=f'FilePollingTimeOut', prow=True, padx=(180, 0), vartype=int,
                 container='AppFrameColumn0', default=defaults['FilePollingTimeOut'])

        self.add(AppSubtitle, text='Interface', container='AppFrameColumn0')
        self.add(AppLabel, text='Tooltip timer (ms):', sid='TooltipTimer', container='AppFrameColumn0', tooltip=True)
        self.add(SettingEntry, sid='TooltipTimer', prow=True, padx=(140, 0), vartype=int,
                 default=defaults['TooltipTimer'], container='AppFrameColumn0')

        # column 1 setup
        self.add(AppSubtitle, text='Mask Deletion', container='AppFrameColumn1', tooltip=True)
        self.add(AppHeading, text='Field Mask', container='AppFrameColumn1')
        self.add(AppLabel, text='Sample type:', sid='SampleTypeFieldMask', container='AppFrameColumn1')
        elem = self.add(SelectionMenu, sid='SampleTypeFieldMask', prow=True, container='AppFrameColumn1', padx=(110, 0),
                        options=('Multi-Field', 'Single-Field'), width=20)
        self.add(AppLabel, text='Field mask:', sid='FieldMask', container='AppFrameColumn1')
        _ = self.add(SelectionMenu, sid='FieldMask', prow=True, container='AppFrameColumn1', padx=(110, 0), options=(),
                     width=20)
        _.tether(elem, 'Select Option', _.tether_action)
        elem.trace_add('write', self._load_field_masks)
        self.add(TextButton, text='Delete Field Mask', command=self._delete_field_mask, container='AppFrameColumn1',
                 warning=True)

        self.add(AppHeading, text='Presentation Mask', container='AppFrameColumn1')
        self.add(AppLabel, text='Presentation mask:', sid='PresentationMask', container='AppFrameColumn1')
        elem = self.add(SelectionMenu, sid='PresentationMask', prow=True, container='AppFrameColumn1', padx=(150, 0),
                        options=(), width=20)
        elem.tether(_, 'Select Option', elem.tether_action)
        _.trace_add('write', self._load_presentation_masks)
        self.add(TextButton, text='Delete Presentation Mask', command=self._delete_presentation_mask,
                 container='AppFrameColumn1', warning=True)

        self.add(AppHeading, text='Field Sorting', container='AppFrameColumn1')
        self.add(AppLabel, text='Field sorting:', sid='FieldSorting', container='AppFrameColumn1')
        _ = self.add(SelectionMenu, sid='FieldSorting', prow=True, container='AppFrameColumn1', padx=(110, 0),
                        options=(), width=20)
        _.tether(elem, 'Select Option', _.tether_action)
        elem.trace_add('write', self._load_field_sortings)
        self.add(TextButton, text='Delete Field Sorting', command=self._delete_field_sorting,
                 container='AppFrameColumn1', warning=True)

        # column 2 setup
        self.add(AppSubtitle, text='Debugging', container='AppFrameColumn2')
        self.add(TextButton, text='Reset Cache', sid='ResetCache', container='AppFrameColumn2',
                 command=self.reset_application_cache, warning=True)
        self.add(AppCheckbutton, text='Toggle debugger', sid='Debugger', container='AppFrameColumn2',
                 selection=self.dv('Debugger'), tooltip=True)

        self.add(AppButton, text='SAVE', command=self.save_settings)
        self.add(TextButton, text='Restart Application', command=self.restart_application, sid='RestartApplication')
        _ = self.add(AppLabel, text='Application restart required'.upper(), sid='RestartApp', font='Arial 6 bold')
        _.grid_remove()

    def restart_application(self):
        self.winfo_toplevel().destroy()
        root = CellexumApplication()
        root.mainloop()

    def reset_application_cache(self):
        for file in ('application', 'directory_memory', 'mask_control_fields', 'saves', 'settings'):
            try:
                os.remove(rf'{supports.__cache__}\{file}.json')
            except FileNotFoundError:
                pass
        supports.tprint(f'Cache was cleared successfully. Application is restarting.')
        self.after(1500, self.restart_application)

    def save_settings(self):
        _ = {'ApplicationSettings': {
            'MaxRelativeCPU': self['SettingEntryMaxRelativeCPU'].get(),
            'MaxAbsoluteCPU': self['SettingEntryMaxAbsoluteCPU'].get(),
            'AuditImageResolution': self['SettingEntryAuditImageResolution'].get(),
            'TooltipTimer': self['SettingEntryTooltipTimer'].get(),
            'Debugger': self['AppCheckbuttonDebugger'].get(),
            'FilePollingFrequency': self['SettingEntryFilePollingFrequency'].get(),
            'FilePollingTimeOut': self['SettingEntryFilePollingTimeOut'].get(),
        }}

        supports.json_dict_push(rf'{supports.__cache__}\application.json', params=_, behavior='update')

        if (self['SettingEntryTooltipTimer'].get() != self.defaults['TooltipTimer'] or
            self['SettingEntryFilePollingFrequency'].get() != self.defaults['FilePollingFrequency'] or
            self['SettingEntryFilePollingTimeOut'].get() != self.defaults['FilePollingTimeOut']):
            self._show_restart_text('required')
        if self['AppCheckbuttonDebugger'].get() != self.defaults['Debugger']:
            self._show_restart_text('optional')

    def _show_restart_text(self, t='required'):
        """Internal method that sets the state of the 'restart application' text."""
        if t == 'required':
            self['AppLabelRestartApp']['text'] = 'Application restart required'.upper()
        elif t == 'optional':
            self['AppLabelRestartApp']['text'] = 'Application restart recommended'.upper()
        self['AppLabelRestartApp'].grid()

    def restore_defaults(self):
        defaults = supports.json_dict_push(rf'{supports.__cwd__}\defaults.json',
                                           behavior='read')['ApplicationSettings']
        for k, v in defaults.items():
            self[f'SettingEntry{k}'].set(v)

        _ = {'ApplicationSettings': defaults}
        supports.json_dict_push(rf'{supports.__cache__}\application.json', params=_, behavior='update')

    def _load_field_masks(self, *_):
        if self['SelectionMenuSampleTypeFieldMask'].get() != 'Select Option':
            try:
                field_mask_json = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json', behavior='read')
                try:
                    masks = field_mask_json[self['SelectionMenuSampleTypeFieldMask'].get()]
                    self['SelectionMenuFieldMask'].update_options(list(masks))
                except KeyError:
                    supports.tprint(r'There exists no associated masks for sample type {!r}.'.format(
                        self['SelectionMenuSampleTypeFieldMask'].get()))
            except IOError:
                supports.tprint(r'There exists no field masks yet.')

    def _load_presentation_masks(self, *_):
        if self['SelectionMenuFieldMask'].get() != 'Select Option':
            try:
                presentation_mask_json = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                                                 behavior='read')
                try:
                    presentation_masks = presentation_mask_json[self['SelectionMenuFieldMask'].get()]
                    self['SelectionMenuPresentationMask'].update_options(list(presentation_masks))
                except KeyError:
                    supports.tprint(r'There exists no associated presentation masks for field mask {!r}.'.format(
                        self['SelectionMenuFieldMask'].get()))
                    self['SelectionMenuPresentationMask'].update_options(())
            except IOError:
                supports.tprint(r'There exists no presentation masks yet.')

    def _load_field_sortings(self, *_):
        if self['SelectionMenuPresentationMask'].get() != 'Select Option':
            try:
                field_sortings_json = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\field_sortings.json',
                                                              behavior='read')
                try:
                    field_mask_sortings = field_sortings_json[self['SelectionMenuFieldMask'].get()]
                    try:
                        presentation_mask_sortings = field_mask_sortings[self['SelectionMenuPresentationMask'].get()]
                        self['SelectionMenuFieldSorting'].update_options(list(presentation_mask_sortings))
                    except KeyError:
                        supports.tprint(r'There exists no associated field sortings for presentation mask {!r}.'.format(
                            self['SelectionMenuPresentationMask'].get()))
                except KeyError:
                    supports.tprint(r'There exists no associated field sorting for field mask {!r}.'.format(
                        self['SelectionMenuFieldMask'].get()))
            except IOError:
                supports.tprint(r'There exists no field sortings yet.')

    def _delete_field_mask(self):
        try:
            # first remove field mask entry
            field_mask_json = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json', behavior='read')
            field_mask_json[self['SelectionMenuSampleTypeFieldMask'].get()].pop(self['SelectionMenuFieldMask'].get())
            supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json', params=field_mask_json, behavior='replace')

            # second remove associated presentation masks
            try:
                presentation_mask_json = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                                                 behavior='read')
                presentation_mask_json.pop(self['SelectionMenuFieldMask'].get())
                supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                        params=presentation_mask_json, behavior='replace')

                shutil.rmtree(r'{}\__masks__\presentation_masks\{}'.format(supports.__cwd__,
                                                                           self['SelectionMenuFieldMask'].get()))

            except KeyError:
                supports.tprint(r'Field mask {!r} has no associated presentation masks.'.format(
                    self['SelectionMenuFieldMask'].get()))

            # third remove associated field sorting options
            try:
                field_sorting_json = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\field_sortings.json',
                                                                 behavior='read')
                field_sorting_json.pop(self['SelectionMenuFieldMask'].get())
                supports.json_dict_push(rf'{supports.__cwd__}\__misc__\field_sortings.json',
                                        params=field_sorting_json, behavior='replace')
            except KeyError:
                supports.tprint(r'Field mask {!r} has no associated field sortings.'.format(
                    self['SelectionMenuFieldMask'].get()))

            # fourth remove associated control fields
            try:
                control_fields_json = supports.json_dict_push(
                    rf'{supports.__cache__}\mask_control_fields.json', behavior='read')
                control_fields_json.pop(self['SelectionMenuFieldMask'].get())
                supports.json_dict_push(rf'{supports.__cache__}\mask_control_fields.json',
                                        params=control_fields_json, behavior='replace')
            except KeyError:
                supports.tprint(r'Field mask {!r} has no associated control fields.'.format(
                    self['SelectionMenuFieldMask'].get()))

            # last remove the mask entry itself
            os.remove(r'{}\__masks__\{}.mask'.format(supports.__cwd__, self['SelectionMenuFieldMask'].get()))
            self._load_field_masks()
            self['AppLabelRestartApp'].grid()

        except KeyError:
            supports.tprint(r'Field mask {!r} for sample type {!r} does not exist'.format(
                self['SelectionMenuFieldMask'].get(), self['SelectionMenuSampleTypeFieldMask'].get()))

    def _delete_presentation_mask(self):
        try:
            presentation_mask_json = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                                             behavior='read')
            presentation_mask_json[self['SelectionMenuFieldMask'].get()].pop(self['SelectionMenuPresentationMask'].get())
            supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                    params=presentation_mask_json, behavior='replace')

            os.remove(r'{}\__masks__\presentation_masks\{}\{}.mask'.format(
                supports.__cwd__, self['SelectionMenuFieldMask'].get(), self['SelectionMenuPresentationMask'].get()))
            self._load_presentation_masks()
            self['AppLabelRestartApp'].grid()

        except KeyError:
            supports.tprint(r'Field mask {!r} for sample type {!r} does not exist'.format(
                self['SelectionMenuFieldMask'].get(), self['SelectionMenuSampleTypeFieldMask'].get()))

    def _delete_field_sorting(self):
        try:
            field_sorting_json = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\field_sortings.json',
                                                         behavior='read')
            field_sorting_json[self['SelectionMenuFieldMask'].get()][self['SelectionMenuPresentationMask'].get()].pop(
                self['SelectionMenuFieldSorting'].get())
            supports.json_dict_push(rf'{supports.__cwd__}\__misc__\field_sortings.json', params=field_sorting_json,
                                    behavior='replace')
            self._load_field_sortings()
            self['AppLabelRestartApp'].grid()
        except KeyError:
            supports.tprint(r'Field sorting {!r} for presentation mask {!r} does not exist'.format(
                self['SelectionMenuFieldSorting'].get(), self['SelectionMenuPresentationMask'].get()))


class CellexumApplication(tk.Tk, TopLevelProperties):
    """Wrapper for the Cellexum application."""

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        TopLevelProperties.__init__(self, dep_var={  # construct global variable dict
            'LastClick': tk.StringVar(self, ''),
            'AvailableChannels': JSONVar(self, value=[]),
            'LatestPreprocessedFile': tk.StringVar(self, ''),
            'ActiveFrame': tk.StringVar(self, ''),
            'Debugger': tk.BooleanVar(self, False)
        })

        apset = supports.json_dict_push(rf'{supports.__cache__}\application.json', behavior='read')
        if 'ApplicationSettings' in apset:
            if 'Debugger' in apset['ApplicationSettings']:
                self.dependent_variables['Debugger'].set(apset['ApplicationSettings']['Debugger'])

        self.dependent_variables['TooltipTimer'] = tk.IntVar(self, self.defaults['TooltipTimer'])

        self.iconbitmap(default=rf'{supports.__gpx__}\icon.ico')

        self.current_frame = ''
        self.dependent_variables['ActiveFrame'].trace_add('write', self.__on_content_change)

        # clear cache upon starting up program  (IDEALLY CLEAN CACHE ON EXIT)
        supports.json_dict_push(rf'{supports.__cache__}\settings.json', behavior='clear')

        self.title('Cellexum')
        self.resizable(True, True)

        app_height = int(self.winfo_screenheight() * .75)
        app_width = 1150 if self.winfo_screenwidth() * .75 <= 1150 else int(self.winfo_screenwidth() * .75)
        self.geometry(f'{app_width}x{app_height}')
        self.configure(background=supports.__cp__['bg'])

        # menu frame
        menu_button_frame = tk.Frame(self, bg=supports.__cp__['dark_bg'], padx=10, pady=20)
        menu_button_frame.pack(side='left', fill='both', expand=False)

        img = cv2.imread(rf'{supports.__gpx__}\application-logo.png', cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        self._logo = ImageTk.PhotoImage(Image.fromarray(img))
        logo = AppLabel(menu_button_frame, image=self._logo, bg=supports.__cp__['dark_bg'])
        logo.pack(side='top', fill='both', expand=False, anchor='center', padx=30, pady=20)

        # interaction frames
        self.content_frame = ContentFrame(self, background=supports.__cp__['bg'])  # construct content frame
        self.content_frame.pack(side='left', fill='both', expand=True)  # place content frame

        self.content_container = {}  # container for application windows and associated buttons
        for frame in (FileSelection, PreprocessingOptions, MaskGallery, ProcessingOptions, AnalysisOptions,
                      ResultOverview, ApplicationSettings):
            _ = frame(self.content_frame.interior)
            self.content_container[_.tkID] = _

        self.menu_buttons = {}
        for button, (fk, fv) in zip(('File Selection', 'Preprocessing', 'Mask Gallery', 'Processing', 'Analysis',
                                     'Results'), self.content_container.items()):
            _ = MenuButton(menu_button_frame, frame=fv, text=button, size='large')
            _.pack(side='top', pady=5)
            self.menu_buttons[fk] = _

        _ = ImageMenuButton(menu_button_frame, image='settings_icon', frame=self.content_container['ApplicationSettings'])
        _.pack(side='bottom', pady=5, anchor='w', padx=(18, 0))

        self.bind('<Configure>', self._update_content_frame)
        self.dependent_variables['ActiveFrame'].set('FileSelection')  # set current frame to the start frame

    def _update_content_frame(self, e):
        self.content_frame.canvas.config(height=self.winfo_height())
        self.content_frame.refresh_content_frame()

    def __on_content_change(self, *_):
        new_frame = self.dependent_variables['ActiveFrame'].get()
        if self.current_frame != new_frame:  # update frame if it is not the same frame that has been clicked
            supports.tprint('Selected Frame: {}'.format(new_frame))

            for v in self.content_container.values():
                v.grid_remove()  # forget loaded content

            _ = self.content_container[new_frame]  # grab new content
            _.load()  # place new content
            _.grid()  # place WindowFrame

            # pre-configure interior width before updating scrollbar
            self.content_frame.interior.configure(width=_.winfo_width(), height=_.winfo_height())
            self.content_frame.canvas.xview_moveto(0)
            self.content_frame.canvas.yview_moveto(0)

            self.current_frame = new_frame  # update current frame


if __name__ == '__main__':
    cellexum = CellexumApplication()
    cellexum.mainloop()
