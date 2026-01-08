"""GUI for cellexum"""
import javabridge
import bioformats
from bioformats import logback
import numpy as np
from skeletons import *
import analysis
import cv2
import supports
import concurrent.futures
import shutil
from tkinter import font as tkfont


graphics_folder = rf'C:\Users\nicho\OneDrive - Aarhus universitet\6SEM\Bachelor\NTSA_PROCESSING\graphics\vector'

active_frames = []  # empty list to store active frames for toggling

class FileSelection(WindowFrame):
    """Frame which allows user to select files for processing."""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self._repeats = 0
        self._directories = {}
        self._repeat_names = {}
        self._selected_repeat = 0
        self.common_name = None
        self.uncommons = None

        # define dependent variables and load widget
        self.dv_define('InputFolder', tk.StringVar(self, ''), group='Directories', branch='Repeat:0')
        self.dv_define('OutputFolder', tk.StringVar(self, ''), group='Directories', branch='Repeat:0')
        self.dv_define('DefaultOutputFolder', tk.StringVar(self, ''), group='Directories', branch='Repeat:0')
        self.dv_define('CommonOutputFolder', tk.StringVar(self, ''), group='Directories')
        self.dv_define('DefaultCommonOutputFolder', tk.StringVar(self, ''), group='Directories')
        self.dv_define('RepeatData', DictVar(self))
        self.load()

    def __base__(self):
        self.add(AppTitle, text='Image File Selection')
        self.add(AppLabel, text='Enter image directory:')
        _ = self.add(DirectoryEntry, sid='InputFolder')
        _.trace_add('write', self.__load_directory)

        self.add(TextButton, text='Add Repeat', command=self.__add_repeat, hide=True)
        self.add(AppFrame, sid='RepeatContainer', hide=True)

        self.add(TextButton, text='Select All', command=self.__select_all, hide=True)
        self.add(TextButton, text='Deselect All', command=self.__deselect_all, padx=(130, 0), prow=True, hide=True)

        self.add(AppFrame, sid='FileSelectionContainer', hide=True)  # construct container for files in directory

        active_frames.append(self)

    def reload(self):
        """Change the reload functionality to simply reset the input folder."""
        self['DirectoryEntryInputFolder'].set('')

    def __select_all(self):
        for elem in self.containers[f'AppFrameRepeatSelection:{self._selected_repeat}']:
            self[elem].set(True)

    def __deselect_all(self):
        for elem in self.containers[f'AppFrameRepeatSelection:{self._selected_repeat}']:
            self[elem].set(False)

    def _update_repeat_tab(self, which):
        """Internal method that updates the data files selection according to the selected repeat."""
        for i in range(self._repeats):
            if i != which:
                try:  # catch deleted repeats
                    self[f'AppFrameRepeatSelection:{i}'].grid_remove()
                    self[f'TextButtonRepeatSelect:{i}']['font'] = 'Arial 8'
                except KeyError:
                    pass
            else:
                self[f'AppFrameRepeatSelection:{which}'].grid()
                self[f'TextButtonRepeatSelect:{which}']['font'] = 'Arial 8 bold'
        self._selected_repeat = which

    def define_common_name(self):
        splitters = {}; repeats = self._repeat_names.values()

        for repeat in repeats:
            for elem in repeat:
                if not elem.isalpha():
                    if elem in splitters:
                        splitters[elem] += 1
                    else:
                        splitters[elem] = 1

        _max = (None, 0)
        for k, v in splitters.items():
            if v > _max[1]:
                _max = (k, v)

        split_repeats = [i.split(_max[0]) for i in repeats]
        sequence = {}
        for c in split_repeats[0]:
            for sr in split_repeats[1:]:
                if c in sr:
                    sequence[c] = c

        join_seq = _max[0] if _max[0] is not None else ''

        uncommons = {}
        for rep_id, rep in zip(self._repeat_names.keys(), split_repeats):
            _ = []
            for elem in rep:
                if elem not in sequence:
                    _.append(elem)
            uncommons[f'Repeat:{rep_id}'] = {'Uncommon': join_seq.join(_)}

        # update global variables and cache settings
        _ = supports.dict_update(self.dv_get('RepeatData'), uncommons)
        self.dv_set('RepeatData', _)
        _ = {'DirectorySettings': _}
        supports.json_dict_push(rf'{supports.__cache__}\settings.json', params=_, behavior='update')

        if sequence:
            common_name = join_seq.join(sequence.keys()) + ' (collection)'
        else:  # catch cases where there is no match between
            common_name = join_seq.join(split_repeats[0]) + ' (collection)'

        return common_name

    def __drop_repeat(self, which):
        """Internal method that drops a repeat from the menu."""
        self.drop(f'TextButtonRepeatDrop:{which}')
        self.drop(f'TextButtonRepeatSelect:{which}')
        self.container_drop(f'AppFrameRepeatSelection:{which}')
        self.dv_remove(group='SelectedFiles', branch=f'Repeat:{which}')

        del self._directories[which]
        del self._repeat_names[which]

        del self.dv('RepeatData')[f'Repeat:{which}']
        self.dv_remove(group='Directories', branch=f'Repeat:{which}')

        if self.dv_check(group='DataGroups', branch=f'Repeat:{which}'):  # drop associated data groups
            self.dv_remove(group='DataGroups', branch=f'Repeat:{which}')

        # drop the repeat entry from the cache settings
        _ = supports.json_dict_push(rf'{supports.__cache__}\settings.json', behavior='read')
        del _['DirectorySettings'][f'Repeat:{which}']
        if len(self._repeat_names) == 1:  # clean-up repeat data for single-repeat data
            del _['DirectorySettings'][f'Repeat:0']['Name']
            del _['DirectorySettings'][f'Repeat:0']['Uncommon']
            del _['DirectorySettings'][f'CommonOutputFolder']

        supports.json_dict_push(rf'{supports.__cache__}\settings.json', _, behavior='replace')

        if len(self._directories) < 2:
            self['AppFrameRepeatContainer'].grid_remove()

            # remove repeat identifiers from file names
            for file in self.dv()['SelectedFiles']['Repeat:0'].keys():
                self[f'AppCheckbutton0:{file}']['text'] = file
        self._update_repeat_tab(0)

    def __add_repeat(self):
        """Internal method that loads a repeat directory into the program."""

        directory = base.configure_directory(self.tkID, 'Repeat', forbidden=self._directories.values(),
                                             forbidden_message='Selected directory has already been added.')

        # cleanup repeat placeholder data to set up repeats properly
        if 0 in self._repeat_names and self._repeat_names[0] is None: del self._repeat_names[0];

        if directory:
            file_folders = [i.removesuffix('.vsi') for i in os.listdir(directory) if i.endswith('.vsi')]
            if not file_folders:  # exit if there are no image files in the provided folder
                supports.tprint(f'No image files in folder {directory}.')
                return
            self['AppFrameRepeatContainer'].grid_remove()

            self.dv_define('InputFolder', tk.StringVar(self, directory), group='Directories',
                           branch=f'Repeat:{self._repeats}')

            # check for existence of predefined saves output
            out_dir = directory + ' (processed)'
            self.dv_define('DefaultOutputFolder', tk.StringVar(self, out_dir), group='Directories',
                           branch=f'Repeat:{self._repeats}')  # set default before checking cache
            try:
                out_dir = supports.json_dict_push(rf'{supports.__cache__}\saves.json', behavior='read'
                                                  )[directory]['OutputFolder']
            except (KeyError, FileNotFoundError): pass;

            self.dv_define('OutputFolder', tk.StringVar(self, out_dir), group='Directories',
                           branch=f'Repeat:{self._repeats}')

            self.__post_available_channels(self._repeats)

            # determine repeat name
            split_input = self['DirectoryEntryInputFolder'].get().split('/')
            repeat_name = None; primary_name = None; parent_dir = None; child_dir = None
            for n, d in enumerate(reversed(directory.split('/'))):
                if d != split_input[-(n + 1)]:
                    repeat_name = d; primary_name = split_input[-(n + 1)]; parent_dir = '/'.join(split_input[:-(n + 1)])
                    if n != 0:
                        child_dir = '/'.join(split_input[-n:])
                    break

            self._repeat_names[0] = primary_name

            self._repeat_names[self._repeats] = repeat_name

            _ = {'Repeat:0': {
                'Name': primary_name,
            },
                 f'Repeat:{self._repeats}': {
                     'Name': repeat_name,
                 }}
            self.dv_set('RepeatData', _, update=True)

            # add file checkbuttons in a new repeat container
            self.add(AppFrame, sid=f'RepeatSelection:{self._repeats}', container='AppFrameFileSelectionContainer',
                     hide=True)

            common_name = self.define_common_name()
            repeat_uncommon = self.dv("RepeatData")[f'Repeat:{self._repeats}']['Uncommon']
            for folder in supports.sort(file_folders):
                elem = self.add(AppCheckbutton, text=rf'{folder} {repeat_uncommon}',
                                sid=f'{self._repeats}:{folder}', container=f'AppFrameRepeatSelection:{self._repeats}')
                self.dv_define(folder, elem.selection, group='SelectedFiles', branch=f'Repeat:{self._repeats}')

            for branch, files in self.dv()['SelectedFiles'].items():
                if branch != rf'Repeat:{self._repeats}':
                    _ruc = self.dv("RepeatData")[f'{branch}']['Uncommon']
                    for file in files.keys():
                        # change checkbutton text to match repeat setup
                        self[f'AppCheckbutton{branch.split(":")[-1]}:{file}']['text'] = rf'{file} {_ruc}'

            # store folders in cache files
            if child_dir is not None: common_out_dir = rf'{parent_dir}/{common_name}/{child_dir} (processed)';
            else: common_out_dir = rf'{parent_dir}/{common_name} (processed)';

            # set default before checking cache
            self.dv_set('DefaultCommonOutputFolder', common_out_dir, group='Directories')

            try:
                common_out_dir = supports.json_dict_push(rf'{supports.__cache__}\saves.json', behavior='read'
                                                  )[self._directories[0]]['CommonOutputFolder']
            except (KeyError, FileNotFoundError): pass;

            self.dv_set('CommonOutputFolder', common_out_dir, group='Directories')

            supports.json_dict_push(rf'{supports.__cache__}\settings.json', behavior='update',
                                    params={'DirectorySettings': {f'Repeat:{self._repeats}': {
                                        'InputFolder': directory,
                                        'OutputFolder': directory + ' (processed)'
                                    }, 'CommonOutputFolder': common_out_dir}}, fetch=True)

            # check for channel curation
            cache = supports.json_dict_push(rf'{supports.__cache__}\saves.json', behavior='read')
            try:
                cache = cache[directory]['LinkSetup']
                if set(self.dv_get('InputFolder', group='Directories', branch=None)) == set(cache['Criterion']):
                    self.dv_set('CuratedChannels', cache['AvailableChannels'])
            except KeyError: pass;

            # add data selection buttons
            if not self.exists('TextButtonRepeatSelect:0'):
                self.add(TextButton, text=f'{primary_name}', sid='RepeatSelect:0', column=1,
                         command=partial(self._update_repeat_tab, 0), font='Arial 8 bold',
                         container='AppFrameRepeatContainer')
            else:
                self[f'TextButtonRepeatSelect:0']['text'] = f'{primary_name}'.upper()

            self.add(TextButton, text='×', sid=f'RepeatDrop:{self._repeats}', font='Arial 8',
                     command=partial(self.__drop_repeat, self._repeats), container='AppFrameRepeatContainer')

            dir_tt = rf'Repeat directory is located at {directory}.'
            self.add(TextButton, text=f'{repeat_name}', sid=f'RepeatSelect:{self._repeats}',
                     command=partial(self._update_repeat_tab, self._repeats), font='Arial 8', column=1, prow=True,
                     container='AppFrameRepeatContainer', tooltip=dir_tt)

            self._directories[self._repeats] = directory
            self._repeats += 1
            self['AppFrameRepeatContainer'].grid()

    def __load_directory(self, *_):
        """Internal method that loads the current selected directory .vsi files."""
        selected_directory = self['DirectoryEntryInputFolder'].get()
        self['AppFrameFileSelectionContainer'].grid_remove()
        if selected_directory == self.dv_get('InputFolder', group='Directories', branch='Repeat:0'):
            pass  # if the chosen folder is the current folder do nothing
        elif selected_directory == '':
            self.dv_remove(group='Directories')  # set default output directory
            self.dv_define('InputFolder', tk.StringVar(self, ''), group='Directories', branch='Repeat:0')
            self.dv_define('OutputFolder', tk.StringVar(self, ''), group='Directories', branch='Repeat:0')
            self.dv_define('CommonOutputFolder', tk.StringVar(self, ''), group='Directories')
            self.dv_define('DefaultOutputFolder', tk.StringVar(self, ''), group='Directories', branch='Repeat:0')
            self.dv_define('DefaultCommonOutputFolder', tk.StringVar(self, ''), group='Directories')

            self.container_drop('AppFrameFileSelectionContainer')  # remove existing files in the container

            # store input folder in cache file
            supports.json_dict_push(rf'{supports.__cache__}\settings.json', behavior='update',
                                    params={'DirectorySettings': {'Repeat:0': {
                                        'InputFolder': selected_directory,
                                        'OutputFolder': selected_directory + ' (processed)'
                                    }}})

            self.dv_remove(group='SelectedFiles')

            # self.__update_selection()  # update selection after setting up check buttons
            self['AppFrameFileSelectionContainer'].grid_remove()
            self['TextButtonAddRepeat'].grid_remove()
            self['TextButtonSelectAll'].grid_remove()
            self['TextButtonDeselectAll'].grid_remove()
            self._repeats = 0
            self._directories = {}
        else:
            """Since the PreprocessingOptions reload is triggered by InputFolder trace, the OutputFolder must be 
            changed before the InputFolder, to avoid reloading the ProcessingOptions with the wrong InputFolder."""

            _ = self._repeat_names.copy().keys()
            for repeat in _:
                if repeat != 0:
                    self.__drop_repeat(repeat)

            out_dir = selected_directory + ' (processed)'
            self.dv_set('DefaultOutputFolder', out_dir, group='Directories', branch='Repeat:0')
            try:
                _ = supports.json_dict_push(rf'{supports.__cache__}\saves.json', behavior='read'
                                                  )[selected_directory]['OutputFolder']
                if _:  # catch instances where the saved folder is empty
                    out_dir = _
            except (KeyError, FileNotFoundError): pass;

            file_folders = [i.removesuffix('.vsi') for i in os.listdir(selected_directory) if i.endswith('.vsi')]
            if not file_folders:  # exit if there are no image files in the provided folder
                supports.tprint(f'No image files in folder {selected_directory}.')
                return

            # update global directories
            self.dv_set('OutputFolder', out_dir, group='Directories', branch='Repeat:0')
            self.dv_set('InputFolder', selected_directory, group='Directories', branch='Repeat:0')
            self.dv_set('RepeatData', {'Repeat:0': {'Name': None, 'Uncommon': None}}, update=True)
            self._repeat_names[0] = None

            self.container_drop('AppFrameFileSelectionContainer')  # remove existing files in the container
            self.container_drop('AppFrameRepeatContainer')  # drop repeats
            self.__post_available_channels(0)  # start loading channels for the valid path

            try:  # drop selected files if any exist
                self.dv_remove(group='SelectedFiles')
            except KeyError:
                pass

            self.add(AppFrame, sid='RepeatSelection:0', container='AppFrameFileSelectionContainer')

            for folder in supports.sort(file_folders):
                elem = self.add(AppCheckbutton, text=folder, sid=f'0:{folder}', container='AppFrameRepeatSelection:0')
                self.dv_define(folder, elem.selection, group='SelectedFiles', branch='Repeat:0')

            # store input folder in cache file
            supports.json_dict_push(rf'{supports.__cache__}\settings.json', behavior='update',
                                params={'DirectorySettings': {'Repeat:0': {
                                    'InputFolder': selected_directory,
                                    'OutputFolder': selected_directory + ' (processed)'
                                }}})
            # self.__update_selection()  # update selection after setting up check buttons
            self['TextButtonAddRepeat'].grid()
            self._directories[0] = selected_directory
            self._repeats += 1
            self['TextButtonSelectAll'].grid()
            self['TextButtonDeselectAll'].grid()
        self['AppFrameFileSelectionContainer'].grid()

    @supports.thread_daemon
    @supports.timer
    def __post_available_channels(self, which):
        """Internal method that posts the available color channels as a global variable."""

        saves = supports.json_dict_push(rf'{supports.__cache__}\saves.json', behavior='read')
        in_folder = self.dv_get('InputFolder', group='Directories', branch=f'Repeat:{which}')
        try:  # only iterate if no saved channels can be found for that data set
            available_channels = saves[in_folder]['AvailableChannels']
            if not available_channels:
                raise KeyError
        except KeyError:
            rname = self._repeat_names[which] if self._repeat_names[which] is not None else in_folder
            supports.tprint(f'Looking for available color channels for {rname}.')
            self.dv_set('AvailableChannels', [])  # ensure that channels are blocked, until new are found
            _in, _out = multiprocessing.Pipe(duplex=False)  # open pipe
            process = multiprocessing.Process(target=self.channel_search, args=(in_folder, _out), daemon=True)
            process.start()
            process.join()
            available_channels = _in.recv()
            process.close()

            update_dict = {in_folder: {'AvailableChannels': available_channels,}}
            supports.json_dict_push(rf'{supports.__cache__}\saves.json', params=update_dict, behavior='update')

        self.dv_set('RepeatData', {f'Repeat:{which}': {'Channels': available_channels}}, update=True)

        channel_data = [i['Channels'] for i in self.dv_get('RepeatData').values()]
        all_channels = []
        for channels in channel_data:
            for c in channels:
                if c not in all_channels:
                    all_channels.append(c)

        self.dv_set('AvailableChannels', all_channels)


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

        self.add(AppLabel, text='Force rescale:', sid='ForceResolution', tooltip=True)
        self.add(SettingEntry, sid='ForceResolution', default='', vartype=float, prow=True, padx=(120, 0))

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
        self.add(AppFrame, sid='RepeatContainer', hide=False)

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
        self.update_repeat_selection('Repeat:0')
        self.add(TextButton, sid='LoadPreviousSettings', text='Load Previous Preprocessing', warning=True,
                     command=self.load_previous_settings)

        self.add(AppButton, text='PREPROCESS', command=self.preprocess)

        self.parent['SelectionMenuMaskChannel'].selection.trace_add('write', self.__show_orc_button)

        self.__show_orc_button()
        self.update_repeats()

    def __load_orientation_reference_creator(self):
        # prior to opening the OrientationReferenceCreator the MaskChannel must be saved to the Settings.json
        _ = {'CollectionPreprocessSettings': {
            'MaskChannel': self.parent['SelectionMenuMaskChannel'].get(),
            'SampleType': self.parent['SelectionMenuSampleType'].get(),
            'MaskSelection': self['SelectionMenuImageMask'].get()
        }}

        _out = self.dv_get('OutputFolder', group='Directories', branch='Repeat:0')
        base.directory_checker(_out, clean=False)
        supports.json_dict_push(rf'{_out}\Settings.json', params=_, behavior='update')

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

    @staticmethod
    def setting_package(file):
        _ = (f'TextCheckbutton:{file}', f'SelectionMenuRotate:{file}', f'SettingEntryAlign:{file}',
                     f'SelectionMenuMaskingMethod:{file}', f'SettingEntryMinFields:{file}',
                     f'SelectionMenuMaskingMethod:{file}', f'SelectionMenuMaskShift:{file}',
                     f'SettingEntryIterativeFidelity:{file}', f'SettingEntryOnsetIntensitySpan:{file}')
        return _

    def __update_image_mask_cache(self, *_):
        value = self['SelectionMenuImageMask'].get()
        if value not in ('Select Option', 'Add ...'):
            supports.post_cache({'PreprocessingSettings': {'ImageMask': value}})

    def load_previous_settings(self):
        """Method that loads the settings used during latest preprocessing."""
        _out = self.dv_get('OutputFolder', group='Directories', branch=self.selected_repeat)
        settings = supports.json_dict_push(rf'{_out}\Settings.json', behavior='read')
        if settings['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            files = list(settings['PreprocessedParameters'].keys())
            _ = []
            for file in self.tags['ImageSelection']:  # set activity status
                if file.split(':')[-2] == self.selected_repeat.split(':')[-1]:  # only act on selected repeat
                    name = file.split(':')[-1]
                    if name in files:
                        self[file].set(True)
                        for k, j in zip(('Rotate', 'MinFields', 'Align', 'MaskShift', 'MaskingMethod',
                                         'IterativeFidelity', 'OnsetIntensitySpan'),
                                        ('SelectionMenu', 'SettingEntry', 'SettingEntry', 'SelectionMenu',
                                         'SelectionMenu', 'SettingEntry', 'SettingEntry')):
                            self[f'{j}{k}:{self.selected_repeat}:{name}'].set(
                                settings['PreprocessedParameters'][name]['FieldParameters'][k])
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
        if not self.preprocess_check(): return

        mask_data = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                            behavior='read')[self.parent['SelectionMenuSampleType'].get()]
        if 'OrientationReference' not in mask_data[self['SelectionMenuImageMask'].get()]:
            for tag in self.tags['ImageSelection']:
                file = tag.split(':')[-1]
                if self[f'SelectionMenuRotate:{file}'].get() == 'Auto' and self[f'TextCheckbutton:{file}'].get() is True:
                    messagebox.showerror('Error', "Cannot use Rotate 'Auto' with no defined orientation reference.",)
                    return
        else:
            _ = self.dv_get('OutputFolder', group='Directories', branch='Repeat:0')
            if not os.path.exists(rf'{_}\_misc\OrientationReference.tiff'):
                supports.tprint('Orientation reference for mask {} is configured with {} from a different sample set.'.format(
                    self['SelectionMenuImageMask'].get(),
                    mask_data[self['SelectionMenuImageMask'].get()]['OrientationReference']['SampleName']))

        for r in self.dv_get('RepeatData'):
            _ = {}
            _out = self.dv_get('OutputFolder', group='Directories', branch=r)
            for file in self.files:
                if file.startswith(r):
                    file_name = self.split_file_name(file)[1]
                    _[file_name] = {
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
                    'InputFolder': self.dv_get('InputFolder', group='Directories', branch=r),
                    'OutputFolder': _out},
                'CollectionPreprocessSettings': {
                    'MaskSelection': self['SelectionMenuImageMask'].get(),
                    'MaskChannel': self.parent['SelectionMenuMaskChannel'].get(),
                    'SampleType': self.parent['SelectionMenuSampleType'].get(),
                    'ForceResolution': self['SettingEntryForceResolution'].get()},
                'IndividualPreprocessSettings': _
            }

            # create output folder if it does not exist
            if not os.path.isdir(_out):
                os.makedirs(_out)
                supports.tprint(f'Output folder created at: {_out}')

            supports.json_dict_push(rf'{_out}\Settings.json', update_dict, 'update')

        common_out = self.dv_get('CommonOutputFolder', group='Directories', branch=None)
        common_out = common_out[0] if isinstance(common_out, list) else common_out

        if common_out:
            if not os.path.isdir(common_out):
                os.makedirs(common_out)

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
        self.add(SettingEntry, sid='FieldWidth', prow=True, padx=(200, 0), vartype=float)
        self.add(AppLabel, text='x', sid='FieldDimensionTimes', prow=True, padx=(260, 0))
        self.add(SettingEntry, sid='FieldHeight', prow=True, padx=(280, 0), vartype=float)

        self.add(AppLabel, text='Field units:')
        self.add(SelectionMenu, sid='FieldUnits', prow=True, padx=(200, 0), options=('nm', 'µm', 'mm', 'cm'))

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
        self.add(AppLabel, text='Force rescale:', sid='ForceResolution', tooltip=True)
        self.add(SettingEntry, sid='ForceResolution', default='', vartype=float, prow=True, padx=(120, 0))

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
        self.add(AppFrame, sid='RepeatContainer', hide=False)

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
        self.update_repeat_selection('Repeat:0')
        self.add(TextButton, sid='LoadPreviousSettings', text='Load Previous Preprocessing', warning=True,
                 command=self.load_previous_settings)

        self.add(AppButton, text='PREPROCESS', command=self.preprocess)
        self.update_repeats()

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

    @staticmethod
    def setting_package(file):
        _ = (f'TextCheckbutton:{file}', f'SettingEntryAlign:{file}', f'SelectionMenuMaskingMethod:{file}',
                     f'SettingEntryIterativeFidelity:{file}')
        return _

    def __add_mask(self):
        level = TopLevelWidget(self); level.title('Mask Creator')
        level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')
        content = SingleFieldMaskCreator(level.main.interior, tie=self); content.grid()

    def load_previous_settings(self):
        """Method that loads the settings used during latest preprocessing."""
        _out = self.dv_get('OutputFolder', group='Directories', branch=self.selected_repeat)
        settings = supports.json_dict_push(rf'{_out}\Settings.json', behavior='read')
        if settings['CollectionPreprocessSettings']['SampleType'] == 'Single-Field':
            files = list(settings['PreprocessedParameters'].keys())
            _ = []
            for file in self.tags['ImageSelection']:  # set activity status
                if file.split(':')[-2] == self.selected_repeat.split(':')[-1]:  # only act on selected repeat
                    name = file.split(':')[-1]
                    if name in files:
                        self[file].set(True)
                        for k, j in zip(('Align', 'MaskingMethod', 'IterativeFidelity'),
                                        ('SettingEntry', 'SelectionMenu', 'SettingEntry')):
                            self[f'{j}{k}:{self.selected_repeat}:{name}'].set(
                                settings['PreprocessedParameters'][name]['FieldParameters'][k])
                    else:
                        self[file].set(False)
            self['SelectionMenuImageMask'].set(settings['CollectionPreprocessSettings']['MaskSelection'])
            self.parent['SelectionMenuMaskChannel'].set(settings['CollectionPreprocessSettings']['MaskChannel'])
        else:
            supports.tprint('Previous settings have the wrong formatting.')

    def preprocess(self):
        """Internal method that runs NTSA preprocessing according to the selected settings."""
        if not self.preprocess_check(): return;

        # extract settings and update Settings.json for each repeat
        for r in self.dv_get('RepeatData'):
            _ = {}
            _out = self.dv_get('OutputFolder', group='Directories', branch=r)
            for file in self.files:
                if file.startswith(r):
                    file_name = self.split_file_name(file)[1]
                    _[file_name] = {
                        'Align': self[f'SettingEntryAlign:{file}'].get(),
                        'MaskingMethod': self[f'SelectionMenuMaskingMethod:{file}'].get(),
                        'IterativeFidelity': self[f'SettingEntryIterativeFidelity:{file}'].get()
                    }

            update_dict = {
                'DirectorySettings': {
                    'InputFolder': self.dv_get('InputFolder', group='Directories', branch=r),
                    'OutputFolder': _out},
                'CollectionPreprocessSettings': {
                    'MaskSelection': self['SelectionMenuImageMask'].get(),
                    'MaskChannel': self.parent['SelectionMenuMaskChannel'].get(),
                    'SampleType': self.parent['SelectionMenuSampleType'].get(),
                    'ForceResolution': self['SettingEntryForceResolution'].get()},
                'IndividualPreprocessSettings': _
            }

            # create output folders if it does not exist
            if not os.path.isdir(_out):
                os.makedirs(_out)
                supports.tprint(f'Output folder created at: {_out}')

            supports.json_dict_push(r'{}\Settings.json'.format(_out), update_dict, 'update')

        common_out = self.dv_get('CommonOutputFolder', group='Directories', branch=None)
        common_out = common_out[0] if isinstance(common_out, list) else common_out

        if common_out:
            if not os.path.isdir(common_out):
                os.makedirs(common_out)

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
        for r in self.dv_get('RepeatData'):
            _out = self.dv_get('OutputFolder', group='Directories', branch=r)
            update_dict = {
                'DirectorySettings': {
                    'InputFolder': self.dv_get('InputFolder', group='Directories', branch=r),
                    'OutputFolder': _out},
                'CollectionPreprocessSettings': {
                    'SampleType': self.parent['SelectionMenuSampleType'].get(),
                    'MaskChannel': self.parent['SelectionMenuMaskChannel'].get()}
            }

            # create output folder if it does not exist
            if not os.path.isdir(_out):
                os.makedirs(_out)
                supports.tprint('Output folder created at: {}'.format(_out))

            supports.json_dict_push(rf'{_out}\Settings.json', update_dict, 'update')

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


class ChannelLinker(PopupWindow):
    def __init__(self, parent, tie, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.tie = tie
        self.channels = tie.dv_get('AvailableChannels')
        self._active = self.channels
        self._removed = []
        self.n = 0
        self.selection_tags = {}
        self._links = {}
        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Link Channels')
        self.add(AppFrame, sid='LinkContainer')
        self.add(TextButton, text='Add Link', command=self._add_link)
        self.add(AppButton, text='Save Links', pady=(5, 0), command=self._save_links)
        self.add(AppButton, text='Cancel', command=self.cancel_window, pady=(5, 0), padx=(140, 0), prow=True)

        self._auto_setup()

    def _auto_setup(self):
        cache = supports.json_dict_push(rf'{supports.__cache__}\saves.json', behavior='read')

        try: cache = cache[self.tie.dv_get('InputFolder', group='Directories', branch='Repeat:0')]['LinkSetup'];
        except KeyError: return;

        if set(cache['Criterion']) == set(self.tie.dv_get('InputFolder', group='Directories', branch=None)):
            setup = []
            for k, v in cache['Linker'].items():
                if k != v:
                    if v not in setup:
                        setup.append(v)
                        self._add_link()
                        self[f'SelectionMenuDominantOptions:{self.n - 1}'].set(v)
                    self[f'SelectionMenuLinkOptions:{self.n - 1}:{self.selection_tags[self.n - 1]}'].set(k)

    def _add_link(self):
        if self.channels:
            self.selection_tags[self.n] = 0
            self.add(AppFrame, sid=f'Link:{self.n}', container='AppFrameLinkContainer')
            self.add(TextButton, text='×', sid=f'Drop:{self.n}', font='Arial 8', container=f'AppFrameLink:{self.n}',
                     command=partial(self._drop_link, self.n))

            self.add(AppFrame, sid=f'LinkSelections:{self.n}', container=f'AppFrameLink:{self.n}',
                     prow=True, column=1)
            _ = self.add(SelectionMenu, sid=f'DominantOptions:{self.n}', placeholder='Select Dominant',
                     options=self.channels, container=f'AppFrameLinkSelections:{self.n}', padx=5)
            _.trace_add('write', partial(self._trigger_add_selection_link, self.n))
            self.add(TextButton, text='+', sid=f'LinkAdd:{self.n}', font='Arial 8', container=f'AppFrameLink:{self.n}',
                     prow=True, command=partial(self._add_selection_link, self.n), column=2)
            self.add(TextButton, text='-', sid=f'LinkDrop:{self.n}', font='Arial 8', container=f'AppFrameLink:{self.n}',
                     prow=True, hide=True, command=partial(self._drop_selection_link, self.n), column=3)
            self.n += 1

    def _trigger_add_selection_link(self, n, *_):
        if self.selection_tags[n] == 0:
            self._add_selection_link(n)

    def _add_selection_link(self, n):
        self.selection_tags[n] += 1
        _ = self.add(SelectionMenu, sid=f'LinkOptions:{n}:{self.selection_tags[n]}', prow=True, tag=f'Link:{n}',
                 container=f'AppFrameLinkSelections:{n}', options=self.channels, column=self.selection_tags[n],
                 overwrite=True, padx=5, placeholder='Select Link')
        _.trace_add('write', partial(self._update_link_profile, n, self.selection_tags[n]))
        if self.selection_tags[n] > 1: self[f'TextButtonLinkDrop:{n}'].grid();

    def _drop_selection_link(self, n):
        self.drop(f'SelectionMenuLinkOptions:{n}:{self.selection_tags[n]}')
        self.selection_tags[n] -= 1
        if self.selection_tags[n] < 2: self[f'TextButtonLinkDrop:{n}'].grid_remove();

    def _drop_link(self, n):
        try:
            for tag in self.tags[f'Link:{n}']:
                del self._links[self[tag].get()]
        except KeyError: pass;
        self.container_drop(f'AppFrameLink:{n}', dtype='destroy', kill=True)
        del self.selection_tags[n]
        self.drop(f'AppFrameLinkSelections:{n}')

    def _update_link_profile(self, n, t, *_):
        self._links[self[f'SelectionMenuLinkOptions:{n}:{t}'].get()] = self[f'SelectionMenuDominantOptions:{n}'].get()

    def _save_links(self):
        link_criterion = self.tie.dv_get('InputFolder', group='Directories', branch=None)

        active_channels = []; linker = {}
        for c in self.channels:
            if c not in self._links:
                active_channels.append(c)
                linker[c] = c
            else:
                linker[c] = self._links[c]

        _ = {}
        for path in link_criterion:
            _[path] = {'LinkSetup': {
                'Criterion': link_criterion,
                'AvailableChannels': active_channels,
                'Linker': linker,
            }}
        supports.json_dict_push(rf'{supports.__cache__}\saves.json', params=_, behavior='update')
        self.tie['SelectionMenuMaskChannel'].update_options(active_channels)
        self.cancel_window()


class PreprocessingOptions(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._active_repeats = []
        self._activation_bypass = True

    def __traces__(self):
        self.dv_trace('InputFolder', 'write', self.reload, group='Directories', branch='Repeat:0')
        self.dv_trace('AvailableChannels', 'write', self.__available_channel_listener)

    def load(self):
        if not self.is_empty():
            self._update_repeats()
            self.update_container_choice()

            try: self['MultiFieldPreprocessingOptionsContainer'].update_repeats();
            except KeyError: pass;

            try: self['SingleFieldPreprocessingOptionsContainer'].update_repeats();
            except KeyError: pass;

        super().load()

    def __base__(self):
        self.add(AppTitle, text='Preprocessing')

        # set up directories
        self.add(AppLabel, text='Enter output directory:', sid='OutputFolder')
        self.add(DirectoryEntry, sid='OutputFolder', trigger_function=self.__on_dir_change,
                     forbidden=self.dv_get(dv='InputFolder', group='Directories', branch=None))
        self.add(TextButton, text='Default Directory', command=self._set_default_directory)

        self.add(AppLabel, text='Sample type:', sid='SampleType', tooltip=True)
        _ = self.add(SelectionMenu, sid='SampleType', options=('Multi-Field', 'Single-Field', 'Zero-Field'), prow=True,
                     padx=(120, 0), width=20)
        _.trace_add('write', self.update_container_choice)
        self.add(AppLabel, text='Mask channel:', sid='MaskChannel', tooltip=True)

        _ = self.dv_get('AvailableChannels') if not self.dv_get('CuratedChannels') else self.dv_get('CuratedChannels')
        elem = self.add(SelectionMenu, sid='MaskChannel', options=_, prow=True, padx=(120, 0), width=20)
        self.add(LoadingCircle, size=24, width=6, bg=supports.__cp__['bg'], aai=4, stepsize=.7, sid='MaskChannel',
                 prow=True, padx=(310, 0), pady=(0, 5))
        if not _:
            elem.disable('Loading')
            self['LoadingCircleMaskChannel'].start()
        self.add(TextButton, text='Link Channels', command=self._open_channel_linker)

        self.add(MultiFieldPreprocessingOptions, sid='Container', hide=True)
        self.add(SingleFieldPreprocessingOptions, sid='Container', hide=True)
        self.add(ZeroFieldPreprocessingOptions, sid='Container', hide=True)

        self.update_container_choice()
        self._update_repeats()

    def _open_channel_linker(self):
        level = TopLevelWidget(self); level.title('Channel Linker')
        level.geometry(f'{int(self.winfo_screenwidth() * .35)}x{int(self.winfo_screenheight() * .25)}')

        content = ChannelLinker(level.main.interior, name='ChannelLinker', tie=self)
        content.grid()

    def _toggle_repeat_visibility(self): pass;

    def _set_default_directory(self):
        """Internal function that resets the selected output directory to the default directory."""
        if self['AppFrameRepeatContainer'].winfo_ismapped():
            self['DirectoryEntryOutputFolder'].set(self.dv_get('DefaultCommonOutputFolder', group='Directories'))
        else:
            self['DirectoryEntryOutputFolder'].set(self.dv_get('DefaultOutputFolder', group='Directories',
                                                               branch='Repeat:0'))
        self.__on_dir_change()  # ensure that the change is cached

    def _update_repeats(self, *_):
        """Internal method that updates the repeat functionality."""
        repeats = self.dv_get('RepeatData')
        if self._active_repeats != list(repeats.keys()) or self._activation_bypass is True:
            if len(repeats) > 1:
                # set output folder entry to the common entry if there are repeats
                self['DirectoryEntryOutputFolder'].set(self.dv_get('CommonOutputFolder', group='Directories'))
                self['AppLabelOutputFolder']['text'] = 'Enter common output directory:'

                _active_repeats = []
                for k, v in repeats.items():
                    if k not in self._active_repeats:
                        _active_repeats.append(k)

                try: self['MultiFieldPreprocessingOptionsContainer']['AppFrameRepeatContainer'].grid();
                except KeyError: pass;

                try: self['SingleFieldPreprocessingOptionsContainer']['AppFrameRepeatContainer'].grid();
                except KeyError: pass;

                self._active_repeats = _active_repeats
            else:  # update default directory entry to match repeat structure
                # check for existence of predefined saves output
                out_dir = self.dv_get('OutputFolder', group='Directories', branch='Repeat:0')
                try:
                    out_dir = supports.json_dict_push(rf'{supports.__cache__}\saves.json', behavior='read'
                                                      )[self.dv_get('InputFolder', group='Directories',
                                                                    branch='Repeat:0')]['OutputFolder']
                except (KeyError, FileNotFoundError): pass;

                self['DirectoryEntryOutputFolder'].set(out_dir)
                self['AppLabelOutputFolder']['text'] = 'Enter output directory:'
                self._active_repeats = []

                try: self['MultiFieldPreprocessingOptionsContainer']['AppFrameRepeatContainer'].grid_remove();
                except KeyError: pass;

                try: self['SingleFieldPreprocessingOptionsContainer']['AppFrameRepeatContainer'].grid_remove();
                except KeyError: pass;

        self._activation_bypass = False  # this ensures that the function is always run when the tab is first loaded in

    def _change_output_folder(self, which):
        _forbidden = self.dv_get('InputFolder', group='Directories', branch=None, as_dict=True)
        directory = base.configure_directory(self.tkID, which, forbidden=_forbidden.keys(),
                                             forbidden_message='Selected directory is a configured input folder.')

        if directory:
            # update entries and cache files
            self[f'TextButtonOutputFolder:{which.split(":")[-1]}']['text'] = directory
            _ = {_forbidden[which]: {'OutputFolder': directory}}
            supports.json_dict_push(rf'{supports.__cache__}\saves.json', _, behavior='update')

            _ = {'DirectorySettings': {which: {'OutputFolder': directory}}}
            supports.json_dict_push(rf'{supports.__cache__}\settings.json', _, behavior='update')

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
                self[f'{value}PreprocessingOptionsContainer'].update_repeat_selection('Repeat:0')
            self[f'{value}PreprocessingOptionsContainer'].grid()

            if 'RepeatSectionButton' in self.tags:
                del_indices = []
                for en, repeat in enumerate(self.tags['RepeatSectionButton']):
                    n = f'Repeat:{repeat.split(":")[-1]}'
                    try:
                        if value != 'ZeroField':
                            self[repeat]['state'] = 'normal'
                            self[repeat]['command'] = partial(self[f'{value}PreprocessingOptionsContainer'].update_repeat_selection, n)
                        else:
                            self[repeat]['state'] = 'disabled'
                            self[repeat]['command'] = None
                    except KeyError:
                        del_indices.append(en)

                if del_indices:
                    for en in del_indices:
                        try:
                            del self.tags['RepeatSectionButton'][en]
                        except IndexError: pass;

    def __available_channel_listener(self, *_):
        if self['SelectionMenuMaskChannel']['state'] == 'disabled':
            self['SelectionMenuMaskChannel'].enable()
            self['LoadingCircleMaskChannel'].stop()
        elif not self.dv_get('AvailableChannels'):
            self['SelectionMenuMaskChannel'].disable()
            self['LoadingCircleMaskChannel'].start()
        self['SelectionMenuMaskChannel'].update_options(self.dv_get('AvailableChannels'))

    def __on_dir_change(self):
        """Internal method that updates local cache for the output folder."""
        out_path = self['DirectoryEntryOutputFolder'].get()
        if self['AppFrameRepeatContainer'].winfo_ismapped():
            settings_save = {'DirectorySettings': {'CommonOutputFolder': out_path}}
            saves_save = {}
            for repeat in self._active_repeats:
                _ = self.dv_get('InputFolder', group='Directories', branch=repeat)
                saves_save[_] = {'CommonOutputFolder': out_path}
            self.dv_set('CommonOutputFolder', out_path, group='Directories')
        else:
            settings_save = {'DirectorySettings': {'Repeat:0': {'OutputFolder': out_path}}}
            saves_save = {self.dv_get('InputFolder', group='Directories', branch='Repeat:0'):
                                         {'OutputFolder': out_path}}
            self.dv_set('OutputFolder', out_path, group='Directories', branch='Repeat:0')

        # update directory settings in cache files
        supports.json_dict_push(rf'{supports.__cache__}\settings.json', settings_save, behavior='update')

        # match the selected directory with the Settings.json
        if os.path.exists(rf'{out_path}\Settings.json'):
            supports.json_dict_push(rf'{out_path}\Settings.json', params=settings_save, behavior='update')

        # match the save data to automatically target the selected directory in the future
        supports.json_dict_push(rf'{supports.__cache__}\saves.json', saves_save, behavior='update')


class MaskGallery(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._current_page = None; self._current_repeat = None
        self.folder_structure = None; self.selected_files = None

    def __traces__(self):
        self.dv_trace('LatestPreprocessedFile', 'write', self.__auto_load_image)

    def __base__(self):
        self.load_settings()

        self.add(AppTitle, text='Mask Gallery')

        repeats = self.dv()['SelectedFiles']
        self.rdata = rdata = self.dv_get('RepeatData')

        self.r = 1
        if len(repeats) > 1:
            self.r = 0
            _ = {}
            for k, v in rdata.items():
                _[v['Name']] = partial(self.change_repeat, k)
                self.r += 1

            self.add(AppLabel, text='Repeat: ', sid='RepeatSelection')
            self.add(SelectionMenu, options=tuple(_), sid='RepeatSelection', prow=True, padx=(70, 0), commands=_,
                     default=0)

        for repeat in repeats:
            self.add(AppFrame, sid=f'{repeat}:ImageContainer', hide=True)
            self.add(AppFrame, sid=f'{repeat}:PageChange', container=f'AppFrame{repeat}:ImageContainer', hide=True)
            self.add(AppFrame, sid=f'{repeat}:ImagePage0', container=f'AppFrame{repeat}:ImageContainer', hide=True)
        self._current_page = 0; self._current_repeat = 0

        # show the first repeat as default
        self['AppFrameRepeat:0:ImageContainer'].grid()
        self['AppFrameRepeat:0:ImagePage0'].grid()

        # construct placeholder containers
        self.ddata = self.dv()['Directories']
        for r, v in self.ddata.items():
            if r != '':
                self._add_image_page(r, v)

    def _add_image_page(self, r, rdata):
        mask_folder = r'{}\_masks for manual control'.format(rdata['OutputFolder'].get())
        selected_files = [k for k, v in self.dv_get(group='SelectedFiles', branch=r, as_dict=True).items() if v is True]

        page_size = 30; page = 0; item = 0
        for file in supports.sort(selected_files):
            self.add(AppFrame, sid=f':{r}:{file}:Container', container=f'AppFrame{r}:ImagePage{page}', pady=(0, 10))
            item += 1
            if item == page_size:
                item = 0; page += 1
                self.add(AppFrame, sid=f'{r}:ImagePage{page}', container=f'AppFrame{r}:ImageContainer', prow=True, hide=True)
        self._pages = page

        if page > 0:
            self.add(TextButton, text='<', command=partial(self.previous_page, None), sid=f'{r}:PreviousPage',
                     container=f'AppFrame{r}:PageChange', font='Arial 10')
            col = 1
            for i in range(page + 1):
                _ = self.add(TextButton, text=str(i), sid=f'{r}:Page{i}', container=f'AppFrame{r}:PageChange', prow=True,
                             column=col, command=partial(self.change_page, i, r), font='Arial 10')
                col += 1
            self.add(TextButton, text='>', command=partial(self.next_page, None), sid=f'{r}:NextPage',
                     container=f'AppFrame{r}:PageChange', prow=True, column=col + 1, font='Arial 10')
            self['TextButtonRepeat:0:Page0']['font'] = 'Arial 10 bold'  # set selection marking
            self.bind_all('<Control-Left>', self.previous_page, add=True)
            self.bind_all('<Control-Right>', self.next_page, add=True)
            self['AppFrameRepeat:0:PageChange'].grid()

        # load existing masks upon initial tab load
        if os.path.exists(mask_folder):
            for file in supports.sort(selected_files):
                if os.path.isfile(rf'{mask_folder}/{file}.png'):
                    self.load_image(file, r)

    def change_repeat(self, repeat):
        if isinstance(repeat, str):
            repeat = int(repeat.split(':')[-1])

        if repeat != self._current_repeat:
            # remove and reset widgets for previous repeat selection
            self[f'AppFrameRepeat:{self._current_repeat}:ImageContainer'].grid_remove()
            self[f'AppFrameRepeat:{self._current_repeat}:PageChange'].grid_remove()

            try:  # change the image page and update attributes
                self.change_page(self._current_page, repeat)
            except KeyError:
                self.change_page(0, repeat)

            # display and update widgets for new selection
            self[f'AppFrameRepeat:{self._current_repeat}:PageChange'].grid()
            self[f'AppFrameRepeat:{self._current_repeat}:ImageContainer'].grid()

    def __auto_load_image(self, *_):
        """Internal method that automatically loads the latest preprocessed mask image into the gallery. The image is
        only loaded if the __base__ has been called prior to calling this method."""

        if self.is_empty() is False:
            file_repeat, file_name = self.dv_get('LatestPreprocessedFile').split(':')[1:]
            self.load_settings()  # reload settings to get data for the new file
            self.load_image(file_name, f'Repeat:{file_repeat}')

    @supports.thread_daemon
    def load_image(self, image, repeat):
        uncommon_tag = ''
        if self.r > 1:
            uncommon_tag = f" {self.rdata[repeat]['Uncommon']}"

        mask_folder = self.ddata[repeat]['OutputFolder'].get()
        self.container_drop(f'AppFrame:{repeat}:{image}:Container')  # drop image container contents
        self.container_drop(f'AppFrame:{repeat}:{image}:Metadata')  # drop metadata container contents
        mask_path = rf'{mask_folder}\_masks for manual control\{image}.png'
        settings = self.settings[repeat]

        # place image in container and store to memory
        _ = self.add(ZoomImageFrame, sid=f':{repeat}:{image}', container=f'AppFrame:{repeat}:{image}:Container')
        _.scroll_scalar = (.75, .75)
        _.set_image(path=mask_path)

        # place metadata in a container next to the image
        self.add(AppFrame, sid=f':{repeat}:{image}:Metadata', container=f'AppFrame:{repeat}:{image}:Container',
                 prow=True, column=1, sticky='n', padx=(20, 0))
        self.add(AppSubtitle, text=f'{image}{uncommon_tag}', sid=f':{repeat}:{image}', pady=(0, 15),
                 container=f'AppFrame:{repeat}:{image}:Metadata')

        try:
            md = settings['PreprocessedParameters'][f'{image}']

            _l1 = ('Quality: Tier {Level:.2f} with Area Deviation {AreaError:.1f}%, and Ratio Deviation '
                   '{RatioError:.3f}%\n\n').format(**md['QualityParameters'])
            _l1 += 'Field Size: ({}, {}), Align Angle: {}°, Scale: {} pix:µm\n\n'.format(
                int(round(md['FieldParameters']['Width'], 0)),
                int(round(md['FieldParameters']['Height'], 0)), round(md['FieldParameters']['Align'], 3),
                round(md['FieldParameters']['ScaleBar'], 3))
            if settings['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
                _l1 += ('Masked with {MaskingMethod!r} and Rotated {Rotate}° with Certainty '
                        '{RotationCertainty:.1f} and Matrix Uniformity {ComparedMatrixUniformity:.3f}').format(
                    **md['FieldParameters'])
            elif settings['CollectionPreprocessSettings']['SampleType'] == 'Single-Field':
                _l1 += 'Masked with {MaskingMethod!r} and Threshold Parameter {TP}'.format(
                    **md['FieldParameters'])

            self.add(AppLabel, text=_l1, sid=f':{repeat}:{image}', container=f'AppFrame:{repeat}:{image}:Metadata',
                     justify='left', overwrite=True)
        except KeyError as e:
            if self.dv_get('Debugger') is True:
                uncommon = self.dv_get('RepeatData')[repeat]['Uncommon']
                supports.tprint(f'Parameter load-in for image mask {image} {uncommon} failed with {e!r}.')

        self.add(TextButton, text='View Mask', function='open_image', data=mask_path, overwrite=True,
                 sid=f'LowRes:{repeat}:{image}', container=f'AppFrame:{repeat}:{image}:Metadata')
        hr_path = r'{}\{}\StructureMask.tiff'.format(self.dv_get('OutputFolder'), image)
        self.add(TextButton, text='View Hi-Res Mask', function='open_image', data=hr_path, overwrite=True,
                 sid=f'HiRes:{repeat}:{image}', container=f'AppFrame:{repeat}:{image}:Metadata')

    def load(self):
        _ = self.dv_get(group='SelectedFiles', branch=None, as_dict=True)
        if not self.is_empty():
            if self.r != len(self.dv_get('RepeatData')) or _ != self.selected_files:
                self.reload()
                self.folder_structure = self.dv()['Directories']
        else:
            super().load()
        self.selected_files = _

    def next_page(self, e):
        if self._current_page + 1 <= self._pages:
            self._change_page(self._current_page + 1, self._current_repeat)

    def previous_page(self, e):
        if self._current_page - 1 >= 0:
            self._change_page(self._current_page - 1, self._current_repeat)

    def change_page(self, page, repeat):
        if page != self._current_page or repeat != self._current_repeat:
            self._change_page(page, repeat)

    def _change_page(self, page, repeat):
        if isinstance(repeat, str):
            repeat = int(repeat.split(':')[-1])

        self[f'AppFrameRepeat:{self._current_repeat}:ImagePage{self._current_page}'].grid_remove()
        try: self[f'TextButtonRepeat:{self._current_repeat}:Page{self._current_page}']['font'] = 'Arial 10';
        except KeyError: pass;  # catch data sets with less than 30 images
        self._current_page = page; self._current_repeat = repeat
        try: self[f'TextButtonRepeat:{self._current_repeat}:Page{self._current_page}']['font'] = 'Arial 10 bold';
        except KeyError: pass;
        self[f'AppFrameRepeat:{self._current_repeat}:ImagePage{self._current_page}'].grid()


class ProcessingOptions(BaseProcessingFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.__files = None
        self._active_repeats = []
        self._activation_bypass = True

    def __traces__(self):
        self.dv_trace('AvailableChannels', 'write', self.__update_channel_selection_menu)
        self.dv_trace('InputFolder', 'write', self.trace_reload, group='Directories', branch='Repeat:0')

    def trace_reload(self):
        self.after(10, super().reload)  # avoid simultaneous triggering of the processing and preprocessing options

    def __base__(self):
        self.add(AppTitle, text='Processing')
        chs = self.dv_get('AvailableChannels') if not self.dv_get('CuratedChannels') else self.dv_get('CuratedChannels')

        # global cell count settings
        self.add(AppSubtitle, text='General Processing Settings')

        self.add(AppLabel, text='Edge exclusion distance (µm):', sid='EdgeExclusionDistance', tooltip=True)
        self.add(SettingEntry, sid='EdgeExclusionDistance', vartype=int, default=50, prow=True, column=0, padx=(220, 0))

        self.add(AppLabel, text='Seeding density (c/cm²):', sid='SeedingDensity')
        self.add(SettingEntry, sid='SeedingDensity', vartype=int, prow=True, column=0, padx=(220, 0))

        self.add(AppLabel, text='Force working resolution:', sid='ForceWorkResolution', tooltip=True)
        self.add(SettingEntry, sid='ForceWorkResolution', default='', vartype=float, prow=True, padx=(220, 0))

        self.add(AppLabel, text='Cell type:', sid='CellType', tooltip=True)
        _ = ('Fibroblast', 'MC3T3', 'Add ...')
        self.add(SelectionMenu, sid='CellType', options=_, default=0, prow=True, column=0, padx=(75, 0))

        # nuclei processing settings
        _ = self.add(AppCheckbutton, text='Nuclei Processing')
        self.add(AppFrame, sid='NucleiProcessingSettings', padx=(22, 0), pady=(5, 20))
        self.add(AppLabel, text='Nuclei channel:', sid='NucleiChannel', tooltip=True, padx=(0, 10),
                 container='AppFrameNucleiProcessingSettings')
        self.add(SelectionMenu, sid='NucleiChannel', options=chs, prow=True, column=1,
                 container='AppFrameNucleiProcessingSettings')
        _.trace_add('write', self.__set_up_np_settings)

        # morphology processing settings
        _ = self.add(AppCheckbutton, text='Morphology Processing')
        self.add(AppFrame, sid='MorphologyProcessingSettings', padx=(22, 0), pady=(5, 20))
        self.add(AppLabel, text='Morphology channel:', sid='MorphologyChannel', tooltip=True, padx=(0, 10),
                 container='AppFrameMorphologyProcessingSettings')
        self.add(SelectionMenu, sid='MorphologyChannel', options=chs, column=1, prow=True,
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

        self.add(AppFrame, sid='RepeatContainer', hide=False)
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
                 command=self.load_previous_processing)

        self.add(AppButton, text='PROCESS', command=self.process)

        self.__update_counting_table_options()
        self._update_repeats()
        self.update_repeat_selection('Repeat:0')

    def _update_repeats(self, *_):
        """Internal method that updates the repeat functionality."""
        repeats = self.dv_get('RepeatData')
        if self._active_repeats != list(repeats.keys()) or self._activation_bypass is True:
            self['AppFrameRepeatContainer'].grid_remove()

            _ = {}; _active_repeats = []
            for k, v in repeats.items():
                _[v['Name']] = partial(self.update_repeat_selection, k)
                if k not in self._active_repeats:
                    _active_repeats.append(k)

            if not self.exists('SelectionMenuRepeatSelection'):
                self.add(AppLabel, text='Repeat: ', sid='RepeatSelection', container='AppFrameRepeatContainer')
                self.add(SelectionMenu, options=[], sid='RepeatSelection', container='AppFrameRepeatContainer',
                         prow=True, padx=(70, 0))

            options = tuple(_)
            self['SelectionMenuRepeatSelection'].update_options(options, commands=_, default=options[0], force=True)

            if len(repeats) > 1:
                self['AppFrameRepeatContainer'].grid()
                self._active_repeats = _active_repeats
            else:  # update default directory entry to match repeat structure
                self._active_repeats = []

        self._activation_bypass = False  # this ensures that the function is always run when the tab is first loaded in

    def update_repeat_selection(self, repeat):
        n = self.selected_repeat.split(':')[-1]
        try: self[f'TextButtonRepeatSelection:{n}'].normal();
        except KeyError: pass;

        super().update_repeat_selection(repeat)
        n = repeat.split(':')[-1]
        try: self[f'TextButtonRepeatSelection:{n}'].bold();
        except KeyError: pass;

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
        repeat_out = self.dv_get('OutputFolder', group='Directories', branch=self.selected_repeat)
        for f in self.dv()['SelectedFiles'][self.selected_repeat]:
            if os.path.isfile(rf'{repeat_out}\{f}\data.json'):
                self[f'TextCheckbutton:{self.selected_repeat}:{f}'].set(False)
            elif self.exists(f'TextCheckbutton:{f}'):
                self[f'TextCheckbutton:{self.selected_repeat}:{f}'].set(True)

    def __update_channel_selection_menu(self, *_):
        self['SelectionMenuNucleiChannel'].update_options(self.dv_get('AvailableChannels'))
        self['SelectionMenuMorphologyChannel'].update_options(self.dv_get('AvailableChannels'))

    def update_counting_settings(self, *_):
        """Internal method that updates setting table."""

        branches = self.dv()['SelectedFiles'].keys()
        _ = []; active_branches = {}
        for b in branches:
            branch_files = self.dv_get(group='SelectedFiles', branch=b, as_dict=True)
            out = self.dv_get('OutputFolder', group='Directories', branch=b)
            settings = supports.json_dict_push(f'{out}\Settings.json', behavior='read')
            active_branches[b] = []; unc = self.dv("RepeatData")[b]["Uncommon"]
            for file, state in branch_files.items():
                if file in settings['PreprocessedParameters']:
                    self.show_table_entry(rf'{b}:{file}')  # ensure that all files exist in the widget space
                    if state is True:
                        _.append(rf'{b}:{file}')
                    else:
                        self.hide_table_entry(rf'{b}:{file}')

                    try:
                        if len(branches) > 1:
                            self[rf'TextCheckbutton:{b}:{file}']['text'] = rf'{file} {unc}'
                        else:
                            self[rf'TextCheckbutton:{b}:{file}']['text'] = rf'{file}'
                    except KeyError:
                        self[rf'TextCheckbutton:{b}:{file}']['text'] = rf'{file}'
                    active_branches[b].append(file)  # add branch as active
        self.__files = _

        for branch, files in self.active_branches.items():
            if branch not in active_branches:
                for k in files:
                    self.remove_table_entry(rf'{branch}:{k}')  # drop entry if its branch has been removed

        self.active_branches = active_branches  # update the active branches

    def load(self):
        if not self.is_empty():
            self.update_counting_settings()
            self._update_repeats()
            self.update_repeat_selection('Repeat:0')
        super().load()

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

    @staticmethod
    def setting_package(file):
        _ = (f'TextCheckbutton:{file}', f'SelectionMenuCountingMethod:{file}',
                     f'SettingEntrySliceSize:{file}', f'SettingEntryCycles:{file}',
                     f'SettingEntryFilter:{file}', f'SettingEntryStep:{file}')
        return _

    def load_previous_processing(self):
        """Method that loads the settings used during latest preprocessing."""
        _out = self.dv_get('OutputFolder', group='Directories', branch=self.selected_repeat)
        settings = supports.json_dict_push(rf'{_out}\Settings.json', behavior='read')
        files = list(settings['IndividualProcessSettings'].keys())
        _ = []
        for file in self.tags['ImageSelection']:  # set activity status
            if file.split(':')[-2] == self.selected_repeat.split(':')[-1]:  # only act on selected repeat
                name = file.split(':')[-1]
                if name in files:
                    self[file].set(True)
                    for k, j in zip(('CountingMethod', 'SliceSize', 'Cycles', 'Filter', 'Step'),
                                    ('SelectionMenu', 'SettingEntry', 'SettingEntry', 'SettingEntry', 'SettingEntry')):
                        self[f'{j}{k}:{self.selected_repeat}:{name}'].set(settings['IndividualProcessSettings'][name][k])
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
        for r in self.dv_get('RepeatData'):
            _ = {}; _out = self.dv_get('OutputFolder', group='Directories', branch=r)
            for file in self.__files:
                if file.startswith(r):
                    file_name = self.split_file_name(file)[1]
                    _[file_name] = {
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
                    'ForceWorkResolution': self['SettingEntryForceWorkResolution'].get(),
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
                    return
                else:
                    update_dict['CollectionProcessSettings']['NucleiChannel'] = self['SelectionMenuNucleiChannel'].get()

            if self['AppCheckbuttonMorphologyProcessing'].get() is False:
                update_dict['CollectionProcessSettings']['MorphologyChannel'] = None
            else:
                if self['SelectionMenuMorphologyChannel'].get() == 'Select Option':
                    messagebox.showerror('Missing Morphology Channel', 'Set a color channel for morphology processing.',
                                         parent=self)
                    return
                else:
                    update_dict['CollectionProcessSettings']['MorphologyChannel'] = self['SelectionMenuMorphologyChannel'].get()

            supports.json_dict_push(rf'{_out}\Settings.json', params=update_dict, behavior='update')

        supports.tprint('Started processing.')
        self.process_daemon()

    @supports.thread_daemon
    def process_daemon(self):
        files = [file for file in self.__files if self[f'TextCheckbutton:{file}'].get() is True]  # get active files

        counter = 1; total = len(files)
        with concurrent.futures.ProcessPoolExecutor(max_workers=supports.get_max_cpu(),
                                                    mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = {executor.submit(base.ProcessingHandler().process, file): file for file in files}
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                file_repeat, file_name = file.split(':')[1:]
                uncommon = self.dv_get("RepeatData")[f"Repeat:{file_repeat}"]["Uncommon"]
                uncommon = f' {uncommon}' if uncommon is not None else ''
                _out = self.dv_get('OutputFolder', group='Directories', branch=f'Repeat:{file_repeat}')
                try:
                    supports.tprint(f'Processed image {file_name}{uncommon} ({counter}/{total}).')
                    supports.json_dict_push(rf'{_out}\Settings.json', params={'ProcessedFiles': {file_name: True}},
                                            behavior='update')  # keep track of processed files
                    counter += 1
                except Exception as e:
                    supports.tprint(f'Failed to process {file_name}{uncommon} with exit: {e!r}.')
                    supports.json_dict_push(rf'{_out}\Settings.json', params={'ProcessedFiles': {file_name: False}},
                                            behavior='update')  # keep track of processed files
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
        self.add(SettingEntry, sid='FieldWidth', prow=True, padx=(200, 0), vartype=float)
        self.add(AppLabel, text='x', sid='FieldDimensionTimes', prow=True, padx=(260, 0))
        self.add(SettingEntry, sid='FieldHeight', prow=True, padx=(280, 0), vartype=float)

        self.add(AppLabel, text='Field spacing (x, y):')
        self.add(SettingEntry, sid='FieldSpacingX', prow=True, padx=(200, 0), vartype=float)
        self.add(AppLabel, text='x', sid='FieldSpacingTimes', prow=True, padx=(260, 0))
        self.add(SettingEntry, sid='FieldSpacingY', prow=True, padx=(280, 0), vartype=float)

        self.add(AppLabel, text='Spacing deviation (x, y):', sid='SpacingDeviation', tooltip=True)
        self.add(SettingEntry, sid='SpacingDeviationX', prow=True, padx=(200, 0), vartype=float, default=0)
        self.add(AppLabel, text='x', sid='SpacingDeviationTimes', prow=True, padx=(260, 0))
        self.add(SettingEntry, sid='SpacingDeviationY', prow=True, padx=(280, 0), vartype=float, default=0)

        self.add(AppLabel, text='Field units:')
        self.add(SelectionMenu, sid='FieldUnits', prow=True, padx=(200, 0), options=('nm', 'µm', 'mm', 'cm'))

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
        self.out = tie.dv_get('OutputFolder', group='Directories', branch='Repeat:0')
        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Reference Setup', pady=(0, 5))
        self.add(AppLabel, text='Reference File:', sid='SampleName', tooltip=True)
        _in = self.tie.dv_get('InputFolder', group='Directories', branch='Repeat:0')
        _ = supports.sort([i.removesuffix('.vsi') for i in os.listdir(_in) if i.endswith('.vsi')])
        _ = self.add(SelectionMenu, options=_, sid='SampleName', prow=True, padx=(150,0))
        _.trace_add('write', self.__load_reference_image_mt)

        self.add(AppLabel, text='Minimum Fields:')
        self.add(SettingEntry, sid='MinFields', prow=True, padx=(150, 0), vartype=int, default=6)
        self.add(AppLabel, text='Mask Shift:')
        self.add(SelectionMenu, options=('Auto', 'None'), sid='MaskShift', prow=True, padx=(150, 0), default=0, width=5)
        self.add(AppLabel, text='Fidelity:')
        self.add(SettingEntry, sid='Fidelity', prow=True, padx=(150, 0), default=3, vartype=int)
        self.add(AppLabel, text='Span:')
        self.add(SettingEntry, sid='Span', prow=True, padx=(150, 0), default=10, vartype=int)
        self.add(AppLabel, text='Force Resolution:')
        self.add(SettingEntry, sid='ForceResolution', prow=True, padx=(150, 0), default='', vartype=float)

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

        scale_bar = supports.json_dict_push(rf'{self.out}\{self["SelectionMenuSampleName"].get()}\metadata.json',
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
                'OnsetIntensitySpan': self['SettingEntrySpan'].get(),
                'ForceResolution': self['SettingEntryForceResolution'].get()
        }}}}

        supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json', params=update_dict, behavior='update')
        supports.json_dict_push(rf'{self.out}\Settings.json', behavior='update',
                                params={'CollectionPreprocessSettings':
                                            {'ForceResolution': self['SettingEntryForceResolution'].get()}})

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
        channel_path = rf'{self.out}\{file}\{channel}.tif'
        dump_path = rf'{self.out}\_misc'

        if not os.path.isdir(dump_path):
            os.makedirs(dump_path)

        if not os.path.isfile(channel_path):
            supports.tprint('Image layers for selected image have not yet been generated. Generating image layers.')
            base.RawImageHandler().handle([rf'Repeat:0:{file}'])

        _ = 1.8 if self['AppCheckbuttonEnhancePreview'].get() is True else 1
        if self['AppCheckbuttonHiResPreview'].get() is True:
            self['ZoomImageFramePreviewImage'].scroll_scalar = (1, 1)
            self['ZoomImageFramePreviewImage'].set_image(path=channel_path, brighten=_, rotate=self.rotation.get())
        else:
            self['ZoomImageFramePreviewImage'].scroll_scalar = (.75, .75)
            lr_path = rf'{dump_path}\{file}_preview.tiff'
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
            load_path = rf'{self.out}\_misc\OrientationReference_preview.tiff'
        else:
            self['ZoomImageFramePreviewMask'].scroll_scalar = (1, 1)
            load_path = rf'{self.out}\_misc\OrientationReference.tiff'

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
    def __init__(self, parent, tie, target_sid, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.tie = tie
        self.target_sid = target_sid
        self.mask_type = None
        self.mask = None
        self.__cf_config = tie['SelectionGridMask'].get_grid_dict(invert=True)
        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='General Options')
        self.add(TextButton, text='Open Existing Presentation Mask', command=self.__open_presentation_mask)
        self.add(AppCheckbutton, text='Enable TeX', sid='UseTex', default=False)
        _ = self.add(AppCheckbutton, text='Convert fields to TeX math', sid='TexMath', default=False)
        self.add(AppCheckbutton, text='Non-cursive math text', sid='NonCursive', default=True, hide=True)
        _.trace_add('write', self.__toggle_upright_text)

        _ = self.add(AppCheckbutton, text='Raw string', sid='RawString', default=False)
        self.add(AppCheckbutton, text='Convert to numbers', sid='Numbers', default=False)
        _.tether(self['AppCheckbuttonTexMath'], True, action=_.tether_action)
        _.tether(self['AppCheckbuttonUseTex'], True, action=_.tether_action)
        self.add(AppLabel, text='Horizontal axis label:', sid='HorizontalAxisLabel')
        self.add(SettingEntry, sid='HorizontalAxisLabel', prow=True, padx=(160, 0), width=25)

        self.add(AppSubtitle, text='Set Presentation Mask', sid='PresentationMask')
        _out = self.tie.dv_get('OutputFolder', group='Directories', branch='Repeat:0')
        _cpps = supports.json_dict_push(rf'{_out}\Settings.json', behavior='read')['CollectionPreprocessSettings']
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

    def __toggle_upright_text(self, *_):
        """Internal method that toggles visibility of the non-cursive math text option."""
        if self['AppCheckbuttonTexMath'].get() is True:
            self['AppCheckbuttonNonCursive'].grid()
        else:
            self['AppCheckbuttonNonCursive'].grid_remove()

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
                    if not prompt: return;
            mask_state = self['EntryGridPresentationMask'].get_grid_state(str_keys=True)
            update_dict = {self.mask_type: {mask_name: {
                'UseTex': self['AppCheckbuttonUseTex'].get(),
                'TexMath': self['AppCheckbuttonTexMath'].get(),
                'NonCursive': self['AppCheckbuttonNonCursive'].get(),
                'RawString': self['AppCheckbuttonRawString'].get(),
                'Numbers': self['AppCheckbuttonNumbers'].get(),
                'HorizontalAxis': self['SettingEntryHorizontalAxisLabel'].get(),
                'Enabled': mask_state
            }}}
            supports.json_dict_push(json_path, params=update_dict, behavior='update')  # update presentation_masks.json
            mask_df = base.craft_dataframe(self['EntryGridPresentationMask'].get_grid())

            # catch non-existent directory save
            if not os.path.isdir(rf'{supports.__cwd__}\__masks__\presentation_masks\{self.tie.process_mask}'):
                os.makedirs(rf'{supports.__cwd__}\__masks__\presentation_masks\{self.tie.process_mask}')

            mask_path = rf'{supports.__cwd__}\__masks__\presentation_masks\{self.tie.process_mask}\{mask_name}.mask'
            mask_df.to_csv(mask_path, sep='\t', index=False, encoding='utf-8', header=False)

            # at last update visible mask options
            if mask_name not in mask_json:  # add option to available options if it does not already exist
                self.tie['SelectionMenuPresentationMask'].add_option(mask_name, order=-1)
                self.tie['SelectionMenuStatisticsPresentationMask'].add_option(mask_name, order=-1)
            self.tie[f'SelectionMenu{self.target_sid}'].set(mask_name)  # set added option as selection

            self.cancel_window(bypass=True)  # destroy mask selector with selection reset bypass

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

    def cancel_window(self, bypass=False):
        """Update cancel window functionality to insert fallback setting."""
        if not bypass:
            fallback = self.tie[f'SelectionMenu{self.target_sid}'].previous
            if fallback == 'Add ...':
                fallback = self.tie[f'SelectionMenu{self.target_sid}'].default
            self.tie[f'SelectionMenu{self.target_sid}'].selection.set(fallback)  # set fallback option as selection
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
        self.add(AppButton, text='Cancel', command=self.cancel_window, pady=(5, 0), padx=(200, 0), prow=True)

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
        def wrapper():
            self.frame.update_suppress.set(True)  # suppress trace triggering while updating from GroupEditEntry
            for tie in self.ties:
                self.frame[f'GroupNameEntry:{tie}'].set(self.get())
            self.frame.update_suppress.set(False)
            self.frame.update_group_list()

            if len(self.frame.table_groups.get()) != len(self.frame.tags['EditGroup']):
                self.frame.load_edit_groups_column()
        self.after(5, wrapper)

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

        elem = self.add(AppCheckbutton, text='Convert data groups to TeX math', sid='DataGroupsToMath', default=False,
                 container='AppFrameBGEContainer', columnspan=2)
        self.add(AppCheckbutton, text='Make math text non-cursive', sid='DataGroupsToUpright', default=True,
                 container='AppFrameBGEContainer', columnspan=2, hide=True)
        elem.trace_add('write', self.__toggle_bge_upright_button)

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
            self.container_drop('AppFrameRenameMGContainer'); self.tags['RenameMultiGroups'] = []
            for mg in self.__mg_handler.multi_groups.keys():
                self.add(AppLabel, text=mg, sid=f'MultiGroup:{mg}', container='AppFrameRenameMGContainer',
                         pady=(2, 5), font='Arial 12 bold')
                _ = self.add(SettingEntry, text=mg, sid=f'MultiGroup:{mg}', container='AppFrameRenameMGContainer',
                         prow=True, column=1, width=15, tag='RenameMultiGroups', padx=(10, 0))
                _.trace_add('write', self.__edit_multi_group_name)
            elem = self.add(AppCheckbutton, text='Convert multi groups to TeX math', sid='MultiGroupsToMath',
                            default=False, container='AppFrameRenameMGContainer', columnspan=2)
            self.add(AppCheckbutton, text='Make math text non-cursive', sid='MultiGroupsToUpright', default=True,
                     container='AppFrameRenameMGContainer', columnspan=2, hide=True)
            elem.trace_add('write', self.__toggle_mg_upright_button)
        else:
            self['AppFrameRenameMGContainer'].grid_remove()
            for k in self.tags['RenameMultiGroups']:
                mg = k.split(':')[-1]
                self[f'AppLabel{mg}']['text'] = mg

    def __toggle_bge_upright_button(self, *_):
        if self['AppCheckbuttonDataGroupsToMath'].get() is True:
            self['AppCheckbuttonDataGroupsToUpright'].grid()
        else:
            self['AppCheckbuttonDataGroupsToUpright'].grid_remove()

    def __toggle_mg_upright_button(self, *_):
        if self['AppCheckbuttonMultiGroupsToMath'].get() is True:
            self['AppCheckbuttonMultiGroupsToUpright'].grid()
        else:
            self['AppCheckbuttonMultiGroupsToUpright'].grid_remove()

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

            _data_groups = {}
            for t in self.tie.tags['GroupEntry']:
                _data_groups[t.split(':')[-1]] = self.tie[t].get()

            update_dict = {mg_name: {
                'Delimiters': self['SettingEntryDelimiters'].get(),
                'UseRegex': self['AppCheckbuttonUseRegex'].get(),
                'RecWhiteSpace': self['AppCheckbuttonRecWhiteSpace'].get(),
                'MultiLetterSeparator': self['SettingEntryMultiLetterSeparator'].get(),
                'GroupIndex': self['SettingEntryGroupIndex'].get(),
                'RenameMultiGroup': self['AppCheckbuttonRenameMGs'].get(),
                'MultiGroupSettings': {
                    'MultiGroupsToMath': self['AppCheckbuttonMultiGroupsToMath'].get(),
                    'MultiGroupsToUpright': self['AppCheckbuttonMultiGroupsToUpright'].get()
                },
                'BulkEditSettings': {
                    'Enabled': self['AppCheckbuttonBulkGroupEdit'].get(),
                    'Delimiter': self['SettingEntryBGEDelimiter'].get(),
                    'Target': self['SettingEntryBGETarget'].get(),
                    'Replacement': self['SettingEntryBGEReplacement'].get(),
                    'DataGroupsToMath': self['AppCheckbuttonDataGroupsToMath'].get(),
                    'DataGroupsToUpright': self['AppCheckbuttonDataGroupsToUpright'].get()
                },
                'MultiGroup': mgs,
                'DataGroups': _data_groups
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

        self.colors = supports.ColorPresets()
        self.dominator = 'Repeat:0'; self.tags['GroupTables'] = set()
        self.links = None
        self.rdata = self.dv_get('RepeatData')
        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Group Data', tooltip=True)
        self.add(AppLabel, text='Color preset:', sid='ColorPreset', tooltip=True)
        _ = self.add(SelectionMenu, options=self.colors.presets, sid='ColorPreset', prow=True, padx=(100, 0),
                 default=8)
        _.trace_add('write', self._trigger_color_update)


        # add group table elements
        self.add(AppFrame, sid='RepeatSelectionContainer', hide=True)
        self.add(AppFrame, sid='GroupTableContainer')

        selected_files = {}
        for repeat, files in self.dv()['SelectedFiles'].items():
            _out = self.dv_get('OutputFolder', group='Directories', branch=repeat)

            try:
                processed_files = supports.json_dict_push(rf'{_out}\Settings.json',
                                                          behavior='read')['ProcessedFiles']
            except KeyError:
                repeat_name = self.rdata[repeat]['Name']
                supports.tprint(f'Files are yet to be processed for {repeat_name}.')
                processed_files = {}

            _ = []
            for file, select_bool in files.items():
                if file in processed_files:
                    if select_bool.get() and processed_files[file]:
                        _.append(file)
            selected_files[repeat] = _
        self.selected_files = selected_files

        unique_names = False
        for repeat, files in selected_files.items():
            # construct group entry for each repeat
            self.dv_define('Group', var=JSONVar(self, value={}), group='DataGroups', branch=repeat)

            if repeat == 'Repeat:0':  # always place the first repeat
                self.add(GroupEditorTable, repeat=repeat, sid=':Repeat:0', container='AppFrameGroupTableContainer',
                         tag='GroupTables')

            if files != selected_files['Repeat:0']:
                self.add(GroupEditorTable, repeat=repeat, sid=f':{repeat}', container='AppFrameGroupTableContainer',
                         hide=True, tag='GroupTables')
                unique_names = True
        self.unique_names = unique_names

        if unique_names:
            """This needs to activate a popup, which ultimately also changes the self.dominator to the selected 
            dominating repeat."""
            self.add(AppLabel, text='Repeat: ', sid='RepeatSelection', container='AppFrameRepeatSelectionContainer')
            _ = {}
            for k, v in self.rdata.items():
                _[v['Name']] = partial(self.toggle_repeat_group, k)
            self.add(SelectionMenu, options=tuple(_), sid='RepeatSelection', prow=True, padx=(100, 0), commands=_,
                     container='AppFrameRepeatSelectionContainer', default=0)
            self['AppFrameRepeatSelectionContainer'].grid()

            self.add(TextButton, text='Link Repeat Groups', command=self._link_repeat_groups)

        self.add(AppLabel, text='Multi-group setup:', sid='MultiGroup', tooltip=True, pady=(5, 5))

        # set multi-group selection menu options
        _out = self.dv_get('OutputFolder', group='Directories', branch=self.dominator)
        _mg_path = rf'{_out}\_misc\multi_group.json'; _ = ['None']
        if os.path.exists(_mg_path):
            _ += list(supports.json_dict_push(_mg_path, behavior='read'))
        _ += ['Add ...']
        _ = self.add(SelectionMenu, options=_, sid='MultiGroup', prow=True, padx=(150, 0), pady=(5, 5), default=0, width=20,
                 commands={'Add ...': self.__add_multi_group_setup})
        _.trace_add('write', self._reset_statistics_group)
        if len(self.rdata) > 1 and unique_names:  # disable multigroup by default if no links have been set, i.e. there are repeats
            self['SelectionMenuMultiGroup'].disable()

        _ = self.add(AppCheckbutton, text='Convert groups to TeX math', default=False, sid='GroupsToMath')
        self['SelectionMenuMultiGroup'].trace_add('write', self.__toggle_tex_to_math)
        self.add(AppCheckbutton, text='Make math text non-cursive', default=True, sid='UprightMathText', hide=True)
        _.trace_add('write', self.__toggle_upright_text)

        self.__toggle_multi_group()

    def _link_repeat_groups(self):
        """Internal method that handles group linking."""

        groups = self.dv_get('Group', group='DataGroups', branch=None, as_dict=True)
        _ = [v.keys() for v in groups.values()]
        links = {}
        if all(i == _[0] for i in _):
            for r, v in groups.items():
                links[r] = {}
                for group in v.keys():
                    links[r][group] = group
            self.links = links
            self.save_repeat_links('Repeat:0')
        else:
            if len({len(i): 0 for i in groups.values()}) != 1:
                messagebox.showerror('Error', "There must be the same amount of groups for each repeat.")
            else:
                level = TopLevelWidget(self); level.title('Repeat Group Linker')
                level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')
                content = RepeatGroupLinker(level.main.interior, tie=self); content.grid()

    def save_repeat_links(self, dominator):
        _ = {}
        for k, v in self.links.items():
            _in = self.dv_get('InputFolder', group='Directories', branch=k)
            _[_in] = v

        for _out in self.dv_get('OutputFolder', group='Directories', branch=None):
            base.directory_checker(rf'{_out}\_misc', clean=False)
            supports.json_dict_push(rf'{_out}\_misc\group_links.json', params=_, behavior='update')

        repeat_selection = self.dv_get('RepeatData')[dominator]['Name']

        try:
            self['SelectionMenuRepeatSelection'].set(repeat_selection)
            self['SelectionMenuRepeatSelection'].disable()
            self.toggle_repeat_group(dominator)
        except KeyError: pass;  # catch situations where unique_names is False

        self['SelectionMenuMultiGroup'].enable()
        self.dominator = dominator

        try:  # change link button status
            self['TextButtonLinkRepeatGroups']['text'] = 'Unlink Repeat Groups'.upper()
            self['TextButtonLinkRepeatGroups']['command'] = self.unlink_repeat_groups
        except KeyError: pass;  # catch non-repeat setups

    def unlink_repeat_groups(self):

        self['SelectionMenuRepeatSelection'].enable(stall=True)
        self['SelectionMenuMultiGroup'].disable()

        # clean json setup
        for _out in self.dv_get('OutputFolder', group='Directories', branch=None):
            supports.json_dict_push(rf'{_out}\_misc\group_links.json', params={}, behavior='replace')

        self.links = None
        self.dominator = 'Repeat:0'

        self['TextButtonLinkRepeatGroups']['text'] = 'Link Repeat Groups'.upper()
        self['TextButtonLinkRepeatGroups']['command'] = self._link_repeat_groups

    def toggle_repeat_group(self, repeat):
        for tag in self.tags['GroupTables']:
            self[tag].grid_remove()
        self[f'GroupEditorTable:{repeat}'].grid()  # toggle correct repeat group visibility

        # update selection settings for the multi group selection menu
        _out = self.dv_get('OutputFolder', group='Directories', branch=repeat)
        _mg_path = rf'{_out}\_misc\multi_group.json'; _ = ['None']
        if os.path.exists(_mg_path):
            _ += list(supports.json_dict_push(_mg_path, behavior='read'))
        _ += ['Add ...']
        self['SelectionMenuMultiGroup'].update_options(_, commands=True)

    def _reset_statistics_group(self, *_):
        self.parent['SelectionEntryPlotSignificance'].suppress_update = True
        self.parent['SelectionEntryPlotSignificance'].set('')
        self.parent['SelectionEntryPlotSignificance'].suppress_update = False

    def __toggle_multi_group(self, *_):
        if self.parent.sample_type in ('Single-Field', 'Zero-Field'):
            self['AppLabelMultiGroup'].grid(); self['SelectionMenuMultiGroup'].grid()
        else:
            self['AppLabelMultiGroup'].grid_remove(); self['SelectionMenuMultiGroup'].grid_remove()

    def __toggle_tex_to_math(self, *_):
        if self['SelectionMenuMultiGroup'].get() == 'None':
            self['AppCheckbuttonGroupsToMath'].grid()
            self['AppCheckbuttonGroupsToMath'].set_previous()
        else:
            self['AppCheckbuttonGroupsToMath'].grid_remove()
            self['AppCheckbuttonGroupsToMath'].set(False)

    def __toggle_upright_text(self, *_):
        if self['AppCheckbuttonGroupsToMath'].get() is True:
            self['AppCheckbuttonUprightMathText'].grid()
        else:
            self['AppCheckbuttonUprightMathText'].grid_remove()

    def _trigger_color_update(self, *_):
        for tag in self.tags['GroupTables']:
            self[tag].trigger_color_update()

    def __add_multi_group_setup(self):
        level = TopLevelWidget(self); level.title('Multi-Group Creator')
        level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')
        content = MultiGroupCreator(level.main.interior, name='MultiGroupCreator', tie=self)
        content.pack(fill='both', expand=True)


class GroupEditorTable(WindowFrame):
    """Wrapper object for the group editor table. This uses links from the DataGroupEditor, and can thus only be used in
    that context."""
    def __init__(self, parent, repeat, *args, **kwargs):
        kwargs['padx'] = 0; kwargs['pady'] = 0
        super().__init__(parent, *args, **kwargs)

        self.repeat = repeat
        self.dirs = {
            'in': self.dv_get('InputFolder', group='Directories', branch=repeat),
            'out': self.dv_get('OutputFolder', group='Directories', branch=repeat)
        }
        self.stall_group_trace = False
        self.table_groups = JSONVar(self, value={})
        self.group_colors = {}
        self.tags['EditGroup'] = set()
        self.update_suppress = tk.BooleanVar(self, False)
        self._parent = parent

        self.load()

    def __base__(self):
        self.files = self.parent.selected_files[self.repeat]

        # construct handlers
        self.add(AppFrame, sid='GroupEditorTable')
        self.add(TextButton, text='Reset', sid='Reset', container='AppFrameGroupEditorTable',
                 command=self.__reset_group_editor, padx=(0, 15))
        self.add(AppLabel, text='Group Name', prow=True, column=1,
                 container='AppFrameGroupEditorTable')
        self.add(TextButton, text='Auto-Group', sid='AutoGroup', prow=True, column=1, padx=(150, 0),
                 container='AppFrameGroupEditorTable', command=self.__auto_group, tooltip=True)
        self.add(AppLabel, text='Edit Groups', prow=True, column=2, tooltip=True,
                 container='AppFrameGroupEditorTable')

        # setup file group rows
        for file in supports.sort(self.files):
            elem = self.add(TextCheckbutton, text=file, container='AppFrameGroupEditorTable', padx=(0, 15),
                            font='Arial 12')
            _ = self.add(GroupNameEntry, sid=f':{file}', prow=True, column=1, tag='GroupEntry',
                         container='AppFrameGroupEditorTable', padx=(0, 15), frame=self)
            _.tether_action['selection'] = ''
            _.trace_add('write', self.group_entry_trace); _.tether(elem, False, action=_.tether_action)
            _.on_tether(self.multi_group_color_mapper, on='both')
        self.load_edit_groups_column()

    def multi_group_color_mapper(self):
        mg_pars = self.parent.parent.mg_pars
        if mg_pars is not None:
            colors = self.parent.colors.get(len(mg_pars), self.parent['SelectionMenuColorPreset'].get())
            for c, (k, v) in zip(colors, mg_pars.items()):
                _c = supports.rgb_to_hex(c)
                fg_c = supports.highlight(_c, -65) if np.mean(c) > .5 else supports.highlight(_c, 130)
                for group in v.keys():
                    if self.exists(f'GroupEditEntry:{group}'):
                        self[f'GroupEditEntry:{group}']['state'] = 'disabled'
                        self[f'GroupEditEntry:{group}']['disabledbackground'] = _c
                        self[f'GroupEditEntry:{group}']['disabledforeground'] = fg_c

    @supports.thread_daemon
    def __reset_group_editor(self):
        self.parent['SelectionMenuMultiGroup'].set('None')
        self.stall_group_trace = True
        for f in self.files: self[f'GroupNameEntry:{f}'].set('');
        self.stall_group_trace = False
        self.group_entry_trace()

    def group_entry_trace(self, *_):
        if not self.stall_group_trace:
            self.__trace_wrapper()

    def __trace_wrapper(self, *_):
        self.update_group_list()
        self.load_edit_groups_column()
        self.update_group_name_colors()
        self.dv_set('Group', self.table_groups.get(), group='DataGroups', branch=self.repeat)

    def update_group_list(self, *_):
        groups = {}  # get unique groups
        for entry in supports.sort(self.tags['GroupEntry']):
            file = entry.split(':')[-1]; group = self[entry].get()
            if group != '':
                if group not in groups:
                    groups[group] = []
                groups[self[entry].get()].extend([file])

        size = len(groups.keys())
        if size > 0:
            colors = self.parent.colors.get(size, self.parent['SelectionMenuColorPreset'].get())
            self.group_colors = dict(zip(groups.keys(), colors))
        self.table_groups.set(groups)

    def update_group_name_colors(self, *_):
        for f in self.files:
            group = self[f'GroupNameEntry:{f}'].get()
            if group != '':  # catch disabled/ungrouped fields
                _c = self.group_colors[group]
                c = supports.rgb_to_hex(_c)
                fg_c = supports.highlight(c, -65) if np.mean(_c) > .5 else supports.highlight(c, 130)
                self[f'GroupNameEntry:{f}']['bg'] = c; self[f'GroupNameEntry:{f}']['fg'] = fg_c

    @supports.thread_daemon
    def __auto_group(self):
        self.stall_group_trace = True
        for f in self.files:
            if self[f'GroupNameEntry:{f}']['state'] in ('normal', tk.NORMAL):  # prevent overwriting disabled fields
                self[f'GroupNameEntry:{f}'].set(' '.join(f.split('_')[:-1]))
        self.stall_group_trace = False
        self.group_entry_trace()

    def load_edit_groups_column(self, *_):
        if not self.update_suppress.get():
            onset_row = self['AppLabelGroupName'].grid_info()['row'] + 1
            groups = self.table_groups.get()

            if groups:
                self['AppLabelEditGroups'].grid()
                _ = self.tags['EditGroup'].copy()  # snapshot tags
                for group in _:
                    self.drop(group)  # drop all existing groups
                self.tags['EditGroup'] = set()  # reset tags

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

    def trigger_color_update(self):
        if self.parent['SelectionMenuMultiGroup'].get() not in ('None', 'Add ...'):
            self.parent.parent.multi_group_trace()
        else:
            groups = self.table_groups.get().keys()
            size = len(groups)
            if size > 0:
                colors = self.parent.colors.get(size, self.parent['SelectionMenuColorPreset'].get())
                self.group_colors = dict(zip(groups, colors))

                self.update_group_name_colors()
                for group in self.tags['EditGroup']:
                    _c = self.group_colors[self[group].get()]
                    c = supports.rgb_to_hex(_c)
                    fg_c = supports.highlight(c, -65) if np.mean(_c) > .5 else supports.highlight(c, 130)
                    self[group]['bg'] = c; self[group]['fg'] = fg_c

    @property
    def parent(self):
        return self._parent.parent

    @parent.setter
    def parent(self, v):
        self._parent = v


class RepeatGroupLinker(PopupWindow):
    def __init__(self, parent, tie, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.tie = tie
        self._max_row = 0
        self.repeat_groups = self.dv_get('Group', group='DataGroups', branch=None, as_dict=True)
        self.links = None
        self.load()

    def __base__(self):
        self.add(AppSubtitle, text='Group Link Setup')
        self.add(AppLabel, text='Select dominant repeat:', sid='DominantRepeat')
        _ = {}; self.selected_dominant = tk.StringVar(self, '')
        for repeat, v in self.dv_get('RepeatData').items():
            _[v['Name']] = partial(self._set_dominant_repeat, repeat)
        self.add(SelectionMenu, options=tuple(_), sid='DominantRepeat', prow=True, padx=(200, 0), commands=_)
        self.add(AppFrame, sid='LinkTable')
        self.add(AppButton, text='Save Links', command=self._save_links, pady=(5, 0))

    def _save_links(self):
        _ = {}; dominant = self.selected_dominant.get()
        for r in self.repeat_groups.keys():
            _[r] = {}
            for g in self.repeat_groups[dominant].keys():
                if r != dominant:
                    _[r][self[rf'SelectionEntry:{g}:{r}'].get()] = g
                else:
                    _[r][self[rf'AppLabel:{g}:{r}']['text']] = g
        self.links = self.tie.links = _
        self.tie.save_repeat_links(dominant)
        self.cancel_window()

    def dv_get(self, *args, **kwargs):
        return self.tie.dv_get(*args, **kwargs)

    def dv_set(self, *args, **kwargs):
        return self.tie.dv_set(*args, **kwargs)

    def _set_dominant_repeat(self, repeat, *_):
        # check for valid repeat selection

        for g in self.repeat_groups.values():
            if not len(g) >= len(self.repeat_groups[repeat]):
                messagebox.showerror('Error', "Dominant repeat cannot have more groups than other repeats.")
                return

        # update repeat if valid
        self.selected_dominant.set(repeat)
        _ = {'DominantRepeat': repeat}
        self.dv_set('RepeatData', _, update=True)

        # update linker table
        self.container_drop('AppFrameLinkTable')  # reset container
        _c = 1
        for r, g in self.repeat_groups.items():
            _r = 1
            uncommon = self.dv_get('RepeatData')[r]['Uncommon']
            if r == repeat:  # set up reference column
                self.add(AppLabel, text=uncommon, sid=f'RepeatLabel:{r}', row=0, container='AppFrameLinkTable',
                         font='Arial 12 bold', column=0)
                for gi in g.keys():
                    self.add(AppLabel, text=gi, sid=f':{gi}:{r}', row=_r, container='AppFrameLinkTable', column=0)
                    _r += 1
            else:
                self.add(AppLabel, text=uncommon, sid=f'RepeatLabel:{r}', row=0, column=_c, padx=(20, 0),
                         container='AppFrameLinkTable', font='Arial 12 bold')
                for _, gi in zip(g.keys(), self.repeat_groups[repeat].keys()):
                    elem = self.add(SelectionEntry, options=tuple(g), sid=f':{gi}:{r}', row=_r, column=_c, padx=(20, 0),
                             container='AppFrameLinkTable')
                    if _ == gi: elem.set(gi);
                    _r += 1
                _c += 1


class AnalysisOptions(WindowFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.mg_previous = 'None'
        self.mg_tether_injection = {}
        self.mg_pars = None
        self.process_mask = None
        self._repeats = ['Repeat:0']; self._dirs = None
        self._selected_files = None

    def __base__(self):
        """Fetch settings for the first repeat. Note that all other repeats must have the same general settings to
        function as a repeat. Consequently, it is only necessary to load in these settings for repeat 0."""
        self._repeats = list(self.dv()['SelectedFiles'].keys())
        self._dirs = self.dv()['Directories']
        out_0 = self.dv_get('OutputFolder', group='Directories', branch='Repeat:0')
        settings_0 = supports.json_dict_push(rf'{out_0}\Settings.json', behavior='read')
        self.sample_type = sample_type = settings_0['CollectionPreprocessSettings']['SampleType']

        self.add(AppTitle, text='Data Analysis')
        self.add(DataGroupEditor, sid='Section')
        self.add(AppSubtitle, text='Analyze Data')

        # NUCLEI ANALYSIS OPTIONS
        elem = self.add(AppCheckbutton, text='Nuclei Analysis', tooltip=True)
        self.add(AppFrame, sid='NucleiAnalysisSettings', padx=(22, 0), pady=(5, 20))

        if sample_type == 'Multi-Field':
            self.process_mask = process_mask = settings_0['CollectionPreprocessSettings']['MaskSelection']
            mask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                                    behavior='read')[sample_type][process_mask]

            # fetch custom presentation masks if they exist
            self.add(AppLabel, text='Presentation mask:', tooltip=True, sid='PresentationMask', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings')
            pmask_data = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                                 behavior='read')
            pmask_options = list(pmask_data[process_mask]) + ['Add ...'] if process_mask in pmask_data else ['Add ...']  # fetch keys
            if 'Native' not in pmask_options:
                pmask_options = ['Native'] + pmask_options  # add native option
            self.add(SelectionMenu, sid='PresentationMask', options=pmask_options, prow=True, column=1, width=17,
                     container='AppFrameNucleiAnalysisSettings',
                     commands={'Add ...': partial(self.__add_mask, 'PresentationMask'), 'Native': self.__add_native_mask})

            self.add(AppLabel, text='Field sorting:', sid='FieldSorting', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings', tooltip=True)
            _ = self.add(SelectionMenu, options=('None', 'Add ...'), sid='FieldSorting', prow=True, column=1, width=17,
                         commands={'Add ...': self.__add_field_sorting}, container='AppFrameNucleiAnalysisSettings')
            _.disable()
            self['SelectionMenuPresentationMask'].trace_add('write', self.__update_nuclei_sorting)

            self.add(AppCheckbutton, text='Mean multiples', sid='MeanIndexMultiples', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings', tooltip=True, default=True)
        elif sample_type in ('Single-Field', 'Zero-Field'):
            self.add(AppLabel, text='External reference group:', sid='ExternalReferenceGroup', padx=(0, 10),
                     container='AppFrameNucleiAnalysisSettings', tooltip=True)
            _ = self.add(SelectionMenu, options=('None',), sid='ExternalReferenceGroup', prow=True, column=1,
                         default=0, container='AppFrameNucleiAnalysisSettings', width=15)
            _.label_bind('<Button-1>', self.update_erg_selection_menu)
            self.add(AppCheckbutton, text='Hide external reference', padx=(0, 10), sid='HideExternalReference',
                     container='AppFrameNucleiAnalysisSettings', default=True)
            _.trace_add('write', self.__set_up_her_checkbutton); self.__set_up_her_checkbutton()

            self['DataGroupEditorSection']['SelectionMenuMultiGroup'].trace_add('write', self.multi_group_trace)

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

        # STATISTICAL ANALYSIS OPTIONS
        elem = self.add(AppCheckbutton, text='Statistical Analysis')
        self.add(AppFrame, sid='StatisticalAnalysisSettings', padx=(22, 0), pady=(5, 20))

        self.add(AppLabel, text='Statistical test:', container='AppFrameStatisticalAnalysisSettings', padx=(0, 10),
                 sid='StatisticalTest')
        self.add(SelectionMenu, sid='StatisticalTest', options=('Tukey\'s HSD', 'T-Test', 'One-Way ANOVA'), column=1,
                 container='AppFrameStatisticalAnalysisSettings', prow=True, width=15)

        self.add(AppLabel, text='Plot significance for:', sid='PlotSignificance', padx=(0, 10), tooltip=True,
                 container='AppFrameStatisticalAnalysisSettings')

        mask_fields = []
        if sample_type == 'Multi-Field':
            mask_fields = ['Control'] + supports.sort(base.load_mask_file(process_mask).to_numpy().flatten())

        _1 = self.add(SelectionEntry, options=mask_fields, sid='PlotSignificance', prow=True, column=1,
                 container='AppFrameStatisticalAnalysisSettings')
        self._mask_fields = mask_fields
        if sample_type in ('Single-Field', 'Zero-Field'):
            _1.bind('<Button-1>', self.__update_statistics_plot_significance_options)

        if sample_type == 'Multi-Field':
            self.add(AppLabel, text='Rename control bar:', sid='RenameControl', padx=(0, 10),
                     container='AppFrameStatisticalAnalysisSettings', group='StatisticsPlotSettings')
            self.add(SettingEntry, sid='RenameControl', prow=True, column=1, width=_['width'],
                     container='AppFrameStatisticalAnalysisSettings', group='StatisticsPlotSettings')

            self.add(AppLabel, text='Presentation mask:', sid='StatisticsPresentationMask', padx=(0, 10),
                     container='AppFrameStatisticalAnalysisSettings', group='StatisticsPlotSettings')
            self.add(SelectionMenu, sid='StatisticsPresentationMask', options=pmask_options, prow=True,
                     column=1, width=17, commands={'Add ...': partial(self.__add_mask, 'StatisticsPresentationMask'),
                                                   'Native': self.__add_native_mask},
                     group='StatisticsPlotSettings', container='AppFrameStatisticalAnalysisSettings')

            self.add(AppLabel, text='Field sorting:', sid='StatisticsFieldSorting', padx=(0, 10),
                     container='AppFrameStatisticalAnalysisSettings', group='StatisticsPlotSettings')
            _ = self.add(SelectionMenu, options=('None', 'Add ...'), sid='StatisticsFieldSorting', prow=True, column=1,
                         width=17, commands={'Add ...': self.__add_field_sorting},
                         container='AppFrameStatisticalAnalysisSettings', group='StatisticsPlotSettings')
            _.disable()
            self['SelectionMenuStatisticsPresentationMask'].trace_add('write', self.__update_statistics_sorting)

        self.add(AppLabel, text='Data label size:', sid='StatisticsLabelSize', padx=(0, 10),
                 container='AppFrameStatisticalAnalysisSettings', group='StatisticsPlotSettings')
        self.add(SettingEntry, sid='StatisticsLabelSize', prow=True, container='AppFrameStatisticalAnalysisSettings',
                 vartype=float, column=1, default=7, group='StatisticsPlotSettings')

        self.add(AppLabel, text='Data label rotation:', sid='StatisticsLabelRotation', padx=(0, 10),
                 container='AppFrameStatisticalAnalysisSettings', group='StatisticsPlotSettings')
        self.add(SettingEntry, sid='StatisticsLabelRotation', container='AppFrameStatisticalAnalysisSettings',
                 default=70, vartype=float, column=1, prow=True, group='StatisticsPlotSettings')

        self.add(AppLabel, text='Display detail:', sid='DisplayDetail', padx=(0, 10), tooltip=True,
                 container='AppFrameStatisticalAnalysisSettings', group='StatisticsPlotSettings')
        self.add(SelectionMenu, options=('None', 'Data Points', 'Std'), sid='DisplayDetail', prow=True, column=1,
                 width=17, group='StatisticsPlotSettings', container='AppFrameStatisticalAnalysisSettings', default=0)

        self.add(AppCheckbutton, text='Save statistics as Excel', sid='ExcelStatistics', columnspan=2, tooltip=True,
                 container='AppFrameStatisticalAnalysisSettings', default=False)

        _1.trace_add('write', self.__update_statistics_options); self.__update_statistics_options()
        elem.trace_add('write', self.__set_up_sa_settings)

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
        self.add(SelectionEntry, width=20, sid='FigureFont', prow=True, padx=(120, 0), column=0,
                 columnspan=3, default='Arial', options=tkfont.families(), display_number=5)
        self.add(AppLabel, text='Figure scalar: ', tooltip=True, sid='FigureScalar')
        self.add(SettingEntry, width=6, sid='FigureScalar', prow=True, padx=(120, 0), column=0,
                 columnspan=3, default=1.2, vartype=float)

        self.add(AppLabel, text='Figure dpi:')
        self.add(SettingEntry, width=6, sid='FigureDpi', prow=True, padx=(120, 0), column=0, default=600, vartype=int)
        self.add(AppCheckbutton, text='Generate Excel overview', sid='ExcelExport', tooltip=True, default=False)

        self.add(AppButton, text='Analyze', command=self.__analyze)

        # change defaults before display
        if sample_type in ('Single-Field', 'Zero-Field'):
            self['SettingEntryLabelRotation'].set(0); self['SettingEntryStatisticsLabelRotation'].set(0)
            self['SettingEntryLabelSize'].set(10); self['SettingEntryStatisticsLabelSize'].set(10)

        self.check_morphology_processing()  # check whether the morphology processing has been performed on widget load

    def load(self):
        _ = self.dv_get(group='SelectedFiles', branch=None, as_dict=True)
        if self._dirs is None:  # only load first time, otherwise reload
            super().load()
        elif self._dirs != self.dv()['Directories'] or self._selected_files != _:
            self._selected_files = _
            self.reload()

    def check_morphology_processing(self):
        for r in self._repeats:
            _out = self.dv_get('OutputFolder', group='Directories', branch=r)
            settings = supports.json_dict_push(rf'{_out}\Settings.json', behavior='read')
            try:
                if settings['CollectionProcessSettings']['MorphologyChannel'] is not None:
                    self['AppCheckbuttonMorphologyAnalysis'].set(True)  # change morph eval default if processed for it
                else:
                    self['AppCheckbuttonMorphologyAnalysis'].grid_remove()
            except KeyError:  # catch missing processing
                self['AppCheckbuttonMorphologyAnalysis'].grid_remove()

    def update_erg_selection_menu(self, e):
        options = list(self.dv_get('Group', group='DataGroups', branch=self['DataGroupEditorSection'].dominator))
        if self['DataGroupEditorSection'].exists('SelectionMenuMultiGroup'):
            if self['DataGroupEditorSection']['SelectionMenuMultiGroup'].get() != 'None':
                _out = self.dv_get('OutputFolder', group='Directories', branch=self['DataGroupEditorSection'].dominator)
                options = supports.json_dict_push(rf'{_out}\_misc\multi_group.json',
                    behavior='read')[self['DataGroupEditorSection']['SelectionMenuMultiGroup'].get()]['MultiGroup']
                options = list(list(options.values())[0].values())
        self['SelectionMenuExternalReferenceGroup'].update_options(['None'] + options, default='None')
        self['SelectionMenuExternalReferenceGroup'].toggle(None)

    def __update_statistics_options(self, *_):
        try:
            if self['SelectionEntryPlotSignificance'].get() in self._mask_fields:
                for g in self.groups['StatisticsPlotSettings']:
                    self[g].grid()
            else:
                for g in self.groups['StatisticsPlotSettings']:
                    self[g].grid_remove()
        except KeyError: pass

    def __update_statistics_plot_significance_options(self, e):
        options = list(self.dv_get('Group', group='DataGroups', branch=self['DataGroupEditorSection'].dominator))
        if self['DataGroupEditorSection'].exists('SelectionMenuMultiGroup'):
            if self['DataGroupEditorSection']['SelectionMenuMultiGroup'].get() != 'None':
                _out = self.dv_get('OutputFolder', group='Directories', branch=self['DataGroupEditorSection'].dominator)
                options = supports.json_dict_push(rf'{_out}\_misc\multi_group.json',
                    behavior='read')[self['DataGroupEditorSection']['SelectionMenuMultiGroup'].get()]['MultiGroup']

                options = list(list(options.values())[0].values())

        self['SelectionEntryPlotSignificance'].replace_options(options)
        self._mask_fields = options

    def multi_group_trace(self, *_):
        self['SelectionMenuExternalReferenceGroup'].set('None')
        current = self['DataGroupEditorSection']['SelectionMenuMultiGroup'].get()
        current = current if current is not None else 'None'
        """This needs to be rewritten to support multi-grouping of repeats, after the links functionality has been 
        established."""
        table_id = f'GroupEditorTable:{self["DataGroupEditorSection"].dominator}'
        if current not in ('None', 'Add ...', 'Select Option'):
            _out = self.dv_get('OutputFolder', group='Directories', branch=self['DataGroupEditorSection'].dominator)
            _ = supports.json_dict_push(rf'{_out}\_misc\multi_group.json', behavior='read')
            _ = _[current]
            self['DataGroupEditorSection'][table_id].stall_group_trace = True
            colors = self['DataGroupEditorSection'].colors.get(
                len(_['MultiGroup']), self['DataGroupEditorSection']['SelectionMenuColorPreset'].get())
            for k, v in _['DataGroups'].items():
                # save configuration data for downstream tether injection

                try:  # catch disabled files
                    entry = self['DataGroupEditorSection'][table_id][f'GroupNameEntry:{k}'].get()
                except KeyError: continue;

                if self.mg_previous in ('None', 'Select Option'):
                    if self['DataGroupEditorSection'][table_id][f'TextCheckbutton{k.split(":")[-1]}'.replace(
                            ' ', '')].get() is True:
                        self.mg_tether_injection[k] = {
                            'selection': entry,
                            'state': self['DataGroupEditorSection'][table_id][f'GroupNameEntry:{k}']['state']
                        }
                    else:
                        self.mg_tether_injection[k] = {
                            'selection': self['DataGroupEditorSection'][table_id][f'GroupNameEntry:{k}'].get_tether_cache('selection'),
                            'state': self['DataGroupEditorSection'][table_id][f'GroupNameEntry:{k}'].get_tether_cache('state')
                        }

                # update current data group configuration to the multi group configuration
                if entry != v and self['DataGroupEditorSection'][table_id][f'TextCheckbutton{k}'].get() is True:
                        self['DataGroupEditorSection'][table_id][f'GroupNameEntry:{k}'].set(v)

            self['DataGroupEditorSection'][table_id].stall_group_trace = False
            self['DataGroupEditorSection'][table_id].group_entry_trace()

            color_map = {}
            for c, (k, v) in zip(colors, _['MultiGroup'].items()):
                _c = supports.rgb_to_hex(c)
                fg_c = supports.highlight(_c, -65) if np.mean(c) > .5 else supports.highlight(_c, 130)
                for group in v.keys():
                    color_map[group] = (_c, fg_c)
                    self['DataGroupEditorSection'][table_id][f'GroupEditEntry:{group}']['state'] = 'disabled'
                    self['DataGroupEditorSection'][table_id][f'GroupEditEntry:{group}']['disabledbackground'] = _c
                    self['DataGroupEditorSection'][table_id][f'GroupEditEntry:{group}']['disabledforeground'] = fg_c
            self.mg_pars = _['MultiGroup']


            for k, v in _['DataGroups'].items():
                try:  # catch disabled files
                    if self['DataGroupEditorSection'][table_id][f'TextCheckbutton{k}'].get() is True:
                        self['DataGroupEditorSection'][table_id][f'GroupNameEntry:{k}']['state'] = 'disabled'
                        self['DataGroupEditorSection'][table_id][f'GroupNameEntry:{k}']['disabledbackground'] = color_map[v][0]
                        self['DataGroupEditorSection'][table_id][f'GroupNameEntry:{k}']['disabledforeground'] = color_map[v][1]
                    else:
                        _ = {
                            'disabledbackground': color_map[v][0],
                            'disabledforeground': color_map[v][1],
                            'state': 'disabled',
                            'selection': v
                        }
                        self['DataGroupEditorSection'][table_id][f'GroupNameEntry:{k}'].manipulate_tether(_)
                except KeyError:
                    pass

            self['DataGroupEditorSection'][table_id]['TextButtonAutoGroup'].grid_remove()  # avoid complications with auto group

        elif current == 'None':
            self['DataGroupEditorSection'][table_id].stall_group_trace = True
            for k in self['DataGroupEditorSection'][table_id].tags['GroupEntry']:
                _k = k.split(":")[-1]
                _ti = self.mg_tether_injection[_k]
                if self['DataGroupEditorSection'][table_id][f'TextCheckbutton{_k}'].get() is True:
                    self['DataGroupEditorSection'][table_id][k]['state'] = 'normal'
                    self['DataGroupEditorSection'][table_id][k].set(_ti['selection'])
                else:  # restore tether cache
                    self['DataGroupEditorSection'][table_id][k].manipulate_tether('selection', _ti['selection'])
                    self['DataGroupEditorSection'][table_id][k].manipulate_tether('state', _ti['state'])
            self['DataGroupEditorSection'][table_id].stall_group_trace = False
            self['DataGroupEditorSection'][table_id].group_entry_trace()
            self.mg_pars = None
            self['DataGroupEditorSection'][table_id]['TextButtonAutoGroup'].grid()
        self.mg_previous = current

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

    def __set_up_sa_settings(self, *_):
        if self['AppCheckbuttonStatisticalAnalysis'].get() is True:
            self['AppFrameStatisticalAnalysisSettings'].grid()
        else:
            self['AppFrameStatisticalAnalysisSettings'].grid_remove()

    def __set_up_her_checkbutton(self, *_):
        if self['SelectionMenuExternalReferenceGroup'].get() != 'None':
            self['AppCheckbuttonHideExternalReference'].grid()
        else:
            self['AppCheckbuttonHideExternalReference'].grid_remove()

    def __update_statistics_sorting(self, *_):
        return self.__update_sorting_options('Statistics')

    def __update_nuclei_sorting(self, *_):
        return self.__update_sorting_options('')

    def __update_sorting_options(self, section):
        """Internal method that updates sorting options based on the selected presentation mask."""
        pmask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                                 behavior='read')[self.process_mask]
        if self[f'SelectionMenu{section}PresentationMask'].get() not in ('Add ...', 'Select Option'):
            if pmask_settings[self[f'SelectionMenu{section}PresentationMask'].get()]['Numbers'] is True:
                self[f'SelectionMenu{section}FieldSorting'].disable(value='None')
            else:
                self[f'SelectionMenu{section}FieldSorting'].enable()

                pmask = self[f'SelectionMenu{section}PresentationMask'].get()
                field_sortings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\field_sortings.json',
                                                         behavior='read')

                options = ('None', 'Add ...')
                if self.process_mask in field_sortings:
                    if pmask in field_sortings[self.process_mask]:
                        options = ['None'] + list(field_sortings[self.process_mask][pmask]) + ['Add ...']

                self[f'SelectionMenu{section}FieldSorting'].update_options(
                    options, commands={'Add ...': self.__add_field_sorting})

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

        if add_native_mask:
            _ = self['SelectionGridMask'].get_grid_dict(invert=True, str_keys=True)  # fetch control field setup

            update_dict = {self.process_mask: {'Native': {
                'UseTex': False,
                'TexMath': False,
                'RawString': False,
                'Enabled': _,
                'Numbers': False
            }}}

            # push native mask to presentation mask dict
            supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json', behavior='update',
                                    params=update_dict)

    def __add_mask(self, target_sid):
        level = TopLevelWidget(self); level.title('Presentation Mask Creator')
        level.geometry(f'{int(self.winfo_screenwidth() * .5)}x{int(self.winfo_screenheight() * .5)}')
        content = PresentationMaskCreator(level.main.interior, name='PresentationMaskCreator', tie=self,
                                          target_sid=target_sid)
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

        for r in self._repeats:
            r_data = self.dv_get('RepeatData')
            r_name = r_data[r]['Name'] + ' ' + r_data[r]['Uncommon']
            out_dir = self.dv_get('OutputFolder', group='Directories', branch=r)

            settings = supports.json_dict_push(rf'{out_dir}\Settings.json', behavior='read')
            _mask = settings['CollectionPreprocessSettings']['MaskSelection']
            if _mask in _existing:
                prompt = messagebox.askokcancel('Warning', f'A control field selection already exists for {r_name} '
                                                           f'mask {_mask!r}. Proceeding will overwrite the existing '
                                                           f'saved selection. Do you want to continue?')
                if not prompt: return;
            _existing[_mask] = self['SelectionGridMask'].get_grid()  # define new mask
            supports.json_dict_push(rf'{supports.__cache__}\mask_control_fields.json', _existing, behavior='replace')

    def __analyze(self):
        """Internal method that extract the cell counting settings."""

        if self['SelectionEntryFigureFont'].get() not in tkfont.families():
            messagebox.showerror('Invalid Font', "Please select a valid font to proceed.",
                                 parent=self.parent)
            return

        dominator = self['DataGroupEditorSection'].dominator

        if not self['DataGroupEditorSection'].unique_names:
            _ = {}
            groups = self.dv_get('Group', group='DataGroups', branch='Repeat:0')
            for r in self._repeats:
                _[r] = dict(zip(groups, groups))
            self['DataGroupEditorSection'].links = _
            self['DataGroupEditorSection'].save_repeat_links('Repeat:0')
        else:  # if links are needed, throw an error if they do not exist
            if len(self._repeats) > 1 and self['DataGroupEditorSection'].links is None:
                messagebox.showerror('Missing Group Links', "Please link data groups to proceed.",
                                     parent=self.parent)
                return

        for r in self._repeats:
            settings = {'CollectionAnalysisSettings': {
                'ColorPreset': self['DataGroupEditorSection']['SelectionMenuColorPreset'].get(),
                'DataGroups': list(self.dv_get('Group', group='DataGroups', branch=dominator)),
                'AnalyzeData': {
                    'NucleiAnalysis': self['AppCheckbuttonNucleiAnalysis'].get(),
                    'NearestNeighbourHistogram': self['AppCheckbuttonNearestNeighborEvaluation'].get(),
                    'StatisticalAnalysis': self['AppCheckbuttonStatisticalAnalysis'].get(),
                },
                'ExcelExport': self['AppCheckbuttonExcelExport'].get(),
                'FigureFont': self['SelectionEntryFigureFont'].get(),
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
                'PlotSignificance': self['SelectionEntryPlotSignificance'].get(),
                'StatisticalTest': self['SelectionMenuStatisticalTest'].get(),
                'StatisticsLabelRotation': self['SettingEntryStatisticsLabelRotation'].get(),
                'StatisticsLabelSize': self['SettingEntryStatisticsLabelSize'].get(),
                'ExcelStatistics': self['AppCheckbuttonExcelStatistics'].get(),
                'DisplayDetail': self['SelectionMenuDisplayDetail'].get(),
                'GroupsToMath': self['DataGroupEditorSection']['AppCheckbuttonGroupsToMath'].get(),
                'UprightMathText': self['DataGroupEditorSection']['AppCheckbuttonUprightMathText'].get(),
            }, 'IndividualAnalysisSettings': {}}

            # add sample-type specific settings
            if self.sample_type == 'Multi-Field':
                if self['AppCheckbuttonNucleiAnalysis'].get() is True:
                    if self['SelectionMenuFieldSorting'].get() == 'Select Option':
                        messagebox.showerror('Missing Field Sorting', "Please select a Field Sorting from the "
                                                                      "drop-down menu to continue.", parent=self.parent)
                        return

                if self['SelectionEntryPlotSignificance'].get() != '':
                    if self['SelectionMenuStatisticsPresentationMask'].get() == 'Select Option':
                        messagebox.showerror('Missing Statistics Presentation Mask',
                                             "Please select a Presentation Mask from the drop-down menu to continue.",
                                             parent=self.parent)
                        return
                    if self['SelectionMenuStatisticsFieldSorting'].get() == 'Select Option':
                        messagebox.showerror('Missing Statistics Field Sorting',
                                             "Please select a Field Sorting from the drop-down menu to continue.",
                                             parent=self.parent)
                        return

                _ = {
                    'ControlFields': self['SelectionGridMask'].get(),
                    'FieldSorting': self['SelectionMenuFieldSorting'].get(),
                    'PresentationMask': self['SelectionMenuPresentationMask'].get(),
                    'MeanIndexMultiples': self['AppCheckbuttonMeanIndexMultiples'].get(),
                    'RenameControl': self['SettingEntryRenameControl'].get(),
                    'StatisticsPresentationMask': self['SelectionMenuStatisticsPresentationMask'].get(),
                    'StatisticsFieldSorting': self['SelectionMenuStatisticsFieldSorting'].get()
                }
            elif self.sample_type in ('Single-Field', 'Zero-Field'):
                _ = {
                    'ExternalReferenceGroup': self['SelectionMenuExternalReferenceGroup'].get(),
                    'HideExternalReference': self['AppCheckbuttonHideExternalReference'].get(),
                    'BarWidth': self['SettingEntryBarWidth'].get(),
                    'MultiGroup': self['DataGroupEditorSection']['SelectionMenuMultiGroup'].get(),
                    'SeedingScoreCriterion': self['SettingEntrySeedingScoreCriterion'].get(),
                }
            settings['CollectionAnalysisSettings']['SampleTypeSettings'] = _

            table_repeat = r if self['DataGroupEditorSection'].unique_names else 'Repeat:0'
            for f in self['DataGroupEditorSection'].selected_files[table_repeat]:
                dg = self['DataGroupEditorSection'][f'GroupEditorTable:{table_repeat}'][f'GroupNameEntry:{f}'].get()
                settings['IndividualAnalysisSettings'][f] = {
                    'DataGroup': self['DataGroupEditorSection'].links[r][dg],
                    'State': self['DataGroupEditorSection'][f'GroupEditorTable:{table_repeat}'][f'TextCheckbutton{f}'.replace(' ', '')].get()
                }

            _out = self.dv_get('OutputFolder', 'Directories', r)
            supports.json_dict_push(rf'{_out}\Settings.json', settings, behavior='mutate')

        _ = {'DominantRepeat': self['DataGroupEditorSection'].dominator,
             'RepeatData': self.dv_get('RepeatData')}
        supports.json_dict_push(rf'{supports.__cache__}\settings.json', _, behavior='update')
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
        TopLevelProperties.__init__(self, dep_var={'': {'': {  # construct global variable dict
            'LastClick': tk.StringVar(self, ''),
            'AvailableChannels': JSONVar(self, value=[]),
            '_AvailableChannels': JSONVar(self, value=[]),
            'CuratedChannels': JSONVar(self, value=[]),
            'LatestPreprocessedFile': tk.StringVar(self, ''),
            'ActiveFrame': tk.StringVar(self, ''),
            'Debugger': tk.BooleanVar(self, False),
        }}})

        apset = supports.json_dict_push(rf'{supports.__cache__}\application.json', behavior='read')
        if 'ApplicationSettings' in apset:
            if 'Debugger' in apset['ApplicationSettings']:
                self.dependent_variables['']['']['Debugger'].set(apset['ApplicationSettings']['Debugger'])

        self.dependent_variables['']['']['TooltipTimer'] = tk.IntVar(self, self.defaults['TooltipTimer'])

        self.iconbitmap(default=rf'{supports.__gpx__}\icon.ico')

        self.current_frame = ''
        self.dependent_variables['']['']['ActiveFrame'].trace_add('write', self.__on_content_change)

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
        self.main = ContentFrame(self, background=supports.__cp__['bg'])  # construct content frame
        self.main.pack(side='left', fill='both', expand=True)  # place content frame

        self.content_container = {}  # container for application windows and associated buttons
        for frame in (FileSelection, PreprocessingOptions, MaskGallery, ProcessingOptions, AnalysisOptions,
                      ResultOverview, ApplicationSettings):
            _ = frame(self.main.interior)
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

        self.dependent_variables['']['']['ActiveFrame'].set('FileSelection')  # set current frame to the start frame

    def _update_content_frame(self, e):

        self.main.canvas.config(height=self.winfo_height())
        self.main.refresh_content_frame()

    def __on_content_change(self, *_):
        self.main.pack_forget()
        new_frame = self.dependent_variables['']['']['ActiveFrame'].get()
        if self.current_frame != new_frame:  # update frame if it is not the same frame that has been clicked
            self.main.set_position_memory(self.current_frame)
            supports.tprint('Selected Frame: {}'.format(new_frame))

            for v in self.content_container.values():
                v.grid_remove()  # forget loaded content
                v.unbind_all()

            _ = self.content_container[new_frame]  # grab new content
            _.load()  # place new content
            _.grid()  # place WindowFrame

            # pre-configure interior width before updating scrollbar
            self.main.interior.configure(width=_.winfo_width(), height=_.winfo_height())
            self.main.get_position_memory(new_frame,(0, 0, _.winfo_width(), _.winfo_height()))

            self.current_frame = new_frame  # update current frame
            _.rebind_all()  # rebind all associated all-bindings
        self.main.pack(side='left', fill='both', expand=True)  # place content frame

if __name__ == '__main__':
    cellexum = CellexumApplication()
    cellexum.mainloop()
