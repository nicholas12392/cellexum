import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mli
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import base
import os
import scipy as sp
import operator
from collections.abc import Sequence
import supports
from supports import lprint as print


class DataAnalysis:
    """Class that handles data analysis for the cellexum application
    :param data: normalized data from the __pretreat_data module in base"""

    def __init__(self, data, repeat):

        # set default values for all parameters
        _ = supports.setting_cache()
        self.dominator = _['DominantRepeat']
        self._dirs = _['DirectorySettings']
        self.rdata = _['RepeatData']
        self.repeat = repeat
        repeat = repeat if repeat is not None else self.dominator
        self.dirs = self._dirs[repeat]
        self.__settings = supports.json_dict_push(rf'{self.dirs["OutputFolder"]}\Settings.json', behavior='read')  # load settings

        if self.repeat is None:  # define parameters for collection analysis
            self.__write_path = rf'{self._dirs["CommonOutputFolder"]}\_figures'
            self.mean_dict = {'Collection Data': data['MeanData']}
            if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
                self.mean_data = {k: base.craft_collection_dataframe(v) for k, v in data['MeanData'].items()}
                self.data_sets = {g: {f: base.craft_collection_dataframe(d) for f, d in f_dict.items()} for g, f_dict in
                                  data['SetData'].items()}
            else:
                self.mean_data = {'Collection Data': base.craft_collection_dataframe(data['MeanData'])}
                self.data_sets = {g: {f: pd.Series(d, name=f) for f, d in f_dict.items()} for g, f_dict in
                                  data['SetData'].items()}

            self.data = {'Collection Data': data['CollectionData']}
            self.groups = data['DataGroups']

            self.uc = ' (Collection)'
        else:  # define parameters for individual analysis
            self.__write_path = rf'{self.dirs["OutputFolder"]}\_figures'
            self.mean_dict = None; self.mean_data = None; self.data_sets = None
            self.groups = None
            self.data = data
            self.uc = f' {self.rdata[repeat]["Uncommon"]}' if len(self.rdata) > 1 else ''

        self.pmask_settings = None
        self.colors = supports.ColorPresets(self.__settings['CollectionAnalysisSettings']['ColorPreset'])

        self.icc = None
        self.additional_data = None
        self.repeat_groups = supports.json_dict_push(rf'{self.dirs["OutputFolder"]}\_misc\group_links.json', behavior='read')

        # check write folder existence
        if not os.path.exists(self.__write_path):
            os.mkdir(self.__write_path)

        # store necessary parameters
        _ = 'None'
        if self['CollectionPreprocessSettings']['SampleType'] in ('Single-Field', 'Zero-Field'):
            _ = self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup']

        if _ == 'None':
            multi_group = {'MultiGroup': None}; mg_groups = 0
        else:
            mg_folder = self._dirs[self.dominator]['OutputFolder']
            multi_group = supports.json_dict_push(rf'{mg_folder}\_misc\multi_group.json', behavior='read')[_]
            mg_groups = {j: n for v in multi_group['MultiGroup'].values() for n, j in enumerate(v.values())}

        ylabel_dict = {
            'Normalized Cell Count': 'Relative Cell Count',
            'Cell Count': 'Mean Cell Count',
            'Mean Cell Distribution (µm)': 'Nearest Neighbour Distance [µm]',
            'Cell Density': r'Cell Density [cells cm$^{-2}$]',
            'Normalized Cell Coverage': 'Relative Cell Coverage',
            'Cell Coverage': 'Cell Coverage Fraction',
            'Average Cell Area (µm^2/c)': r'Average Cell Area [µm$^2$ cell$^{-1}$]',
            'SC Cell Count': 'Compensated Cell Count',
            'SC Cell Density': 'Compensated Cell Density [cells cm$^{-2}$]',
            'SC Cell Coverage': 'Compensated Cell Coverage',
            'SC Normalized Cell Coverage': 'Compensated Cell Coverage',
            'SC Normalized Cell Count': 'Compensated Cell Count',
            'Normalized Cell Area': 'Relative Cell Area',
        }

        self._pars = {
            'MG': multi_group['MultiGroup'],
            'MG_GROUPS': mg_groups,
            'FS': self['CollectionAnalysisSettings']['FigureScalar'],
            'Y_LABELS': ylabel_dict,
            'MG_SETTINGS': multi_group
        }

        self._unit_conversion = {'nm': 1e-3, 'µm': 1, 'mm': 1e3, 'cm': 1e4}

    def __getitem__(self, item):
        return self.__settings[item]

    def group_data(self):
        """Method that determines data means based on the preset grouping in the application window."""
        data_groups = self['CollectionAnalysisSettings']['DataGroups']

        self.data_sets = {}
        if data_groups:
            group_dict = {}  # construct appendable dict
            for group in data_groups:
                group_dict[group] = []
                self.data_sets[group] = {}

            for file, member in self['IndividualAnalysisSettings'].items():  # catch group members
                group = member['DataGroup']
                if group != '' and member['State'] is True:
                    group_dict[group].append(file)  # add file names to appendable dict
            self.groups = group_dict  # save groups for later use

            # find the group means and construct a new DataFrame based on the groups
            if self['CollectionPreprocessSettings']['SampleType'] in ('Single-Field', 'Zero-Field'):
                group_means = []
                _stds = {'Cell Density': {}, 'Average Cell Area (µm^2/c)': {}, 'Cell Coverage': {},
                         'SC Cell Density': {}, 'SC Cell Coverage': {}}
                for group, files in group_dict.items():
                    group_means.append(self.data['Collection Data'].loc[files].mean(axis=0).to_frame(group).T)
                    for t in ('Cell Density', 'Average Cell Area (µm^2/c)', 'Cell Coverage', 'SC Cell Density',
                              'SC Cell Coverage'):
                        try:  # catch missing morphology data
                            _stds[t][group] = self.data['Collection Data'][t].loc[files].std(ddof=1)  # changed second t from Cell Density (note changes)
                            self.data_sets[group][t] = self.data['Collection Data'][t].loc[files]
                        except KeyError:
                            pass
                self.mean_data = self.data.copy(); self.mean_data['Collection Data'] = pd.concat(group_means, axis=0)
                self.mean_dict = self.mean_data['Collection Data'].to_dict()
                for k, v in _stds.items():
                    self.mean_dict[f'{k} Std'] = v

                supports.json_dict_push(rf'{self["DirectorySettings"]["OutputFolder"]}\AnalysisResults.json',
                                        {'GroupAnalysis': self.mean_dict}, behavior='update')

            elif self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
                self.mean_data = self.data
                self.additional_data = {'ControlCount': {}, 'ControlArea': {}, 'ControlDensity': {}}
                data_dict = {}
                for sheet in ('Cell Count', 'Normalized Cell Count', 'Area Cell Count', 'Mean Cell Distribution (µm)',
                              'Std Cell Distribution (µm)', 'Average Cell Area (µm^2/c)', 'Cell Coverage',
                              'Normalized Cell Coverage', 'Normalized Cell Area', 'Seeding Score',
                              'SC Normalized Cell Count', 'SC Normalized Cell Coverage'):
                    sheet_means = []
                    for group, files in group_dict.items():
                        try:  # catch missing morphology data
                            group_data = self.data[sheet][files]
                            sheet_means.append(group_data.mean(axis=1).to_frame(group))
                            self.data_sets[group][sheet] = group_data
                        except KeyError:
                            pass

                        # save control counts as additional data
                        _control_counts = []; _control_area = []; _control_density = []
                        for file in files:
                            _path = r'{}\{}\data.json'.format(self.__settings['DirectorySettings']['OutputFolder'], file)
                            _data = supports.json_dict_push(_path, behavior='read')['AdditionalData']
                            _control_counts.append(_data['InternalControlCountMean'])
                            _scale = self['PreprocessedParameters'][file]['FieldParameters']['ScaleBar']
                            _w = (self['PreprocessedParameters'][file]['FieldParameters']['Width'] / _scale -
                                  self['CollectionProcessSettings']['EdgeProximity'] * 2)  # deduct proximity area
                            _h = (self['PreprocessedParameters'][file]['FieldParameters']['Height'] / _scale -
                                  self['CollectionProcessSettings']['EdgeProximity'] * 2)
                            _area = _w * _h
                            _control_area.append(_area)
                            _control_density.append(_data['InternalControlCountMean'] / _area * 1e8)  # /µm^2 -> /cm^2

                        for k, v in zip(('ControlCount', 'ControlArea', 'ControlDensity'),
                                        (np.mean(_control_counts), np.mean(_control_area), np.mean(_control_density))):
                            self.additional_data[k][group] = v

                    try:  # catch missing sheets from analysis options
                        self.mean_data[sheet] = pd.concat(sheet_means, axis=1)
                        if sheet != 'Std Cell Distribution (µm)':
                            data_dict[sheet] = self.mean_data[sheet].to_dict()
                    except ValueError:
                        pass
                self.mean_dict = data_dict

                supports.json_dict_push(rf'{self["DirectorySettings"]["OutputFolder"]}\AnalysisResults.json',
                                        {'GroupAnalysis': data_dict, 'ControlParameters': self.additional_data},
                                        behavior='update')
        else:
            self.mean_data = self.data

    def multi_field_group_sort(self):
        """Sorts and orders the input array data based on the selected presentation mask and the field sorting."""

        # load the selected presentation mask and accompanying settings
        mask_name = self['CollectionPreprocessSettings']['MaskSelection']
        mask = base.strip_dataframe(base.load_mask_file(mask_name))
        pmask_name = self['CollectionAnalysisSettings']['SampleTypeSettings']['PresentationMask']
        self.pmask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                             behavior='read')[mask_name][pmask_name]
        if pmask_name == 'Native':
            pmask = mask
        else:
            pmask = base.strip_dataframe(base.load_mask_file(rf'presentation_masks\{mask_name}\{pmask_name}'))

        pmask_field_ids = [base.str_to_tuple(k, int) for k, v in  # fetch active presentation mask field indices
                             self.pmask_settings['Enabled'].items() if v is True]
        pmask_fields = [pmask[k] for k in pmask_field_ids]  # fetch only active fields
        pmask_dict = dict(zip(pmask_field_ids, pmask_fields))  # construct field code-index dict

        # load sorting settings
        field_sorting = self['CollectionAnalysisSettings']['SampleTypeSettings']['FieldSorting']
        if field_sorting != 'None':
            sort_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\field_sortings.json',
                                                    behavior='read')[mask_name][pmask_name][field_sorting]
            fc_handler = base.FieldCodeHandler(pmask_fields)
            fc_handler.split_codes(delimiters=sort_settings['Delimiters'], regex=sort_settings['UseRegex'],
                                   white_space=sort_settings['RecWhiteSpace'],
                                   separator=sort_settings['MultiLetterSeparator'])
            fc_handler.sort_codes(indices=sort_settings['SortingIndices'], reverse_order=sort_settings['RevOrder'],
                                  field_codes=pmask_field_ids)
            field_codes = fc_handler.id_codes
        else:
            field_codes = pmask_field_ids

        # construct sorted dataframe
        temp = self.mean_data
        for sheet in ('Cell Count', 'Normalized Cell Count', 'Area Cell Count', 'Mean Cell Distribution (µm)',
                      'Std Cell Distribution (µm)', 'Cell Density', 'Cell Coverage', 'Normalized Cell Coverage',
                      'Average Cell Area (µm^2/c)', 'Normalized Cell Area', 'Seeding Score',
                      'SC Normalized Cell Count', 'SC Normalized Cell Coverage'):
            _ = []
            try:
                data_sheet = self.mean_data[sheet]
            except KeyError:
                continue
            for field_code in field_codes:
                elem = data_sheet.loc[mask[field_code]]  # fetch data for field code
                elem.rename(pmask_dict[field_code], inplace=True)  # rename data row to presentation mask field code
                _.append(elem)
            temp[sheet] = pd.concat(_, axis=1).T
        self.mean_data = temp

    def determine_icc(self):

        if self['CollectionAnalysisSettings']['MarkIdealCellCount'] is True:
            _mask = self['CollectionPreprocessSettings']['MaskSelection']
            _type = self['CollectionPreprocessSettings']['SampleType']
            _mask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                                     behavior='read')[_type][_mask]

            # rescale to µm if defined units are different
            _mask_settings['FieldWidth'] *= self._unit_conversion[_mask_settings['FieldUnits']]
            _mask_settings['FieldHeight'] *= self._unit_conversion[_mask_settings['FieldUnits']]
            _area = ((_mask_settings['FieldWidth'] - self['CollectionProcessSettings']['EdgeProximity']) *
                     (_mask_settings['FieldHeight'] - self['CollectionProcessSettings']['EdgeProximity']))

            # calculate the ideal cell count from the seeding density (1e-8 is conversion from µm^2 to cm^2)
            _icc = self['CollectionProcessSettings']['SeedingDensity'] * _area * 1e-8
            if _type == 'Multi-Field':
                _control_mean = np.mean(list(self.additional_data['ControlCount'].values()))
                self.icc = _icc / _control_mean
            else:
                self.icc = _icc

    def statistical_analysis(self):
        """Internal method that determines statistics for the data group with regard to normalized cell count."""

        if self.repeat is None:
            out = self._dirs["CommonOutputFolder"]
            settings = supports.json_dict_push(rf'{self.dirs["OutputFolder"]}\Settings.json', behavior='read')
        else: out = self.dirs['OutputFolder'];
        out = rf'{out}\_statistics'

        _ = StatisticalAnalysis(self, self.data_sets, self.__settings, self._pars, out)
        _.analyze()


    def multi_field_nuclei_analysis(self):

        self.multi_field_group_sort()  # sort data and load masks

        plt.rcParams.update({
            'text.usetex': self.pmask_settings['UseTex'],
            'font.family': self['CollectionAnalysisSettings']['FigureFont']
        })

        sheets = ['Normalized Cell Count', 'Mean Cell Distribution (µm)', 'Seeding Score']
        if self['CollectionAnalysisSettings']['MorphologyAnalysis'] is True:
            sheets += ['Normalized Cell Coverage', 'Average Cell Area (µm^2/c)', 'Normalized Cell Area']
        if self['CollectionAnalysisSettings']['SeedingCompensation'] is True:
            sheets += ['SC Normalized Cell Count']
            if self['CollectionAnalysisSettings']['MorphologyAnalysis'] is True:
                sheets += ['SC Normalized Cell Coverage']

        self.determine_icc()  # calculate ideal cell count if prompted

        for sheet in sheets:
            fig = plt.figure(figsize=(8 / self._pars['FS'], 3.5 / self._pars['FS']),  # initiate figure generation
                             dpi=self['CollectionAnalysisSettings']['FigureDpi'])

            try:  # catch missing morphology analysis
                data = self.mean_data[sheet]
            except KeyError:
                continue

            if data.shape[0] != len(dict(zip(data.index, data.index))) or self.pmask_settings['Numbers'] is True:
                _lw = 0
            else:  # remove lines between data points if there are index multiples
                _lw = 1
            if self['CollectionAnalysisSettings']['SampleTypeSettings']['MeanIndexMultiples'] is True:
                data = data.groupby(by=data.index, sort=False).mean()
                _lw = 1

            # change x-ticks according to the presentation mask settings
            x_ticks = data.index
            if self.pmask_settings['TexMath'] is True:
                if self.pmask_settings['NonCursive'] is True:
                    x_ticks = [r'$\mathrm{%s}$' % e for e in data.index]
                else:
                    x_ticks = [rf'${e}$' for e in data.index]
            elif self.pmask_settings['RawString'] is True:
                x_ticks = [rf'{e}' for e in data.index]

            if self.pmask_settings['Numbers'] is True:
                x_ticks = [float(i) for i in data.index]

            if 'HorizontalAxis' in self.pmask_settings:
                if self.pmask_settings['HorizontalAxis'] != '':
                    plt.xlabel(self.pmask_settings['HorizontalAxis'])


            colors = self.colors.get(data.shape[1])
            for ys, c in zip(data.items(), colors):  # construct cell count graphs
                if self.pmask_settings['Numbers'] is False:
                    for x, y in zip(x_ticks, ys[1]):  # draw lines to each data point
                        plt.plot((x, x), (-2, y), color=c, lw=1, alpha=0.5, zorder=0)

                # construct dummy plot to use as label constructor
                plot_data = ys[1].to_numpy()
                plt.plot(x_ticks[0], plot_data[0], label=self.group_editor(ys[0]), color=c, ms=4, marker='o',
                         lw=0, zorder=-1)
                plt.plot(x_ticks, plot_data, color=c, ms=4, marker='o', lw=_lw, zorder=1)

            if sheet in self._pars['Y_LABELS']:
                plt.ylabel(self._pars['Y_LABELS'][sheet])
            else:
                plt.ylabel(sheet)

            plt.xticks(rotation=self['CollectionAnalysisSettings']['LabelRotation'],
                       fontsize=self['CollectionAnalysisSettings']['LabelSize'])
            if self.pmask_settings['Numbers'] is True:
                x_span = max(x_ticks) - min(x_ticks); x_extension = x_span * .01
                plt.xlim(min(x_ticks) - x_extension, max(x_ticks) + x_extension)
            else:
                plt.xlim(- len(x_ticks) * .01, len(x_ticks) * 1.01)

            if self.icc is not None:
                if sheet in ('Normalized Cell Count', 'SC Normalized Cell Count'):
                    plt.axhline(self.icc, color='k', ls='--', lw=1, label='Seeded', zorder=0)

            # get y-range
            y_lims = (data.to_numpy().min(), data.to_numpy().max())
            y_range = y_lims[1] - y_lims[0]
            y_lim_max = y_lims[1] + y_range * .2  # add 20% of the y-range to the y_lim_max as default
            y_lim_min = y_lims[0] - y_range * .1  # reduce 10% of the y-range to the y_lim_min
            if self['CollectionAnalysisSettings']['GraphStyle'] == 'Crisp':
                plt.legend(frameon=True, ncols=3, handletextpad=0.1, facecolor='#ebebeb', edgecolor='#ebebeb',
                           fancybox=False, loc=1)
                y_lim_max = y_lims[1] + y_range * .3  # add 30% of the y-range to the y_lim_max
            elif self['CollectionAnalysisSettings']['GraphStyle'] == 'Crisp (No L-Frame)':
                plt.legend(frameon=False, ncols=3, handletextpad=0.1)

            # fix layout before adding patches
            plt.ylim(y_lim_min, y_lim_max)
            plt.tight_layout()
            plt.gca().spines['right'].set_visible(False)  # remove right figure border
            plt.gca().spines['top'].set_visible(False)  # remove top figure border

            mask_name = self['CollectionPreprocessSettings']['MaskSelection']
            pmask_name = self['CollectionAnalysisSettings']['SampleTypeSettings']['PresentationMask']
            field_sorting = self['CollectionAnalysisSettings']['SampleTypeSettings']['FieldSorting']
            na_path = rf'{self.__write_path}\nuclei_analysis'
            base.directory_checker(rf'{na_path}\svg', clean=False)
            plt.savefig(rf'{na_path}\svg\{sheet}_{mask_name}-{pmask_name}-{field_sorting}.svg'.replace(' (µm^2/c)', ''))
            plt.savefig(rf'{na_path}\{sheet}_{mask_name}-{pmask_name}-{field_sorting}.png'.replace(' (µm^2/c)', ''))
            plt.close('all')  # kill all figures on exit

    def single_field_nuclei_analysis(self):
        bar_width = self['CollectionAnalysisSettings']['SampleTypeSettings']['BarWidth']
        data = self.mean_data['Collection Data']

        plt.rcParams.update({
            'font.family': self['CollectionAnalysisSettings']['FigureFont']
        })

        _erg = self['CollectionAnalysisSettings']['SampleTypeSettings']['ExternalReferenceGroup']
        cell_sheet = 'Normalized Cell Count' if _erg != 'None' else 'Cell Count'
        coverage_sheet = 'Normalized Cell Coverage' if _erg != 'None' else 'Cell Coverage'

        hide_group = []
        if self['CollectionAnalysisSettings']['SampleTypeSettings']['HideExternalReference'] is True and _erg != 'None':
            if self._pars['MG'] is not None:
                _ = []
                for elem in self._pars['MG'].values():
                    for k, v in elem.items():
                        if v == _erg:
                            _.append(k)
                hide_group = _
            else:
                hide_group = [_erg]

        sheets = [cell_sheet, 'Mean Cell Distribution (µm)', 'Cell Density', 'Seeding Score']
        if self['CollectionAnalysisSettings']['MorphologyAnalysis'] is True:
            sheets += [coverage_sheet, 'Average Cell Area (µm^2/c)']
        if self['CollectionAnalysisSettings']['SeedingCompensation'] is True:
            sc_cell_sheet = 'SC Normalized Cell Count' if _erg != 'None' else 'SC Cell Count'
            sc_coverage_sheet = 'SC Normalized Cell Coverage' if _erg != 'None' else 'SC Cell Coverage'
            sheets += [sc_cell_sheet, sc_coverage_sheet, 'SC Cell Density']

        self.determine_icc()

        for sheet in sheets:
            fig = plt.figure(figsize=(8 / self._pars['FS'], 3.5 / self._pars['FS']),  # initiate figure generation
                             dpi=self['CollectionAnalysisSettings']['FigureDpi'])

            colors = self.colors.get(data.shape[0], hex=True)

            try:  # catch missing morphology processing
                _data = data[sheet].items()
            except KeyError:
                continue

            if self._pars['MG'] is None:
                for (tick, value), c in zip(_data, colors):
                    plot_tick = self.group_editor(tick)
                    if sheet in ('Cell Count', 'Normalized Cell Count', 'SC Cell Count'):
                        if tick not in hide_group:
                            if self.repeat is None:  # adjust for collection analysis
                                plt.bar(plot_tick, value, color=c, yerr=self.mean_dict[f'{sheet} Std'][tick],
                                        ecolor=supports.highlight(c, -20), capsize=4, width=bar_width)
                            else:
                                plt.bar(plot_tick, value, color=c, width=bar_width)
                    elif sheet == 'Mean Cell Distribution (µm)':
                        if self.repeat is None:  # adjust for collection analysis
                            plt.bar(plot_tick, value, color=c, yerr=self.mean_dict[f'{sheet} Std'][tick],
                                    ecolor=supports.highlight(c, -20), capsize=4, width=bar_width)
                        else:
                            plt.bar(plot_tick, value, color=c, yerr=data['Std Cell Distribution (µm)'][tick],
                                    ecolor=supports.highlight(c, -20), capsize=4, width=bar_width)
                    elif sheet in ('Cell Density', 'SC Cell Density'):
                        plt.bar(plot_tick, value, color=c, yerr=self.mean_dict[f'{sheet} Std'][tick],
                                ecolor=supports.highlight(c, -20), capsize=4, width=bar_width)
                    elif sheet in ('Cell Coverage', 'Average Cell Area (µm^2/c)', 'Normalized Cell Coverage',
                                   'Seeding Score', 'SC Cell Coverage', 'SC Normalized Cell Coverage'):
                        if tick not in hide_group:
                            _ = []
                            for n in self.groups[tick]:
                                value = self.data['Collection Data'][sheet][n]
                                plt.plot(plot_tick, value, color=c, ms=4, marker='o', lw=0, alpha=.7)
                                _.append(value)

                            if self.repeat is None:
                                plt.errorbar(plot_tick, np.mean(_), color=c, marker='_', ms=6, lw=0, capsize=4,
                                         yerr=self.mean_dict[f'{sheet} Std'][tick], ecolor=supports.highlight(c, -20),
                                             elinewidth=1)
                            else:
                                plt.plot(plot_tick, np.mean(_), color=c, marker='_', ms=6, lw=0)
                            _lh = (mli.Line2D([], [], color='black', label='Individuals', linewidth=0, markersize=4, marker='o'),
                            mli.Line2D([], [], color='black', label='Mean', linewidth=0, markersize=4, marker='_'))
                            plt.legend(frameon=False, handletextpad=0.1, handles=_lh)
            else:
                if sheet in ('Cell Density', 'SC Cell Density', 'Mean Cell Distribution (µm)', 'SC Normalized Cell Count',
                                                     'Cell Count', 'Normalized Cell Count', 'SC Cell Count'):
                    offset = 0; label_conversion = {}
                    for (mg, g), c in zip(self._pars['MG'].items(), self.colors.get(len(self._pars['MG']), hex=True)):
                        # collect single-group data from multi-group
                        values = []; errors = []; tick_values = []; indexer = 0
                        for k, v in g.items():
                            try:  # catch zero-member multi-groups
                                if k in hide_group and sheet in ('Normalized Cell Count', 'SC Normalized Cell Count'):
                                    pass
                                else:
                                    values.append(data[sheet][k])
                            except KeyError:
                                supports.tprint(f'No data for {k}{self.uc}. Skipping group.', branch=2)
                                continue

                            if k in hide_group and sheet in ('Normalized Cell Count', 'SC Normalized Cell Count'):
                                indexer = 1
                            else:
                                tick_values.append(self._pars["MG_GROUPS"][v] - indexer)  # ensure that all values are correctly assigned
                                # respect bulk edit settings
                                label_conversion[v] = self.bulk_editor(v)

                            if self.repeat is not None:
                                if sheet in ('Cell Density', 'SC Cell Density'):
                                    errors.append(self.mean_dict[f'{sheet} Std'][k])
                                elif sheet == 'Mean Cell Distribution (µm)':
                                    errors.append(data['Std Cell Distribution (µm)'][k])
                            elif k in hide_group and sheet in ('Normalized Cell Count', 'SC Normalized Cell Count'):
                                pass
                            else:
                                errors.append(self.mean_dict['Collection Data'][f'{sheet} Std'][k])

                        tick_values = np.array(tick_values) + offset
                        if self.repeat is None:
                            plt.bar(tick_values, values, color=c, label=self._mg_editor(mg), yerr=errors,
                                    width=bar_width / 2, ecolor=supports.highlight(c, -20), capsize=4)
                        else:
                            if sheet in ('Cell Count', 'Normalized Cell Count', 'SC Cell Count', 'SC Normalized Cell Count'):
                                plt.bar(tick_values, values, color=c, label=self._mg_editor(mg), width=bar_width / 2)
                            else:
                                plt.bar(tick_values, values, color=c, label=self._mg_editor(mg), yerr=errors,
                                        width=bar_width / 2, ecolor=supports.highlight(c, -20), capsize=4)
                        offset += bar_width / 2

                    if hide_group and sheet in ('Normalized Cell Count', 'SC Normalized Cell Count'):
                        _ = []; indexer = 0
                        for k, v in self._pars['MG_GROUPS'].items():
                            if k != _erg:
                                _.append(v - indexer)
                            else:
                                indexer = 1
                        label_ticks = [i + np.median(np.arange(len(self._pars['MG']))) * bar_width / 2 for i in _]
                    else:
                        label_ticks = [i + np.median(np.arange(len(self._pars['MG']))) * bar_width / 2 for i in
                                       self._pars['MG_GROUPS'].values()]

                    plt.gca().set_xticks(label_ticks, list(label_conversion.values()))

                elif sheet in ('Cell Coverage', 'Average Cell Area (µm^2/c)', 'Normalized Cell Coverage',
                               'Seeding Score', 'SC Cell Coverage', 'SC Normalized Cell Coverage'):
                    label_conversion = {}; offset = 0
                    _lh = [mli.Line2D([], [], color='black', label='Individuals', linewidth=0, markersize=4, marker='o'),
                           mli.Line2D([], [], color='black', label='Mean', linewidth=0, markersize=4, marker='_')]
                    for (mg, g), c in zip(self._pars['MG'].items(), self.colors.get(len(self._pars['MG']), hex=True)):
                        # collect single-group data from multi-group
                        indexer = 0
                        for tick, (k, v) in enumerate(g.items()):
                            _ = []
                            try:  # catch zero-member multi-groups
                                if k in hide_group and sheet in ('Normalized Cell Coverage', 'SC Normalized Cell Coverage'):
                                    indexer = 1
                                else:
                                    for n in self.groups[k]:
                                        value = self.data['Collection Data'][sheet][n]
                                        plt.plot(tick + offset - indexer, value, color=c, ms=4, marker='o', lw=0, alpha=.7)
                                        _.append(value)
                            except KeyError:
                                supports.tprint(f'No data for {k}{self.uc}. Skipping group.', branch=2)
                                continue

                            if k in hide_group and sheet in ('Normalized Cell Coverage', 'SC Normalized Cell Coverage'):
                                indexer = 1
                            else:
                                label_conversion[v] = self.bulk_editor(v)
                                if self.repeat is None:
                                    plt.errorbar(tick + offset - indexer, np.mean(_), color=c, marker='_', ms=6, lw=0,
                                             label=self._mg_editor(mg), yerr=np.std(_, ddof=1), elinewidth=1,
                                             ecolor=supports.highlight(c, -20), capsize=4)
                                else:
                                    plt.plot(tick + offset - indexer, np.mean(_), color=c, marker='_', ms=6, lw=0,
                                             label=self._mg_editor(mg))
                        _lh.append(mli.Line2D([], [], color=c, label=self._mg_editor(mg), linewidth=0,
                                              markersize=4, marker='s'))
                        offset += .15
                    plt.legend(frameon=False, handletextpad=0.1, handles=_lh)

                    if hide_group and sheet in ('Normalized Cell Coverage', 'SC Normalized Cell Coverage'):
                        _ = []; indexer = 0
                        for k, v in self._pars['MG_GROUPS'].items():
                            if k != _erg:
                                _.append(v - indexer)
                            else:
                                indexer = 1
                        label_ticks = [i + np.median(np.arange(len(self._pars['MG']))) * .15 for i in _]
                    else:
                        label_ticks = [i + np.median(np.arange(len(self._pars['MG']))) * .15 for i in
                                       self._pars['MG_GROUPS'].values()]

                    plt.gca().set_xticks(label_ticks, list(label_conversion.values()))

            if sheet in self._pars['Y_LABELS']:
                plt.ylabel(self._pars['Y_LABELS'][sheet])
            else:
                plt.ylabel(sheet)

            if sheet in ('Cell Density', 'SC Cell Density'):
                if self['CollectionProcessSettings']['SeedingDensity'] != '':
                    plt.axhline(self['CollectionProcessSettings']['SeedingDensity'], color='k', ls='--', lw=1,
                                label='Seeding Density', zorder=0)
                    plt.legend(frameon=False)

            if self.icc is not None:
                if sheet in ('Cell Count', 'SC Cell Count'):
                    plt.axhline(self.icc, color='k', ls='--', lw=1, label='Seeded', zorder=0)
                    plt.legend(frameon=False)

            plt.xticks(rotation=self['CollectionAnalysisSettings']['LabelRotation'],
                       fontsize=self['CollectionAnalysisSettings']['LabelSize'])
            if sheet in ('Cell Count', 'Normalized Cell Count', 'SC Cell Count', 'Cell Density', 'SC Cell Density',
                         'Mean Cell Distribution (µm)', 'SC Normalized Cell Count'):
                plt.ylim(bottom=0)  # fix the lower limit to be 0, i.e. prevent stds from clipping the plot
                if self._pars['MG'] is not None:
                    plt.legend(frameon=False)
            plt.tight_layout()
            plt.gca().spines['right'].set_visible(False)  # remove right figure border
            plt.gca().spines['top'].set_visible(False)  # remove top figure border

            if self['CollectionPreprocessSettings']['SampleType'] == 'Single-Field':
                mask_name = self['CollectionPreprocessSettings']['MaskSelection']
            else:
                mask_name = 'ZeroField'

            na_path = rf'{self.__write_path}\nuclei_analysis'
            base.directory_checker(rf'{na_path}\svg', clean=False)
            if sheet in ('Cell Count', 'Normalized Cell Count', 'Cell Coverage', 'Normalized Cell Coverage',
                         'SC Cell Count', 'SC Normalized Cell Count', 'SC Cell Coverage', 'SC Normalized Cell Coverage'):
                plt.savefig(rf'{na_path}\svg\{sheet}_{mask_name}-{_erg}.svg')
                plt.savefig(rf'{na_path}\{sheet}_{mask_name}-{_erg}.png')
            else:
                plt.savefig(rf'{na_path}\svg\{sheet}_{mask_name}.svg'.replace(' (µm^2/c)', ''))
                plt.savefig(rf'{na_path}\{sheet}_{mask_name}.png'.replace(' (µm^2/c)', ''))
            plt.close('all')  # kill all figures on exit

    def bulk_editor(self, name):
        """Internal method that handles bulk group edit settings."""
        if self._pars['MG_SETTINGS']['BulkEditSettings']['DataGroupsToMath'] is True:
            if self._pars['MG_SETTINGS']['BulkEditSettings']['DataGroupsToUpright'] is True:
                name = r'$\mathrm{%s}$' % name
            else:
                name = r'${%s}$' % name
        return name

    def _mg_editor(self, name):
        """Internal method that handles bulk editing of multi-group names defined in the settings."""
        if self._pars['MG_SETTINGS']['MultiGroupSettings']['MultiGroupsToMath'] is True:
            if self._pars['MG_SETTINGS']['MultiGroupSettings']['MultiGroupsToUpright'] is True:
                name = r'$\mathrm{%s}$' % name
            else:
                name = r'${%s}$' % name
        return name

    def group_editor(self, name):
        """Internal method that handles bulk editing of group names."""
        if self['CollectionAnalysisSettings']['GroupsToMath'] is True:
            if self['CollectionAnalysisSettings']['UprightMathText'] is True:
                name = r'$\mathrm{%s}$' % name
            else:
                name = r'$%s$' % name
        return name

    def _analyze_model(self, model, np_data, criterion):

        np_data = [i for i in np_data if i < criterion]
        bins = list(dict(zip(np_data, np_data)).keys())

        if self['CollectionAnalysisSettings']['ZeroLock'] is True:  # fit data to selected distribution model
            pars = model.fit(np_data, floc=0)
        else:
            pars = model.fit(np_data)

        frozen = model(*pars)  # freeze found distribution
        span = np.linspace(min(bins), max(bins), 1000); fit = frozen.pdf(span)
        mode = (span[np.where(fit == max(fit))[0][0]], max(fit))  # estimate distribution mode
        fwhm_spline = sp.interpolate.make_splrep(span, fit - np.max(fit) / 2, s=0)
        fwhm_roots = sp.interpolate.sproot(fwhm_spline); fwhm_density = sp.integrate.quad(frozen.pdf, *fwhm_roots)
        return frozen, span, mode, fwhm_roots, fwhm_density, fit, pars

    def nearest_neighbour_analysis(self):

        # setup parameters
        out_path = rf'{self.__write_path}\nearest_neighbour_distribution_analysis'
        base.directory_checker(rf'{out_path}\_overview')

        # define distribution model
        dist = self['CollectionAnalysisSettings']['DistributionModel']
        if dist == 'Normal':
            model = sp.stats.norm
        elif dist == 'Log Normal':
            model = sp.stats.lognorm
        elif dist == 'Skewed Normal':
            model = sp.stats.skewnorm
        else:
            raise ValueError(f'Unknown distribution model {dist!r}.')

        # set up data
        data = self.data['NN Distances (µm)']
        data.sort_index(axis=1, inplace=True)

        # analyze NND data and make individual statistical plots
        colors = self.colors.get(data.shape[1], hex=True)
        results = {}
        for (n, d), c in zip(data.items(), colors):
            base.directory_checker(rf'{out_path}\{n}')  # check out directory existence
            plt.figure(figsize=(8 / self._pars['FS'], 3.5 / self._pars['FS']),
                       dpi=self['CollectionAnalysisSettings']['FigureDpi'])
            np_data = d.dropna().to_numpy()  # drop missing values and convert to numpy array
            bins = sorted(list(dict(zip(np_data, np_data)).keys())); size = len(bins)  # get discrete entries
            plt.hist(np_data, bins=size, color=c, density=True, rwidth=.8)  # bin and plot the data

            # generate model data points and features
            _max_bin_reduction = 0; _failed = False
            while True:
                try:
                    _criterion = bins[:int(size * (1 - _max_bin_reduction))][-1]
                    frozen, span, mode, fwhm_roots, fwhm_density, fit, pars = self._analyze_model(model, np_data,
                                                                                                  _criterion)
                    break
                except TypeError:
                    _max_bin_reduction += .01
                    if _max_bin_reduction >= round(self['CollectionAnalysisSettings']['ForceFit'] / 100, 2):
                        _failed = True; supports.tprint(f'Distribution analysis for {n}{self.uc} failed.', branch=2); break
                    supports.tprint('Distribution analysis for {}{} failed. Retrying without {}% of the largest bins.'.format(
                        self.uc, n, int(_max_bin_reduction * 100)), branch=2)
                except ValueError:
                    _failed = True; supports.tprint(f'Distribution analysis for {n}{self.uc} failed.', branch=2); break

            if _failed is True:  # skip the rest of the iteration if the fit failed
                continue

            results[n] = {
                'Model': dist,
                'BinCount': size,
                'Observations': len(np_data),
                'Mode': mode[0],
                'DataMean': np.mean(np_data),
                'DataStd': np.std(np_data, ddof=1),
                'DataMedian': np.median(np_data),
                'ModelParameters': pars,
                'Min': min(bins),
                'Max': max(bins),
                'ModelMean': frozen.mean(),
                'ModelStd': frozen.std(),
                'ModelMedian': frozen.median(),
                'SpanFWHM': np.diff(fwhm_roots)[0],
                'DensityFWHM': fwhm_density[0],
                'DensityErrorFWHM': fwhm_density[1],
                'ForceFit': _max_bin_reduction
            }

            plt.axvspan(*fwhm_roots, facecolor='black', alpha=.05, zorder=0, label=r'$FWHM={:.1f}$, $\rho={:.2f}$'.format(
                results[n]['SpanFWHM'], results[n]['DensityFWHM']))
            plt.plot(span, fit, color='black', ms=0, lw=1)  # plot distribution model

            # plot misc data
            plt.plot(results[n]['ModelMean'], frozen.pdf(results[n]['ModelMean']), color='black', ms=4,
                     marker='o', lw=0, label=r'$\bar x={:.1f}$, $\sigma={:.1f}$'.format(results[n]['ModelMean'],
                                                                         results[n]['ModelStd']))
            plt.plot(*mode, color='gray', ms=3, marker='o', lw=0,
                     label=r'$Mode={:.1f}$'.format(results[n]['Mode']))
            plt.plot(results[n]['ModelMedian'], frozen.pdf(results[n]['ModelMedian']), color='lightgray', ms=2,
                     marker='o', lw=0, label=r'$\tilde x={:.1f}$'.format(results[n]['ModelMedian']))

            plt.ylabel('Probability Density'); plt.xlabel('Nearest Neighbour Bins [µm]')
            plt.legend(frameon=False, handletextpad=0.1)
            plt.tight_layout(); plt.savefig(rf'{out_path}\{n}\probability-{dist}.png')
            plt.savefig(rf'{out_path}\_overview\{n}-{dist}.png', dpi=100)

            # determine qqplot
            plt.figure(figsize=(8 / self._pars['FS'], 3.5 / self._pars['FS']),
                       dpi=self.__settings['CollectionAnalysisSettings']['FigureDpi'])
            model_quantiles = frozen.ppf(np.arange(1.0, len(np_data) + 1) / (len(np_data) + 1))
            real_quantiles = np.sort(np.array(np_data))
            plt.plot(model_quantiles, real_quantiles, lw=0, marker='o', ms=4, color=c)

            # fit linear regression to quantile relation
            series, parameters = np.polynomial.Polynomial([2, 1]).fit(model_quantiles, real_quantiles, deg=1, full=True)
            coef = series.convert().coef
            cod = 1 - parameters[0][0] / np.sum(np.square(real_quantiles - np.mean(real_quantiles)))
            results[n]['QQPlot'] = {
                'DeterminationCoefficient': cod,
                'Intercept': coef[0],
                'Slope': coef[1]
            }

            plt.plot(model_quantiles, self.linreg(model_quantiles, *coef), lw=1, color='black',
                     label=rf'$R^2$={cod:.4f}')

            plt.xlim(min(model_quantiles) - max(model_quantiles) * .01, max(model_quantiles) * 1.01)
            plt.ylabel('Real Quantiles'); plt.xlabel('Model Quantiles'); plt.legend(frameon=False, handletextpad=0.1)
            plt.tight_layout(); plt.savefig(rf'{out_path}\{n}\qqplot-{dist}.png')

            # determine cdf plot
            plt.figure(figsize=(8 / self._pars['FS'], 3.5 / self._pars['FS']),
                       dpi=self.__settings['CollectionAnalysisSettings']['FigureDpi'])
            plt.hist(np_data, bins=size, color=c, density=True, rwidth=.8, cumulative=True)  # bin and plot the data
            plt.plot(span, frozen.cdf(span), color='black', ms=0, lw=1)  # plot distribution model

            results[n]['CDFQuantiles'] = {}
            for p, _c, ms in zip((.50, .75, .90, .95, .99), (0, .2, .4, .6, .8), (5, 4, 3, 2, 1)):
                _ = frozen.ppf(p)
                _p = str(int(p * 100))
                label = r'$p_{%s}$=' % _p + f'{_:.1f}'
                results[n]['CDFQuantiles'][f'p{_p}'] = _
                plt.plot(_, p, color=str(_c), ms=ms, lw=0, marker='o', label=label)

            plt.ylabel('Probability'); plt.xlabel('Nearest Neighbour Bins [µm]')
            plt.legend(frameon=False, handletextpad=0.1)
            plt.tight_layout(); plt.savefig(rf'{out_path}\{n}\cumulative-{dist}.png')

            plt.close('all')
            supports.tprint(f'Analyzed data from {n}{self.uc}.', branch=2)

        # construct overview figures
        plt.figure(figsize=(8 / self._pars['FS'], 3.5 / self._pars['FS']), dpi=self['CollectionAnalysisSettings']['FigureDpi'])
        plt.axhline(y=0, color='black', ls='--', lw=.5, alpha=0.5)  # make x-axis line
        _maxs = []; _mins = []
        if self['CollectionAnalysisSettings']['ApplyDataGroupsToNNE'] is False:
            legend_handles = None
            for n, c in zip(data, colors):
                if n in results:
                    frozen = model(*results[n]['ModelParameters'])  # freeze distribution
                    span = np.linspace(results[n]['Min'], results[n]['Max'], 1000)
                    plt.plot(span, frozen.pdf(span), color=c, ms=0, lw=1, label=n)  # area normalized distribution
                    _mins.append(results[n]['Min']); _maxs.append(results[n]['Max'])
        else:
            legend_handles = []
            for (group, members), c in zip(self.groups.items(), self.colors.get(len(self.groups))):
                if self['CollectionPreprocessSettings']['SampleType'] in ('Single-Field', 'Zero-Field'):
                    if self._pars['MG'] is None:
                        group = self.group_editor(group)
                    else:
                        pass
                else:
                    group = self.group_editor(group)

                legend_handles.append(mli.Line2D([], [], color=c, label=group, linewidth=0, markersize=4,
                                                 marker='o'))  # create label placeholder
                for n in members:
                    if n in results:
                        frozen = model(*results[n]['ModelParameters'])  # freeze distribution
                        span = np.linspace(results[n]['Min'], results[n]['Max'], 1000)
                        plt.plot(span, frozen.pdf(span), color=c, ms=0, lw=1)
                        _mins.append(results[n]['Min']); _maxs.append(results[n]['Max'])

        self.__NND_plot_tail(_mins, _maxs, legend_handles=legend_handles)

        nne_path = rf'{self.__write_path}\nearest_neighbor_evaluation'
        base.directory_checker(rf'{nne_path}\svg', clean=False)
        plt.savefig(rf'{nne_path}\svg\Individual NND.svg')
        plt.savefig(rf'{nne_path}\Individual NND.png')

        # write the mean and standard deviation to the analyzed results file
        supports.json_dict_push(r'{}\AnalysisResults.json'.format(self['DirectorySettings']['OutputFolder']),
                            {'NearestNeighbourEvaluation': results}, behavior='update')

        # construct mean model plot for each group if groups are applied to the NNE
        if self['CollectionAnalysisSettings']['ApplyDataGroupsToNNE'] is True:
            plt.figure(figsize=(8 / self._pars['FS'], 3.5 / self._pars['FS']),
                       dpi=self['CollectionAnalysisSettings']['FigureDpi'])
            plt.axhline(y=0, color='black', ls='--', lw=.5, alpha=0.5)  # make x-axis line

            _mins, _maxs = [], []; _lh = None
            if self._pars['MG'] is None:
                for (group, members), c in zip(self.groups.items(), self.colors.get(len(self.groups))):
                    group_pars, __mins, __maxs = [], [], []
                    for n in members:
                        if n in results:
                            group_pars.append(results[n]['ModelParameters'])
                            __mins.append(results[n]['Min']); __maxs.append(results[n]['Max'])

                    min_avg = np.mean(__mins); max_avg = np.mean(__maxs)
                    span = np.linspace(min_avg, max_avg, 1000)
                    try:
                        frozen = model(*np.mean(group_pars, axis=0))  # freeze mean distribution
                    except TypeError:
                        supports.tprint(f'Failed collection distribution analysis for group {group}{self.uc}.', branch=2)
                        continue

                    plt.plot(span, frozen.pdf(span), color=c, ms=0, lw=1, label=self.group_editor(group))
                    _mins.append(min_avg); _maxs.append(max_avg)
            else:
                # determine how many line styles to generate
                style_count = len({j for v in self._pars['MG'].values() for j in v.values()})
                _lh = []; __lh = {}
                for (mg, g), c in zip(self._pars['MG'].items(), self.colors.get(len(self._pars['MG']), hex=True)):
                    _lh.append(mli.Line2D([], [], color=c, label=self._mg_editor(mg), linewidth=0,
                                          markersize=4, marker='s'))
                    group_pars, __mins, __maxs = [], [], []
                    for (tick, (k, v)), ls in zip(enumerate(g.items()), supports.linestyles(style_count)):
                        __lh[v] = mli.Line2D([], [], color='black', label=self.bulk_editor(v), linewidth=1,
                                             linestyle=ls)

                        try:
                            for n in self.groups[k]:
                                if n in results:
                                    group_pars.append(results[n]['ModelParameters'])
                                    __mins.append(results[n]['Min']); __maxs.append(results[n]['Max'])
                        except KeyError:
                            supports.tprint(f'No data for {k}{self.uc}. Skipping group.', branch=2)
                            continue

                        min_avg = np.mean(__mins); max_avg = np.mean(__maxs)
                        span = np.linspace(min_avg, max_avg, 1000)
                        try:
                            frozen = model(*np.mean(group_pars, axis=0))  # freeze mean distribution
                        except TypeError:
                            supports.tprint(f'Failed collection distribution analysis for group {k}{self.uc}.', branch=2)
                            continue

                        plt.plot(span, frozen.pdf(span), color=c, ms=0, lw=1, linestyle=ls)
                        _mins.append(min_avg); _maxs.append(max_avg)
                _lh += list(__lh.values())

            self.__NND_plot_tail(_mins, _maxs, legend_handles=_lh)

            plt.savefig(rf'{nne_path}\svg\Group NND.svg')
            plt.savefig(rf'{nne_path}\Group NND.png')
        plt.close('all')

        for sheet, yl in zip(('ModelMean', 'ModelMedian', 'SpanFWHM', 'Mode'),
                         ('NN Distribution Mean [µm]', 'NN Distribution Median [µm]',
                          r'NN Distribution FWHM [$\Delta$µm]', 'NN Distribution Mode [µm]')):
            plt.figure(figsize=(8 / self._pars['FS'], 3.5 / self._pars['FS']),
                       dpi=self['CollectionAnalysisSettings']['FigureDpi'])
            _lh = [mli.Line2D([], [], color='black', label='Individuals', linewidth=0, markersize=4, marker='o'),
                   mli.Line2D([], [], color='black', label='Mean', linewidth=0, markersize=4, marker='_')]

            if self._pars['MG'] is None:
                for (group, members), c in zip(self.groups.items(), self.colors.get(len(self.groups))):
                    _ = []
                    for n in members:
                        if n in results:
                            value = results[n][sheet]
                            plt.plot(self.group_editor(group), value, color=c, ms=4, marker='o', lw=0, alpha=.7)
                            _.append(value)

                    plt.plot(self.group_editor(group), np.mean(_), color=c, marker='_', ms=6, lw=0)
            else:
                offset = 0; label_conversion = {}
                for (mg, g), c in zip(self._pars['MG'].items(), self.colors.get(len(self._pars['MG']), hex=True)):
                    for tick, (k, v) in enumerate(g.items()):
                        label_conversion[v] = self.bulk_editor(v); _ = []
                        try:
                            for n in self.groups[k]:
                                if n in results:  # if the NND analysis failed, skip the current point
                                    value = results[n][sheet]
                                    plt.plot(tick + offset, value, color=c, ms=4, marker='o', lw=0, alpha=.7)
                                    _.append(value)
                        except KeyError:
                            supports.tprint(f'No data for {k}{self.uc}. Skipping group.', branch=2)
                            continue

                        plt.plot(tick + offset, np.mean(_), color=c, marker='_', ms=6, lw=0,
                                 label=self._mg_editor(mg))
                    _lh.append(mli.Line2D([], [], color=c, label=self._mg_editor(mg), linewidth=0,
                                          markersize=4, marker='s'))
                    offset += .15
                label_ticks = [i + np.median(np.arange(len(self._pars['MG']))) * .15 for i in self._pars['MG_GROUPS'].values()]
                plt.gca().set_xticks(label_ticks, list(label_conversion.values()))

            plt.ylabel(yl)
            plt.legend(frameon=False, handletextpad=0.1, handles=_lh)
            plt.tight_layout()
            plt.gca().spines['right'].set_visible(False)  # remove right figure border
            plt.gca().spines['top'].set_visible(False)  # remove top figure border

            plt.savefig(rf'{nne_path}\svg\{sheet}.svg')
            plt.savefig(rf'{nne_path}\{sheet}.png')
            plt.close('all')

    @staticmethod
    def linreg(x, *c):
        return c[1] * x + c[0]

    def __NND_plot_tail(self, mins, maxs, legend_handles):
        plt.ylabel('Probability Density')
        plt.xlabel('Nearest Neighbour Distances [µm]')
        if self['CollectionAnalysisSettings']['MaxNND']:
            plt.xlim(min(mins), self['CollectionAnalysisSettings']['MaxNND'])
        else:
            plt.xlim(min(mins), max(maxs))

        if self['CollectionAnalysisSettings']['GraphStyle'] == 'Crisp':
            plt.legend(frameon=True, ncols=2, handletextpad=0.1, facecolor='#ebebeb', edgecolor='#ebebeb',
                       fancybox=False, loc=1, handles=legend_handles)
        elif self['CollectionAnalysisSettings']['GraphStyle'] == 'Crisp (No L-Frame)':
            plt.legend(frameon=False, ncols=2, handletextpad=0.1, handles=legend_handles)
        plt.tight_layout()
        plt.gca().spines['right'].set_visible(False)  # remove right figure border
        plt.gca().spines['top'].set_visible(False)  # remove top figure border


class StatisticalAnalysis:
    """Class that handles statistical analysis for a data set. Requires functionality from the DataAnalysis class to
    function as intended."""

    def __init__(self, anchor, data, settings, pars, out):
        self._anchor = anchor
        self.data = data
        self.out = out
        self.__settings = settings
        self.__pars = pars

        # define general variables
        cols = supports.ColorPresets(self.__settings['CollectionAnalysisSettings']['ColorPreset'])
        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            _ = list(data.values())[0]['Normalized Cell Count']
            self.colors = cols.get(len(data))

            self.n = _.columns.size
            self._ctrl = [1, ] * self.n
            ctrl_name = 'Control'
            if self['CollectionAnalysisSettings']['SampleTypeSettings']['RenameControl'] != '':
                ctrl_name = self['CollectionAnalysisSettings']['SampleTypeSettings']['RenameControl']
            self.ctrl_name = ctrl_name

            mask_name = self['CollectionPreprocessSettings']['MaskSelection']
            pmask_name = self['CollectionAnalysisSettings']['SampleTypeSettings']['StatisticsPresentationMask']
            field_sorting = self['CollectionAnalysisSettings']['SampleTypeSettings']['StatisticsFieldSorting']
            mask = base.load_mask_file(mask_name, strip=True)
            if self['CollectionAnalysisSettings']['PlotSignificance'] != '':
                if pmask_name != 'Native':
                    pmask = base.load_mask_file(rf'presentation_masks\{mask_name}\{pmask_name}', strip=True)
                else:
                    pmask = mask
                pmask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\presentation_masks.json',
                                                         behavior='read')[mask_name][pmask_name]
                mask_ids = {}
                pmask_field_ids = {}
                for k, v in pmask_settings['Enabled'].items():
                    k = base.str_to_tuple(k)
                    if v is True:
                        elem = pmask[k]
                        if pmask_settings['TexMath'] is True:
                            if pmask_settings['NonCursive'] is True:
                                elem = r'$\mathrm{%s}$' % elem
                            else:
                                elem = rf'${elem}$'
                        elif pmask_settings['Numbers'] is True:
                            elem = float(elem)
                        mask_ids[k] = mask[k]
                        pmask_field_ids[k] = elem

                if field_sorting != 'None':
                    sort_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\field_sortings.json',
                                                            behavior='read')[mask_name][pmask_name][field_sorting]
                    fc_handler = base.FieldCodeHandler(pmask_field_ids.values())
                    fc_handler.split_codes(delimiters=sort_settings['Delimiters'], regex=sort_settings['UseRegex'],
                                           white_space=sort_settings['RecWhiteSpace'],
                                           separator=sort_settings['MultiLetterSeparator'])
                    fc_handler.sort_codes(indices=sort_settings['SortingIndices'],
                                          reverse_order=sort_settings['RevOrder'],
                                          field_codes=pmask_field_ids)
                    field_codes = fc_handler.id_codes
                else:
                    field_codes = list(pmask_field_ids.keys())

                csv_codes = {}  # strip the index
                for idx, pcode in pmask_field_ids.items():
                    if pmask_settings['TexMath'] is True:
                        if pmask_settings['NonCursive'] is True:
                            pcode = pcode.lstrip(r'$\mathrm{'); pcode = pcode.rstrip(r'}$')
                        else:
                            pcode = pcode.lstrip(r'$'); pcode = pcode.rstrip(r'$')
                    csv_codes[idx] = pcode

                self.mask_converter = {'IDs': mask_ids, 'SortCodes': field_codes, 'pCodes': pmask_field_ids,
                                       'csvCodes': csv_codes}
            self.indices = [ctrl_name] + list(_.index)
        else:
            _ = list(data.values())[0]['Cell Density']
            if self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup'] == 'None':
                self.colors = cols.get(len(data))
                self.indices = list(data.keys())
            else:
                self.colors = cols.get(len(self.__pars['MG']))
                self.indices = self.__pars['MG_GROUPS'].keys()

        self.slices = [1, 0.05, 0.01, 0.001, 0.0001]

    def __getitem__(self, item):
        return self.__settings[item]

    def analyze(self):
        """Method that wraps the statistical analysis depending on the selected settings in the analysis options."""

        _ps = self['CollectionAnalysisSettings']['PlotSignificance']
        test_type = self['CollectionAnalysisSettings']['StatisticalTest']; pvals = None; _out = None
        _out = rf'{self.out}\{test_type}'; base.directory_checker(_out, False)

        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            for (g, d), c in zip(self.data.items(), self.colors):
                d = d['Normalized Cell Count']
                unpacked = [self._ctrl]; means = [1]; stds = [0]; s_index = [self.ctrl_name]; _index = [self.ctrl_name]
                for idx in self.mask_converter['SortCodes']:
                    points = d.loc[self.mask_converter['IDs'][idx]]
                    s_index.append(self.mask_converter['csvCodes'][idx])
                    _index.append(self.mask_converter['IDs'][idx])
                    unpacked.append(points); means.append(np.mean(points)); stds.append(np.std(points, ddof=1))

                if test_type == 'Tukey\'s HSD':
                    pvals = self.tukey_hsd(unpacked, s_index, _out, g)
                elif test_type == 'T-Test':
                    pvals = self.ttest(unpacked, s_index, _out, g)
                elif test_type == 'One-Way ANOVA':
                    pvals = self.one_way_anova(unpacked, s_index, _out, g)
                if _ps:
                    self.statistics_plot(means, stds, pvals, _index, d, c, _out, g)
        else:
            if self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup'] == 'None':
                unpacked = []; means = []; stds = []
                for g, d in self.data.items():
                    points = d['Cell Density']
                    unpacked.append(points); means.append(np.mean(points)); stds.append(np.std(points, ddof=1))

                if test_type == 'Tukey\'s HSD':
                    pvals = self.tukey_hsd(unpacked, self.indices, _out)
                elif test_type == 'T-Test':
                    pvals = self.ttest(unpacked, self.indices, _out)
                elif test_type == 'One-Way ANOVA':
                    pvals = self.one_way_anova(unpacked, self.indices, _out)

                if _ps:
                    c_dict = dict(zip(self.indices, self.colors))
                    self.statistics_plot(means, stds, pvals, self.data, c_dict[_ps], _out)
            else:
                for (mg, g), c in zip(self.__pars['MG'].items(), self.colors):
                    unpacked = []; means = []; stds = []; feed_data = {}
                    for k, v in g.items():
                        feed_data[v] = self.data[k]
                        points = self.data[k]['Cell Density']
                        unpacked.append(points); means.append(np.mean(points)); stds.append(np.std(points, ddof=1))

                    if test_type == 'Tukey\'s HSD':
                        pvals = self.tukey_hsd(unpacked, _out, group=mg)
                    elif test_type == 'T-Test':
                        pvals = self.ttest(unpacked, _out, group=mg)
                    elif test_type == 'One-Way ANOVA':
                        pvals = self.one_way_anova(unpacked, _out, group=mg)

                    if _ps:
                        self.statistics_plot(means, stds, pvals, self.indices, feed_data, c, _out, g=mg)

    def statistics_plot(self, means, stds, pvals, index, d, c, _out, g=None):
        """Method that plots data from a statistical analysis"""

        # define scalars
        _ratio = 65 / len(self.indices)
        scalars = {
            's': _ratio * 9,
            'linewidth': .95 + _ratio * .05,
            'capsize': 1.5 + _ratio * .5,
            'elinewidth': .75 + _ratio * .05,
            'ps': 6 + 10 * _ratio * .4,
            'p-shift': max(means) * .05 * (.95 + _ratio * .05),
            'pms': 3.8 + 4 * _ratio * .05
        }

        _ps = self['CollectionAnalysisSettings']['PlotSignificance']
        fs = self['CollectionAnalysisSettings']['FigureScalar']
        fig = plt.figure(figsize=(8 / fs, 3.5 / fs), dpi=self['CollectionAnalysisSettings']['FigureDpi'])

        # generate plot data
        _means = means; _stds = stds
        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            _indices = [index[0]]
            _conversion = _pval_conversion = {index[0]: (index[0], means[0])}
            for _id, mean in zip(self.mask_converter['SortCodes'], _means[1:]):
                _indices.append(self.mask_converter['pCodes'][_id])
                _conversion[self.mask_converter['IDs'][_id]] = (self.mask_converter['pCodes'][_id], mean)
                _pval_conversion[self.mask_converter['csvCodes'][_id]] = (self.mask_converter['pCodes'][_id], mean)
        else:
            _indices = index
            _conversion = dict(zip(_indices, zip(_indices, means)))
            _pval_conversion = _conversion

            # respect bulk changes to group names
            if self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup'] == 'None':
                _indices = [self._anchor.group_editor(i) for i in _indices]
            else:
                _indices = [self._anchor.bulk_editor(i) for i in _indices]

        _dd = self['CollectionAnalysisSettings']['DisplayDetail']
        if _dd == 'Std':
            plt.bar(_indices, _means, color=c, yerr=_stds, capsize=scalars['capsize'],
                    error_kw={'elinewidth': scalars['elinewidth'], 'capthick': scalars['elinewidth']},
                    ecolor=supports.highlight(supports.rgb_to_hex(c), -20))
        else:
            plt.bar(_indices, _means, color=c)

        for code, p in zip(pvals.iloc[0].keys(), pvals[_ps]):
            if p <= .05 and code in _pval_conversion:
                x_label, y_label = _pval_conversion[code]

                # respect bulk changes to group names
                if self['CollectionPreprocessSettings']['SampleType'] in ('Single-Field', 'Zero-Field'):
                    if self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup'] == 'None':
                        x_label = self._anchor.group_editor(x_label)
                    else:
                        x_label = self._anchor.bulk_editor(x_label)

                plt.scatter(x_label, y_label, marker='|', s=scalars['ps'], color='black', zorder=0)
                if p <= .0001:
                    plt.scatter(x_label, y_label + scalars['p-shift'], marker='D', s=scalars['ps'], color='black', zorder=10)
                elif p <= .001:
                    plt.scatter(x_label, y_label + scalars['p-shift'], marker='P', s=scalars['ps'], color='black', zorder=10)
                elif p <= .01:
                    plt.scatter(x_label, y_label + scalars['p-shift'], marker='v', s=scalars['ps'], color='black', zorder=10)
                else:
                    plt.scatter(x_label, y_label + scalars['p-shift'], marker='$*$', s=scalars['ps'], color='black', zorder=10)

        if _dd == 'Data Points':
            if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
                plt.scatter(self.indices[0], 1, edgecolors=supports.highlight(supports.rgb_to_hex(c), -20),
                            facecolor='none', s=scalars['s'], linewidth=scalars['linewidth'], zorder=5, alpha=.5)
                for idx in d.index:
                    if idx in _conversion:
                        values = d.loc[idx]
                        plt.scatter([_conversion[idx][0], ] * len(values), values, facecolor='none', s=scalars['s'],
                                    alpha=.5, edgecolors=supports.highlight(supports.rgb_to_hex(c), -20), zorder=5,
                                    linewidth=scalars['linewidth'])
            else:
                for group, values in d.items():
                    values = values['Cell Density']

                    # respect bulk changes to group names
                    if self['CollectionPreprocessSettings']['SampleType'] in ('Single-Field', 'Zero-Field'):
                        if self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup'] == 'None':
                            group = self._anchor.group_editor(group)
                        else:
                            group = self._anchor.bulk_editor(group)

                    plt.scatter([group, ] * len(values), values, facecolor='none', s=scalars['s'], alpha=.5,
                                linewidth=scalars['linewidth'],
                                edgecolors=supports.highlight(supports.rgb_to_hex(c), -20), zorder=5)

        # configure plot parameters
        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            plt.ylabel('Relative Cell Count')
        else:
            plt.ylabel('Cell Density [cells cm$^{-2}$]')

        legend_handles = []
        for pval, stars in zip(self.slices[1:], ('$*$', 'v', 'P', 'D')):
            legend_handles.append(
                mli.Line2D([], [], color='black', label=rf'p ≤ {pval}', linewidth=0,
                           markersize=scalars['pms'], marker=rf'{stars}'))  # create label placeholder

        if self['CollectionAnalysisSettings']['GraphStyle'] == 'Crisp':
            plt.legend(frameon=True, ncols=4, handletextpad=0.1, facecolor='#ebebeb', edgecolor='#ebebeb',
                       fancybox=False, loc=1, handles=legend_handles)
        elif self['CollectionAnalysisSettings']['GraphStyle'] == 'Crisp (No L-Frame)':
            plt.legend(frameon=False, ncols=4, handletextpad=0.1, handles=legend_handles)
        plt.xticks(rotation=self['CollectionAnalysisSettings']['StatisticsLabelRotation'],
                   fontsize=self['CollectionAnalysisSettings']['StatisticsLabelSize'])
        plt.gca().set_ymargin(.2)
        plt.tight_layout()
        plt.gca().spines['right'].set_visible(False)  # remove right figure border
        plt.gca().spines['top'].set_visible(False)  # remove top figure border

        test_type = self['CollectionAnalysisSettings']['StatisticalTest']
        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            _sorting = self['CollectionAnalysisSettings']['SampleTypeSettings']['StatisticsFieldSorting']
            plt.savefig(rf'{_out}\{test_type}_{g}_{_ps}_{_dd}_{_sorting}.svg')
            plt.savefig(rf'{_out}\{test_type}_{g}_{_ps}_{_dd}_{_sorting}.png')
        else:
            if self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup'] != 'None':
                plt.savefig(rf'{_out}\{test_type}_{g}_{_ps}_{_dd}.svg')
                plt.savefig(rf'{_out}\{test_type}_{g}_{_ps}_{_dd}.png')
            else:
                plt.savefig(rf'{_out}\{test_type}_{_ps}_{_dd}.svg')
                plt.savefig(rf'{_out}\{test_type}_{_ps}_{_dd}.png')

    def tukey_hsd(self, unpacked_data, index, output_folder, group=None):
        """Method that performs the Tukey HSD method for a data set."""

        t_hsd = sp.stats.tukey_hsd(*unpacked_data)
        pval_df = pd.DataFrame(np.tril(t_hsd.pvalue), index=index, columns=index)
        pval_df = pval_df[0 <= pval_df]
        stat_df = pd.DataFrame(t_hsd.statistic, index=index, columns=index)

        if self['CollectionAnalysisSettings']['ExcelStatistics']:
            _out = self._group_excel_stat_out(output_folder, "Tukey's HSD", group)
            base.excel_df_push(_out, dataframes=[pval_df[pval_df <= i] for i in self.slices] + [stat_df],
                               sheets=[rf'≤ {i}' for i in self.slices] + ['statistic'], behavior='replace')

        return pd.DataFrame(t_hsd.pvalue, index=index, columns=index)

    def ttest(self, unpacked_data, index, output_folder, group=None):
        """Method that performs the T-Test for a data set."""

        pvals, stats, dfs = [], [], []
        for row in unpacked_data:
            _pval_row, _stat_row, _df_row = [], [], []
            for col in unpacked_data:
                _ttest = sp.stats.ttest_ind(row, col)
                _pval_row.append(_ttest.pvalue); _stat_row.append(_ttest.statistic); _df_row.append(_ttest.df)
            pvals.append(np.array(_pval_row))
            stats.append(np.array(_stat_row))
            dfs.append(np.array(_df_row))
        pval_df = pd.DataFrame(np.tril(pvals), index=index, columns=index)
        pval_df = pval_df[0 <= pval_df]
        stat_df = pd.DataFrame(stats, index=index, columns=index)
        df_df = pd.DataFrame(np.tril(dfs), index=index, columns=index)
        df_df = df_df[0 <= df_df]

        if self['CollectionAnalysisSettings']['ExcelStatistics']:
            _out = self._group_excel_stat_out(output_folder, 'T-Test', group)
            base.excel_df_push(_out, dataframes=[pval_df[pval_df <= i] for i in self.slices] + [stat_df, df_df],
                               sheets=[rf'≤ {i}' for i in self.slices] + ['statistic', 'df'], behavior='replace')

        return pd.DataFrame(pvals, index=index, columns=index)

    def one_way_anova(self, unpacked_data, index, output_folder, group=None):
        """Method that performs the one-way ANOVA method for a data set."""

        pvals, stats = [], []
        for row in unpacked_data:
            _pval_row, _stat_row = [], []
            for col in unpacked_data:
                _stat, _pval = sp.stats.f_oneway(row, col)
                _pval_row.append(_pval); _stat_row.append(_stat)
            pvals.append(np.array(_pval_row)); stats.append(np.array(_stat_row))
        pval_df = pd.DataFrame(np.tril(pvals), index=index, columns=index)
        pval_df = pval_df[0 <= pval_df]
        stat_df = pd.DataFrame(stats, index=index, columns=index)

        if self['CollectionAnalysisSettings']['ExcelStatistics']:
            _out = self._group_excel_stat_out(output_folder, 'One-Way ANOVA', group)
            base.excel_df_push(_out,
                               dataframes=[pval_df[pval_df <= i] for i in self.slices] + [stat_df],
                               sheets=[rf'≤ {i}' for i in self.slices] + ['statistic'], behavior='replace')

        return pd.DataFrame(pvals, index=index, columns=index)

    def _group_excel_stat_out(self, out, stat, group=None):
        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field' or (
                self['CollectionPreprocessSettings']['SampleType'] in ('Single-Field', 'Zero-Field') and
                self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup'] != 'None'):
            _out = rf'{out}\{stat}_{group}.xlsx'
        else:
            _out = rf'{out}\{stat}.xlsx'
        return _out

class ImageAnalysis:
    """Class that handles image analysis for the Cellexum application."""

    def __init__(self):
        # fetch settings and set up default parameters
        self.mask_channel = None
        self.__pars = {}
        self.dirs = supports.json_dict_push(rf'{supports.__cache__}\settings.json',
                                        behavior='read')['DirectorySettings']

    def __getitem__(self, item):
        return self.__pars[item]

    def __setitem__(self, key, value):
        self.__pars[key] = value

    def construct_orientation_matrix(self):
        """Method that determines the reference orientation matrix."""

        # import settings
        ORM_settings = supports.setting_cache('PreprocessingSettings')
        mask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                            behavior='read')[ORM_settings['SampleType']][ORM_settings['ImageMask']]
        or_name = mask_settings['OrientationReference']['SampleName']

        # ensure that all required files are present before running the script
        channels = base.LoadImage(self.dirs['Repeat:0'], rf'Repeat:0:{or_name}')
        rotation = mask_settings['OrientationReference']['Rotate']

        # rotate reference image
        (h, w) = channels['MaskChannel'].shape[:2]
        (cX, cY) = (w // 2, h // 2)
        _ = {}
        if rotation != 0:
            for name, channel in channels.channels.items():
                _M = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
                _[name] = cv2.warpAffine(channel, _M, (w, h))
        channels.channels = _
        self.mask_channel = channels['MaskChannel']

        _mc = base.MaskConstruction(channels)
        index_mask, _settings = _mc.multi_field_identification(return_settings=True, orc=True)

        self['MaskSelection'] = ORM_settings['ImageMask']
        self['ColorMatrix'], self['ColorSquareBounds'] = get_color_matrix(self.mask_channel, index_mask)
        self['Tbox'] = list(index_mask.values())
        # save matrix for future use
        update_dict = {ORM_settings['SampleType']: {ORM_settings['ImageMask']: {
                'OrientationReference': {
                    'Align': _settings['FieldParameters']['Align'],
                    'Width': _settings['FieldParameters']['Width'],
                    'Height': _settings['FieldParameters']['Height'],
                    'ReferenceMatrix': {str(k): v for k, v in self['ColorMatrix'].items()}
        }}}}
        supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json', params=update_dict, behavior='update')

    def create_reference_mask_image(self):
        """Method that creates the masked image."""
        out_dir = r'{}\_misc'.format(self.dirs['Repeat:0']['OutputFolder'])
        img = cv2.cvtColor(self.mask_channel, cv2.COLOR_GRAY2RGB)
        font_scale = 4 / 8700 * img.shape[1]

        if not os.path.exists(out_dir):  # check directory existence
            os.makedirs(out_dir)

        # draw onto the orientation reference image
        mask_dict = base.strip_dataframe(base.load_mask_file(self['MaskSelection']))
        cv2.drawContours(img, self['Tbox'], -1, (0, 255, 255), 2)
        for (k, v), cnt, tb in zip(self['ColorMatrix'].items(), self['ColorSquareBounds'], self['Tbox']):
            cv2.drawContours(img, cnt, -1, [v,] * 3, cv2.FILLED)
            x, y, w, h = cv2.boundingRect(tb)
            cv2.putText(img, text=str(int(np.round(v, 0))), org=(int(x + w * .05), int(y + h * .15)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=3)
            cv2.putText(img, text=mask_dict[k], org=(int(x + w * .05), int(y + h * .95)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=(0, 255, 255), thickness=3)

        cv2.imwrite(rf'{out_dir}\OrientationReference.tiff', img)

        img = base.criterion_resize(img)  # set low res image
        cv2.imwrite(rf'{out_dir}\OrientationReference_preview.tiff', img)

    def compare_matrix(self, index_mask, channels):
        """Method that compares a surface of fields to the reference orientation matrix and rotates it accordingly.
        :param index_mask: the indexed mask dict of mask contours
        :param channels: the loaded image channels to compare from the LoadImage class in base."""

        # import settings
        dirs = channels.dirs
        settings = supports.json_dict_push(rf'{dirs["OutputFolder"]}\Settings.json', behavior='read')
        mask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                            behavior='read')[settings['CollectionPreprocessSettings']['SampleType']][
            settings['CollectionPreprocessSettings']['MaskSelection']]

        # determine the correct orientation related to the reference
        if settings['IndividualPreprocessSettings'][channels.metadata['FileName']]['Rotate'] != 'Auto':
            optimal_rotation = [int(settings['IndividualPreprocessSettings'][channels.metadata['FileName']]['Rotate']),
                                0, np.nan]
        else:
            # field_grid = get_point_mask(tbox, mask.point_separation, pars['FieldParameters']['Align'])
            color_dict, _ = get_color_matrix(channels['MaskChannel'], index_mask)
            color_matrix = base.craft_dataframe(color_dict).to_numpy()
            reference_matrix = base.craft_dataframe(
                {base.str_to_tuple(k): v for k, v in mask_settings['OrientationReference']['ReferenceMatrix'].items()
                 }).to_numpy()
            rotation_deviance = []
            for i in range(4):
                division_matrix = np.rot90(color_matrix, k=i) / reference_matrix
                _std = np.std(division_matrix, ddof=1)
                rotation_deviance.append([i * 90, _std])
            sorted_rotations = sorted(rotation_deviance, key=operator.itemgetter(1))
            optimal_rotation = sorted_rotations[0] + [sorted_rotations[1][1] / sorted_rotations[0][1]]

        self['Rotate'] = optimal_rotation
        update_dict = {'PreprocessedParameters': {channels.metadata['FileName']: {'FieldParameters': {
            'Rotate': optimal_rotation[0],
            'ComparedMatrixUniformity': optimal_rotation[1],  # (lower is better)
            'RotationCertainty': optimal_rotation[2]  # (higher is better)
        }}}}
        supports.json_dict_push(rf'{dirs["OutputFolder"]}\Settings.json', params=update_dict, behavior='update')

        return optimal_rotation

    def rotate_matrix(self, channels, index_mask):
        """Method that rotates the reference orientation matrix."""
        dirs = channels.dirs
        (h, w) = channels['MaskChannel'].shape[:2]
        (cX, cY) = (w // 2, h // 2)
        font_scale = 3 / 8700 * w

        _M = cv2.getRotationMatrix2D((cX, cY), self['Rotate'][0], 1.0)

        # rotate input images
        rotated_channels = {}
        for name, channel in channels.channels.items():
            try:
                rotated_channels[name] = cv2.warpAffine(channel, _M, (w, h))
            except cv2.error:
                raise RuntimeError('Failed to rotate channel {} for {}. This is likely due to a corrupted channel file. '
                                   'If so, delete the corrupted file and retry.'.format(
                    name, channels.metadata['FileName']))

        # rotate mask contours
        im_keys = index_mask.keys()
        im_values = index_mask.values()
        key_matrix = base.craft_dataframe(dict(zip(im_keys, im_keys))).to_numpy()
        rim_keys = np.rot90(key_matrix, k=-self['Rotate'][0] / 90).flatten()
        rim_values = contour_rotate(im_values, self['Rotate'][0], [cX, cY])
        rotated_index_mask = dict(zip(tuple(map(tuple, rim_keys)), rim_values))

        # write control image for manual evaluation
        _mask_rotations = rf'{dirs["OutputFolder"]}\_misc\mask_rotations'
        if not os.path.exists(_mask_rotations):
            os.makedirs(_mask_rotations)

        _img = cv2.cvtColor(rotated_channels['MaskChannel'], cv2.COLOR_GRAY2BGR)  # DEV TEST
        cv2.drawContours(_img, list(im_values), -1, (0, 0, 255), 10)
        cv2.drawContours(_img, rim_values, -1, (0, 255, 0), 10)
        for i in im_keys:
            rotc = base.contour_center(rotated_index_mask[i])
            x, y = base.contour_center(index_mask[i])
            cv2.putText(_img, str(i), rotc,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 10)
            cv2.putText(_img, str(i), (x, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 10)

        cv2.putText(_img, 'Green: Mask fields AFTER rotation', (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 10)
        cv2.putText(_img, 'Red: Mask fields BEFORE rotation', (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 10)
        _img = base.criterion_resize(_img)  # downscale to MaxSize pixels
        cv2.imwrite(rf'{_mask_rotations}\{channels.metadata["FileName"]}.png', _img)

        return rotated_index_mask, rotated_channels


def contour_rotate(contours: list[np.ndarray], angle: float, center: list[float]) -> list[np.ndarray]:
    """Function that rotates a set of contours.
    :param contours: the contours to rotate.
    :param angle: int, float, the rotation angle in degrees.
    :param center: the center of the rotation."""
    rotated_contours = []
    for c in contours:
        norm_c = c - center  # move contour to image center
        xs, ys = norm_c.T  # fetch x and y coordinates for corners
        rs, thetas = np.sqrt(xs ** 2 + ys ** 2), np.arctan2(ys, xs)  # convert to polar coordinates
        thetas += np.deg2rad(-angle)  # turn the contour points
        rotated_c = np.array([rs * np.cos(thetas), rs * np.sin(thetas)]).T + center  # revert to cartesian
        rotated_contours.append(np.int64(rotated_c))
    return rotated_contours


def get_color_matrix(img, index_mask):
    """Function that determines the average color of a 1/4 square within each masked field."""
    color_matrix = {}
    square_bounds = []
    for cid, c in index_mask.items():
        # get bounding box and reduce its size by half in all dimensions
        (x, y), (w, h), a = cv2.minAreaRect(c)
        w //= 2
        h //= 2
        box = cv2.boxPoints(((x, y), (w, h), a))
        cnt = np.int64(box)
        x, y, w, h = cv2.boundingRect(cnt)
        color_matrix[cid] = np.mean(img[y:y + h, x:x + w])
        square_bounds.append([cnt])
    return color_matrix, square_bounds


def get_point_mask(tbox, point_separation, angle):
    """Function that defines coordinates for each field of the mask."""
    aa_field_separation = (point_separation * np.cos(np.deg2rad(angle)))
    point_grid = [i[0] for i in tbox]
    _ys, _xs = list(zip(*point_grid))  # note that x = columns and y = rows
    _ox, _oy = min(_xs), min(_ys)
    field_grid = list(zip([int(np.round((i - _ox) / aa_field_separation, 0)) for i in _xs],
                          [int(np.round((i - _oy) / aa_field_separation, 0)) for i in _ys]))
    return field_grid


class CellDetector:
    def __init__(self, img):

        self.slice = None
        self.__mean_colors = []
        self.points = []
        self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.img = {
            'RAW': img,
            'GRAY': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            'SIZE': img.shape[:2]
        }
        self.cell = {
            'AXIS': (23.5, 16.3),  # in µm
            'SCALE': 2.5595238095238075 ** -1  # µm to pix
        }
        self.cell['SEMI-AXIS'] = [i // 2 for i in self.cell['AXIS']]
        self.cell['AREA'] = np.pi * np.prod(self.cell['SEMI-AXIS'])

    def set_slice(self, center: Sequence[int, int], size: Sequence[int, int], **kwargs) -> None:
        """Method that defines a slice of the image based on a center coordinate and slice size.
        :param center: Coordinate (x, y) of the center of the slice.
        :param size: Size (w, h) of the slice.
        :keyword corner: If True, the center argument will instead act as the top left corner of the slice.
        Default is False."""

        pars = {
            'corner': False
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        if kwargs['corner'] is False:
            _s = (size[0] // 2, size[1] // 2)
            gray = self.img['GRAY'][center[1] - _s[1]:center[1] + _s[1], center[0] - _s[0]:center[0] + _s[0]]
            raw = self.img['RAW'][center[1] - _s[1]:center[1] + _s[1], center[0] - _s[0]:center[0] + _s[0]]
        else:
            gray = self.img['GRAY'][center[1]:center[1] + size[1], center[0]:center[0] + size[0]]
            raw = self.img['RAW'][center[1]:center[1] + size[1], center[0]:center[0] + size[0]]
        self.slice = {
            'RAW': raw,
            'GRAY': gray,
            'SIZE': size,
            'MEAN': np.mean(gray),
            'MAX': np.max(gray),
        }

    def ellipse_finder(self, tonset: int=10, trange: int=100, clearance: int | float=0, **kwargs) -> bool | None:
        """Method that detects ellipses on an image with a certain binary threshold defined by the args.
        :param tonset: Threshold onset for binary conversion.
        :param trange: Threshold range for binary conversion.
        :param clearance: For each iteration, the allowed cell area decreases and the blocked area increases as the
        threshold becomes narrower and brighter. The clearance sets the relative incremental change for these increases
        and decreases.
        :keyword shift: Shifts the found ellipse contours with (x, y) when storing them. Default is (0, 0)."""

        pars = {
            'shift': (0, 0)
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        binary = cv2.threshold(self.slice['ITER'], tonset, tonset + trange, cv2.THRESH_BINARY)[1]  # set threshold
        if binary.size == binary[binary == 0].size:
            return False

        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]  # find contours
        for c in contours:
            if 5 < len(c):  # fit ellipse to contour if contour has more than 5 pixels
                (cX, cY), (axMaj, axMin), angle = cv2.fitEllipse(c)  # fit ellipse to contour
                e_area = np.pi * axMaj * axMin * self.cell['SCALE'] ** -2  # determine the fitted ellipse area

                if (all(l < max(self.cell['AXIS']) * self.cell['SCALE'] for l in (axMaj, axMin))
                        and self.cell['AREA'] * (2 - clearance * 4) > e_area > self.cell['AREA'] * .5):
                    cv2.ellipse(self.slice['ITER'], (int(cX), int(cY)),
                                (int(axMaj * (1 + clearance * axMaj / axMin)), int(axMin * (1 + clearance))), angle, 0, 360,
                                (0, 0, 0), -1)
                    self.__mean_colors.append(int(self.slice['MEAN']))
                    self.points.append(c + kwargs['shift'])

    def slice_iterator(self, expand: int | float=2.2, **kwargs) -> None:
        """Method that iterates different binary thresholds with ellipse_finder over a slice of the image. The slice
        must be defined through set_slice. Keyword arguments are passed directly to the ellipse_finder method.
        :param expand: Extend the slice mean by a factor of expand before performing tozero."""
        t_mean = int(self.slice['MEAN'] * expand)
        tozero = cv2.threshold(self.slice['GRAY'], t_mean,
                               255, cv2.THRESH_TOZERO)[1]  # aggressive background removal
        self.slice['ITER'] = tozero

        for i in range(self.slice['MAX'] - t_mean):
            # iterate over as much as possible, and narrow down the range each cycle
            proceed = self.ellipse_finder(tonset=t_mean + i, trange=self.slice['MAX'] - t_mean - i, **kwargs)
            if proceed is False:
                break  # quit the loop early if the entire binary is black

    def image_iterator(self, size: int, **kwargs) -> None:
        """Method that iterates different binary thresholds with ellipse_finder over the entire input image through
        smaller image slices.
        :param size: The size of each slice. If slice size['SIZE'] >= img['SIZE'], the image size will be used
        instead.
        :keyword expand_cycles: The number of times to iterate through the image with a decreased expand coefficient
        for the slice_iterator.
        :keyword expand_step_size: The expand coefficient increase for each cycle of iterations.
        :keyword expand_onset: The floor value for the expand coefficient during the cycles.
        :keyword clearance: Refer to the ellipse_finder doc string."""

        pars = {
            'expand_cycles': 1,
            'expand_step_size': .1,
            'expand_onset': 2.2,
            'clearance': .03,
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        overlap = int(max(self.cell['AXIS']) * self.cell['SCALE'])
        for cycle in range(kwargs['expand_cycles']):
            for x in range(self.img['SIZE'][0] // size + 1):
                for y in range(self.img['SIZE'][1] // size + 1):
                    shift = (x * size, y * size)
                    self.set_slice(center=shift, size=(size + overlap, size + overlap), corner=True)
                    self.slice_iterator(expand=kwargs['expand_onset'] + (cycle * kwargs['expand_step_size']), shift=shift,
                                        clearance=kwargs['clearance'])
            for c, col in zip(self.points, self.__mean_colors):  # black-out contours in the gray image for future cycles
                color = (col, col, col)
                (cX, cY), (axMaj, axMin), angle = cv2.fitEllipse(c)  # fit ellipse to contour
                cv2.ellipse(self.img['GRAY'], (int(cX), int(cY)),
                            (int(axMaj * (1 + kwargs['clearance'] * axMaj / axMin)),
                             int(axMin * (1 + kwargs['clearance']))), angle, 0, 360, color=color, thickness=-1)

        # remove ellipses overlapping others with more than 50% of the minor axis
        xCs, yCs = [], []
        for c in self.points:
            ellipse = cv2.fitEllipse(c)
            xCs.append(ellipse[0][0])
            yCs.append(ellipse[0][1])

        _xs, _ys = np.array(xCs), np.array(yCs)
        _n = 0
        for x, y in zip(xCs, yCs):
            vector_lengths = np.sqrt(np.square(x - _xs) + np.square(y - _ys))
            vector_check = sorted(vector_lengths)[1]
            if vector_check < min(self.cell['SEMI-AXIS']) * self.cell['SCALE']:
                self.points = self.points[:_n] + self.points[_n + 1:]
                _xs, _ys = np.delete(_xs, [_n]), np.delete(_ys, [_n])
            else:  # update index if a contour is not removed in place
                _n += 1
    
    @staticmethod
    def connected_component_analysis(gray, bgkern: int, ekern: int, dkern: int):
        clear = base.background_subtraction(gray, bgkern)  # remove background
        erosion = cv2.erode(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ekern, ekern)), iterations=1)
        dilation = cv2.dilate(erosion, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dkern, dkern)), iterations=1)
        otsu = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        centroids = cv2.connectedComponentsWithStats(otsu, 4, cv2.CV_32S)[-1][1:]

        return centroids
    
    def connected_component_counting(self, size):
        
        if size not in ('', 0, -1):
            centroids = np.array([])
            for x in range(self.img['SIZE'][0] // size + 1):
                for y in range(self.img['SIZE'][1] // size + 1):
                    shift = (x * size, y * size)
                    self.set_slice(center=shift, size=(size, size), corner=True)
                    _ = self.connected_component_analysis(self.slice['GRAY'], 11, 3, 3)
                    _ += shift
                    try:
                        centroids = np.concatenate((centroids, _), axis=0)
                    except ValueError:
                        centroids = _
        else:
            centroids = self.connected_component_analysis(self.img['GRAY'], 11, 3, 3)

        self.points = centroids

