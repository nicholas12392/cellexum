import os
import cv2
import numpy as np
import pandas as pd

import analysis
import re
import operator
import scipy as sp
import math
import bioformats
import javabridge
from bioformats import logback
from bioformats.omexml import qn
import supports
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont
import warnings
import time
from tkinter import filedialog
import tifffile as tif
import logging
from supports import lprint as print

warnings.filterwarnings('ignore', category=sp.optimize.OptimizeWarning)


class SampleParameters:
    """Defines an instance with sample parameters. Needs to be in units of pix:µm."""

    def __init__(self, scalar, **kwargs):
        self.__scalar = scalar  # in pix/µm
        self.params = {}

        if 'configure' in kwargs:
            self.configure(kwargs['configure'])

    def configure(self, sample):
        if sample == 'NTSA':
            self.params = {'length': 20e3 * self.__scalar,
                           'field_length': 2.1e3 * self.__scalar,
                           'field_separation': .4e3 * self.__scalar}
            self['field_area'] = self['field_length'] ** 2
        if sample in ('Fibroblast', 'fibroblast'):
            self.params = {'nuclei_radius': np.array([5, 27]) * self.__scalar}
            self['nuclei_area'] = np.square(self.params['nuclei_radius']) * np.pi

    def __getitem__(self, item):
        return self.params[item]

    def __setitem__(self, key, value):
        self.params[key] = value


class PixelParameters:
    """Defines an instance with sample parameters in pixels. Needs to be in units of pix:µm."""

    def __init__(self, scalar, dirs, **kwargs):
        self.scalar = scalar ** -1  # in pix/µm
        _ = supports.setting_cache()
        self.__out_path = dirs['OutputFolder']
        mask_settings = _['PreprocessingSettings']
        self.__mask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json',
                                                       behavior='read')[
            mask_settings['SampleType']][mask_settings['ImageMask']]

        self.params = {}
        self.configure(mask_settings['SampleType'])  # auto-configure according to sample type

    def configure(self, ctype):
        params = self.__mask_settings

        # set up scalar
        unit_conversion = {'nm': 1e-3, 'µm': 1, 'mm': 1e3, 'cm': 1e4}
        try:
            _scale = self.scalar * unit_conversion[params['FieldUnits']]
        except KeyError:
            raise ValueError(f'Unsupported field units: {params["FieldUnits"]}.')

        # set up masking parameters
        if ctype == 'Multi-Field':
            for k in ('FieldWidth', 'FieldHeight', 'FieldSpacingX', 'FieldSpacingY'):
                params[k] = int(np.round(params[k] * _scale, 0))
            params['MaskSpanX'] = params['Columns'] * params['FieldWidth'] + params['FieldSpacingX'] * (
                    params['Columns'] + params['SpacingDeviationX'])
            params['MaskSpanY'] = params['Rows'] * params['FieldHeight'] + params['FieldSpacingY'] * (
                    params['Rows'] + params['SpacingDeviationY'])
        elif ctype == 'Single-Field':
            for k in ('FieldWidth', 'FieldHeight'):
                params[k] = int(np.round(params[k] * _scale, 0))

        params['FieldArea'] = params['FieldWidth'] * params['FieldHeight']
        self.params = params

    def __getitem__(self, item):
        return self.params[item]

    def __setitem__(self, key, value):
        self.params[key] = value


class LoadImage:
    """Class for the Cellexum application that handles the image reading and conversion of .vsi files."""
    def __init__(self, dirs, file):
        self.channels = {}  # placeholder dict for storing loaded image layers

        file_repeat, self.file_name = file.split(':')[1:]
        self.file = file
        self.file_repeat = f'Repeat:{file_repeat}'
        self.dirs = dirs

        # load masking channel name
        self.__mask_channel = supports.json_dict_push(rf'{dirs["OutputFolder"]}\Settings.json', behavior='read')[
            'CollectionPreprocessSettings']['MaskChannel']

        self.__file_out = rf'{dirs["OutputFolder"]}\{self.file_name}'
        self.metadata = supports.json_dict_push(rf'{self.__file_out}\metadata.json', behavior='read')
        self.cache = supports.json_dict_push(rf'{supports.__cache__}\saves.json',
                                             behavior='read')[dirs['InputFolder']]

        _ = set([v['InputFolder'] for k, v in supports.setting_cache('DirectorySettings').items() if k != 'CommonOutputFolder'])

        self.channel_linker = {}
        if 'LinkSetup' in self.cache:
            if _ == set(self.cache['LinkSetup']['Criterion']):
                self.channel_linker = self.cache['LinkSetup']['Linker']

        for n, channel in enumerate(self.metadata['ImageData']['Channels'].keys()):
            self.load_channel(channel)

    def load_channel(self, c_name):
        """Method that loads a channel image.
        :param c_name: the channel name"""

        """Due to the initial handling of the .vsi to .tif, the resulting .tif files will lack associated metadata. This 
        causes a warning log from tifffile, which is turned off here."""
        logger = logging.getLogger('tifffile')
        logger.disabled = True

        gray = tif.imread(rf'{self.__file_out}\{c_name}.tif')
        if self.channel_linker:  # streamline channel name to link if set up
            c_name = self.channel_linker[c_name]
        if self.metadata['ImageData']['PixelType'] == 'uint8':
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self[c_name] = img
        else:  # rescale the bit depth to 8 if it is not already
            gray = gray / gray.max() * 255
            gray = gray.astype(np.uint8)
            self[c_name] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if c_name == self.__mask_channel:
            self['MaskChannel'] = gray

    def __getitem__(self, item):
        return self.channels[item]

    def __setitem__(self, key, value):
        self.channels[key] = value


class RawImageHandler:
    def __init__(self):
        self.dirs = None; self.files = None; self.metadata = None

    def handle(self, files):
        self.dirs = supports.json_dict_push(rf'{supports.__cache__}\settings.json', behavior='read')['DirectorySettings']

        if len(self.dirs) == 1:  # inject repeat metadata if there is only one data set to comply with structure
            self.dirs[f'Repeat:0']['Name'] = None; self.dirs[f'Repeat:0']['Uncommon'] = None

        futures = []; counter = 1; total = len(files)
        with concurrent.futures.ProcessPoolExecutor(max_workers=supports.get_max_cpu(),
                                                    max_tasks_per_child=1) as executor:
            for file in files:
                repeat, file_name = file.split(':')[1:]
                futures.append(executor.submit(self._handler, self.dirs[f'Repeat:{repeat}'], file_name))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                uc = f' {result["RepeatUncommon"]}' if result["RepeatUncommon"] is not None else ''
                supports.tprint(f'Handled image file {result["FileName"]}{uc} ({counter}/{total})')
                counter += 1
        supports.tprint('Completed handling image files.')

    @staticmethod
    def _handler(dirs, file):
        javabridge.start_vm(class_path=bioformats.JARS, max_heap_size="2G"); logback.basic_config()
        file_out = r'{}\{}'.format(dirs['OutputFolder'], file)
        if not os.path.isdir(file_out):
            os.makedirs(file_out)

        # look for metadata
        md_path = rf'{file_out}\metadata.json'
        if os.path.isfile(md_path):
            metadata = supports.json_dict_push(md_path, behavior='read')
            metadata['RepeatName'] = dirs['Name']; metadata['RepeatUncommon'] = dirs['Uncommon']
            supports.json_dict_push(md_path, params=metadata, behavior='update')
        else:
            metadata = RawImageHandler.store_metadata(rf'{dirs["InputFolder"]}\{file}.vsi')
            metadata['RepeatName'] = dirs['Name']; metadata['RepeatUncommon'] = dirs['Uncommon']
            supports.json_dict_push(md_path, params=metadata, behavior='update')

        # look for tif conversions
        channels = metadata['ImageData']['Channels'].keys()
        dir_files = [e.removesuffix('.tif') for e in os.listdir(file_out) if e.endswith('.tif')]
        if any(c not in dir_files for c in channels):
            with bioformats.ImageReader(rf'{dirs["InputFolder"]}\{file}.vsi') as reader:
                for n, channel in enumerate(channels):
                    if channel not in dir_files:
                        RawImageHandler.store_channel(metadata, reader, n, channel, file_out)
        javabridge.kill_vm()

        return metadata

    @staticmethod
    def store_channel(metadata, reader, c_id, c_name, file_out):
        """Method that stores a channel image.
        :param metadata: the image metadata from the store_metadata method
        :param reader: the bioformats reader object for the .vsi file
        :param c_id: the channel ID
        :param c_name: the channel name
        :param file_out: the output file directory"""

        cimage = reader.read(c_id, rescale=False)
        bioformats.formatwriter.write_image(rf'{file_out}\{c_name}.tif', cimage,
                                            metadata['ImageData']['PixelType'],
                                            c=c_id, size_c=len(metadata['ImageData']['Channels'].keys()),
                                            channel_names=metadata['ImageData']['Channels'].keys())

    @staticmethod
    def store_metadata(file_path):
        """Method that collects the useful metadata from the .vsi file."""
        meta = bioformats.get_omexml_metadata(file_path)
        metadata = bioformats.omexml.OMEXML(meta)
        omen = OMENavigator(metadata)

        pix = omen.n('Image').n('Pixels')
        c_layers = int(pix['SizeC'])
        c_info = {}
        for layer in range(c_layers):
            try:
                c_name = pix.n('Channel', layer)['Name']
                c_info[c_name] = {}
                for key in ('EmissionWavelength', 'EmissionWavelengthUnit'):
                    c_info[c_name][key] = pix.n('Channel', layer)[key]
                for key in ('ExposureTime', 'ExposureTimeUnit'):
                    c_info[c_name][key] = pix.n('Plane', layer)[key]
            except IndexError:
                supports.tprint('No layer {} channel info for {}.'.format(
                    layer, file_path.split(os.sep)[-1].removesuffix('.vsi')))

        metadata = {
            'FileName': file_path.split(os.sep)[-1].removesuffix('.vsi'),
            'ImageData': {
                'Channels': c_info,
                'ScaleBarX': float(pix['PhysicalSizeX']),
                'ScaleBarY': float(pix['PhysicalSizeY']),
                'ScaleBarXUnit': pix['PhysicalSizeXUnit'],
                'ScaleBarYUnit': pix['PhysicalSizeYUnit'],
                'NominalMagnification': float(omen.n('Instrument').n('Objective')['NominalMagnification']),
                'ImagePixelWidth': int(pix['SizeX']),
                'ImagePixelHeight': int(pix['SizeY']),
                'DataAcquisitionTime': omen.n('Image')['AcquisitionDate'],
                'PixelType': pix['Type'],
            },
            'InstrumentData': {
                'DetectorModel': omen.n('Instrument').n('Detector')['Model'],
                'DetectorManufacturer': omen.n('Instrument').n('Detector')['Manufacturer'],
                'ObjectiveModel': omen.n('Instrument').n('Objective')['Model'],
                'ObjectiveNumericalAperture': float(omen.n('Instrument').n('Objective')['LensNA'])
            }
        }

        metadata['ImageData']['ScaleBarRMS'] = np.sqrt((metadata['ImageData']['ScaleBarX'] ** 2 +
                                                        metadata['ImageData']['ScaleBarY'] ** 2) / 2)
        return metadata


class OMENavigator:
    def __init__(self, ome, node=None):
        self.ome = ome
        self.node = node

    def n(self, name, index=0):
        if self.node is None:
            _ = self.ome.root_node.findall(qn(self.ome.ns['ome'], name))[index]
        else:
            _ = self.node.findall(qn(self.ome.ns['ome'], name))[index]
        return OMENavigator(ome=self.ome, node=_)

    def __getitem__(self, item, reset=True):
        _ = self.node.get(item)
        if _ is None:
            _ = self.node.find(qn(self.ome.ns['ome'], item))
            if _ is not None:
                _ = _.text
        return _


class PreprocessingHandler:
    def __init__(self):

        self.mask_dict = {}
        self.__settings = None

        # import necessary settings
        cache = supports.setting_cache()
        self.dirs = cache['DirectorySettings']
        self._pps = cache['PreprocessingSettings']

        if self._pps['SampleType'] == 'Multi-Field':
            mask = load_mask_file(self._pps['ImageMask'])
            self.mask_dict = strip_dataframe(mask)

        # define attributes
        self.__forced_square_contour = None
        self.channels = None

    def preprocess(self, file) -> dict:
        # set settings and channels for the current file in the process pool
        file_repeat, file_name = file.split(':')[1:]; file_repeat = f'Repeat:{file_repeat}'
        self.dirs = self.dirs[file_repeat]
        self.__settings = supports.json_dict_push(rf'{self.dirs["OutputFolder"]}\Settings.json', behavior='read')
        self.channels = LoadImage(self.dirs, file)

        wpars = {}
        if self._pps['SampleType'] != 'Zero-Field':
            mask_constructor = MaskConstruction(self.channels); rotated_index_mask = None
            if self._pps['SampleType'] == 'Multi-Field':
                index_mask, wpars = mask_constructor.multi_field_identification()
                analyser = analysis.ImageAnalysis()
                optimal_rotation = analyser.compare_matrix(index_mask, self.channels)
                rotated_index_mask, rotated_channels = analyser.rotate_matrix(self.channels, index_mask)
                self.channels.channels = rotated_channels  # update channels dict
                mask_constructor.draw_parameters['PreprocessedParameters'][file_name]['FieldParameters'].update({
                    'Rotate': optimal_rotation[0],
                    'ComparedMatrixUniformity': optimal_rotation[1],  # (lower is better)
                    'RotationCertainty': optimal_rotation[2]  # (higher is better)
                })
                conversion_dict = dict(zip(index_mask.keys(), rotated_index_mask.keys()))
                mask_constructor.draw_parameters['PreprocessedParameters'][file_name] = supports.dict_update(
                    mask_constructor.draw_parameters['PreprocessedParameters'][file_name], {
                        'FieldParameters': {
                            'Rotate': optimal_rotation[0],
                            'ComparedMatrixUniformity': optimal_rotation[1],  # (lower is better)
                            'RotationCertainty': optimal_rotation[2]  # (higher is better)
                        },
                        'DrawParameters': {
                            'RealFieldIndices': [conversion_dict[i] for i in mask_constructor.draw_parameters
                            ['PreprocessedParameters'][file_name]['DrawParameters']['RealFieldIndices']]
                        }
                    })
            elif self._pps['SampleType'] == 'Single-Field':
                index_mask, wpars = mask_constructor.single_field_identification()
                rotated_index_mask = index_mask

            self.create_mask_image(rotated_index_mask, self.channels,
                                   mask_constructor.draw_parameters['PreprocessedParameters'][file_name])
            self.cut_mask(rotated_index_mask, self.channels)
        else:
            wpars = self.handle_zero_field(self.channels)

        return wpars  # return write parameters for the main loop Settings.json

    def handle_zero_field(self, channels) -> dict:
        """Method handles image preprocessing of zero-field sample types.
        :param channels: the channels contained in the mask from the LoadImage class."""

        # remove mask channel from channels
        h, w = channels.channels['MaskChannel'].shape[:2]
        cX, cY, a = w / 2, h / 2, 0
        contour = np.int64(cv2.boxPoints(((cX, cY), (w, h), a)))

        cs = {k: v for k, v in channels.channels.items() if k != 'MaskChannel'}

        for name in cs.keys():
            directory_checker(r'{}\{}\{}'.format(self.dirs['OutputFolder'], channels.file_name, name))

        # cut images according to mask and write the cuts out
        _ = {'SampleName': channels.file_name,
             'ContourMask': {
                 channels.file: {
                     'FieldPosition': channels.file,
                     'FieldContour': contour,
                     'MinAreaRect': [[cX, cY], [w, h], a],
                     'BoundingBox': [0, 0, w, h],
                    'RealCenter': [cX, cY],
                    'RealTopLeft': [0, 0]
                 }
             }}

        for name, channel in cs.items():
            layer_path = r'{}\{}\{}\_{}.tiff'.format(self.dirs['OutputFolder'], channels.file_name,
                                                     name, channels.file_name)
            cv2.imwrite(layer_path, channel)

        supports.json_dict_push(r'{}\{}\maskdata.json'.format(self.dirs['OutputFolder'], channels.file_name),
                                params=_, behavior='replace')

        _ = {'PreprocessedParameters': {channels.file_name: {'FieldParameters': {
            'ScaleBar': channels.metadata['ImageData']['ScaleBarRMS'] ** -1
        }}}}

        return _

    def cut_mask(self, rotated_index_mask: dict | list, channels):
        """Method that cuts a masked image according to the mask.
        :param rotated_index_mask: the rotated index mask. Note if SampleType == Single-Field, this is instead the
        packed contour from the single_field_identification.
        :param channels: the channels contained in the mask from the LoadImage class."""

        # remove mask channel from channels
        cs = {k: v for k, v in channels.channels.items() if k != 'MaskChannel'}

        for name in cs.keys():
            directory_checker(r'{}\{}\{}'.format(self.dirs['OutputFolder'], channels.file_name, name))

        if self._pps['SampleType'] == 'Single-Field':
            rotated_index_mask = {0: rotated_index_mask[0]}
            self.mask_dict = {0: channels.file_name}
            exception = ('Image {} {} could not be cut according to the mask. '.format(channels.file_name, channels.file_repeat) +
                         'The mask is likely misaligned. Try a different \'MaskingMethod\'.')
        elif self._pps['SampleType'] == 'Multi-Field':
            exception = ('Image {} {} could not be cut according to the mask. '.format(channels.file_name, channels.file_repeat) +
                         'The mask is likely misaligned. Try adjusting \'MinFields\' or shift the mask and retry.')

        # cut images according to mask and write the cuts out
        _ = {'SampleName': channels.file_name,
             'ContourMask': {}}  # placeholder dict for maskdata.json
        for cid, c in rotated_index_mask.items():
            _id = str(self.mask_dict[cid])
            x, y, w, h = cv2.boundingRect(c)

            if x < 0:
                x = 0
                supports.tprint('The mask top left corner is negative. The x-coordinate has been corrected to 0.')
            if y < 0:
                y = 0
                supports.tprint('The mask top left corner is negative. The y-coordinate has been corrected to 0.')

            (cX, cY), (mW, mH), a = cv2.minAreaRect(c)

            _['ContourMask'][_id] = {
                'FieldPosition': _id,
                'FieldContour': c,
                'MinAreaRect': [[cX - x, cY - y], [mW, mH], a],
                'BoundingBox': [0, 0, w - x, h - y],
                'RealCenter': [cX, cY],
                'RealTopLeft': [x, y]
            }

            for name, channel in cs.items():
                layer_cut = channel[y:y + h, x:x + w]
                layer_path = r'{}\{}\{}\_{}.tiff'.format(self.dirs['OutputFolder'], channels.file_name, name, _id)
                try:
                    cv2.imwrite(layer_path, layer_cut)
                except cv2.error:
                    raise IndexError(exception)
        supports.json_dict_push(r'{}\{}\maskdata.json'.format(self.dirs['OutputFolder'], channels.file_name),
                                params=_, behavior='replace')

    def create_mask_image(self, index_mask, channels, parameters):
        """Method that creates a mask image."""
        img = cv2.cvtColor(channels['MaskChannel'], cv2.COLOR_GRAY2BGR)
        img_h, img_w = img.shape[:2]  # shape yields (rows, columns, z) -> (y, x, z) or (h, w, z)
        font_scale = 3 / 8700 * img_w
        w, h = parameters['FieldParameters']['Width'], parameters['FieldParameters']['Height']

        # add sample-specific items to image
        if self._pps['SampleType'] == 'Multi-Field':
            for cid, c in index_mask.items():  # add mask contours and IDs to mask channel
                x, y = contour_center(c)
                cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
                cv2.putText(img, text=self.mask_dict[cid], org=(int(x - w * .45), int(y + h * .4)),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=(255, 255, 255), thickness=3)
                cv2.putText(img, text=str(cid), org=(int(x - w * .45), int(y + h * .4 - 120)),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=(0, 255, 255), thickness=3)
                if cid in parameters['DrawParameters']['RealFieldIndices']:  # i.e. if MaskMethod == 'Hybrid'
                    cv2.circle(img, (int(x - w * .3), int(y - h * .3)), 50, (0, 255, 255), -1)

            # write Hi-Res mask image
            cv2.imwrite(r'{}\{}\StructureMask.tiff'.format(self.dirs['OutputFolder'], channels.file_name), img)

            l1_2 = (r'Rotated {Rotate} DEG with Certainty {RotationCertainty:.1f} and Uniformity '
                    r'{ComparedMatrixUniformity:.3f}').format(**parameters['FieldParameters'])
            cv2.putText(img, l1_2, (img_w // 3, 120), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 255), 3)
        elif self._pps['SampleType'] == 'Single-Field':
            cv2.drawContours(img, index_mask, -1, (0, 255, 255), 2)

        # add sample-unspecific items to image
        l1_1 = r'Quality: T{Level:.2f}: +/-{AreaError:.1f}% Area, +/-{RatioError:.3f}% Ratio'.format(
            **parameters['QualityParameters'])
        cv2.putText(img, l1_1, (50, 120), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 255), 3)

        l2 = r'Field Parameters: W{}, H{}, A{:.3f} at scale {:.3f}'.format(
            int(w), int(h), parameters['FieldParameters']['Align'], channels.metadata['ImageData']['ScaleBarRMS'])
        cv2.putText(img, l2, (50, 240), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 255), 3)

        l3 = (r'Image {FileName} Acquired {ImageData[DataAcquisitionTime]} at x{ImageData[NominalMagnification]} with a '
              r'{InstrumentData[DetectorModel]} by {InstrumentData[DetectorManufacturer]} and a '
              r'{InstrumentData[ObjectiveModel]} with NA {InstrumentData[ObjectiveNumericalAperture]}').format(**channels.metadata)
        cv2.putText(img, l3, (50, img_h - 120), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 3)

        # write Low-Res mask image
        out_path = r'{}\_masks for manual control'.format(self.dirs['OutputFolder'])
        lr_path = r'{}\{}.png'.format(out_path, channels.file_name)
        if not os.path.exists(out_path):
            try:  # if more subprocesses finish simultaneously, this path creation can overlap, raising an error
                os.makedirs(out_path)
            except FileExistsError:
                pass
        scaled_img = criterion_resize(img)
        scaled_img = cv2.convertScaleAbs(scaled_img, alpha=2, beta=20)  # contrast enhance and brighten ctrl image
        cv2.imwrite(lr_path, scaled_img)


class MaskConstruction:
    """Class that constructs masks for the Cellexum application.
    :params channels: the loaded image channels to compare from the LoadImage class"""
    def __init__(self, channels):
        self.channels = channels
        self.dirs = channels.dirs
        self.__settings = supports.json_dict_push(rf'{self.dirs["OutputFolder"]}\Settings.json', behavior='read')

        # load mask parameters (scale bar is pixel/µm)
        self.draw_parameters = None
        self.metadata = channels.metadata
        _ = self.metadata['ImageData']['ScaleBarRMS']
        _fr = self.__settings['CollectionPreprocessSettings']['ForceResolution']
        if _fr == '':
            self.scale, self.scale_r = define_scalars(_, 1.7, 2.5, False)
        elif _fr > _:
            self.scale, self.scale_r = define_scalars(_, None, _fr, False)
        else:
            self.scale = _; self.scale_r = 1
        self.img = read_channel(img=channels['MaskChannel'], scalar=self.scale_r)

        self.unc = f' {self.metadata["RepeatUncommon"]}' if self.metadata['RepeatUncommon'] is not None else ''

        self.mask_pars = PixelParameters(self.scale, self.dirs)

        self.blur_image = cv2.medianBlur(self.img, 5)
        self.found_fields = []
        self.TP = None



    def __contour_filter(self, contours, pae=.15, pre=.02):
        """Internal method that makes a rough filtering of the found contours for further processing.
        :param contours: the list of contours to be filtered
        :param pae: the permitted maximum area error
        :param pre: the permitted maximum ratio error"""
        cnt = []
        for c in contours:
            c = cv2.convexHull(c)
            (x, y), (w, h), a = cv2.minAreaRect(c)
            if h > 0:
                c_area = w * h
                c_ratio = w / h
                norm_rel_area = abs(c_area / self.mask_pars['FieldArea'] - 1)
                norm_ratio = abs(c_ratio - self.mask_pars['FieldWidth'] / self.mask_pars['FieldHeight'])

                # find squares by matching height and width ratio
                if norm_rel_area < pae and norm_ratio < pre:
                    # pack contours, contour area, relative contour area, and contour dimension ratio
                    error_score = norm_rel_area * norm_ratio  # determine an error score
                    cnt.append([c, c_area, c_ratio, norm_rel_area, norm_ratio, error_score])
        return cnt

    def __contour_detect(self, onset, span):

        _binary = onset - span
        if _binary < 0:
            _binary = 0

        thresh_img = cv2.threshold(self.blur_image, onset, 255, cv2.THRESH_TOZERO_INV)[1]  # convert to binary
        thresh_img = cv2.threshold(thresh_img, _binary, 255, cv2.THRESH_BINARY)[1]  # convert to binary

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # define morphology kernel
        morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, morph_kernel,
                                     iterations=2)  # apply morphology to threshold

        cont = cv2.findContours(morph_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]  # find contours

        col_ratio = np.mean(morph_img) / 255  # determine binary coverage with the current threshold

        return cont, col_ratio

    @staticmethod
    def _attempt_fit(data) -> list | None:
        """Attempts to fit data to a linear regression. If TypeError occurs, the popt defaults to None. Otherwise popt
        is returned."""
        try:
            popt, _ = sp.optimize.curve_fit(linear_regression, *list(zip(*data)))
        except TypeError:
            popt = None
        return popt

    def multi_field_identification(self, **kwargs):
        # find intensity spans between 10-70% white field ratio with at least settings[MinFields] number of fields

        pars = {
            'return_settings': False,
            'orc': False
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        if kwargs['orc'] is True:
            mask_settings = supports.json_dict_push(rf'{supports.__cwd__}\__misc__\masks.json', behavior='read')[
                self.__settings['CollectionPreprocessSettings']['SampleType']][
                self.__settings['CollectionPreprocessSettings']['MaskSelection']]['OrientationReference']
        else:
            mask_settings = self.__settings['IndividualPreprocessSettings'][self.channels.file_name]
        min_fields = int(mask_settings['MinFields'])

        max_len = 0; intensity_span = mask_settings['OnsetIntensitySpan']; onset_intensity = 0
        trimmed_onset_intensity_span = intensity_span
        contour_sets = []; contour_grid = []
        stop = False
        while not stop:  # iterate intensity thresholds until 70% of the image is white
            cont, white_field_ratio = self.__contour_detect(onset_intensity, intensity_span)  # detect contours
            contours = self.__contour_filter(cont)  # find a rough estimate of the fields in the image

            # discard contour sets with < 10% white field ratio and <= settings[MinFields]
            if white_field_ratio > .1 and len(contours) >= min_fields:
                # sort contours based on error score and remove fields with an angle outside 50% of the mean
                contours = sorted(contours, key=operator.itemgetter(-1))
                angles = []
                for c in contours:
                    box = cv2.minAreaRect(c[0])
                    bp = cv2.boxPoints(box)

                    """If the minimum angled rectangle has its lowest x point with a smaller y value than the center 
                    of the box, the true angle for the box is that of the minAreaRect - 90 degrees."""
                    if bp[0][1] < box[0][1]:
                        angles.append(box[-1] - 90)
                    else:
                        angles.append(box[-1])
                angle_mean = np.mean(angles)

                _ = []  # curate the contours for angle outliers
                for c, a in zip(contours, angles):
                    deviation = a / angle_mean if angle_mean != 0 else a  # catch infinity division
                    if deviation < 1.5:
                        _.append((c, a))

                if len(_) >= min_fields:
                    if max_len == 0:  # determine the intensity span for which structures are starting to be found
                        trimmed_onset_intensity_span = intensity_span
                    max_len = len(_) if len(_) > max_len else max_len  # track the largest contour set

                    # collect errors and dimensions for contour sets from the best [min_fields]
                    widths, heights, angles, centres = [], [], [], []
                    for c, a in _:
                        (_cX, _cY), (_w, _h) = cv2.minAreaRect(c[0])[:-1]
                        widths.append(_w)
                        heights.append(_h)
                        angles.append(a)
                        centres.append((_cX, _cY))
                        contour_grid.append((c[0], (_cX, _cY), c[-1]))

                    mean_angle = np.mean(angles[:min_fields])
                    contour_sets.append(([len(contours), centres],
                                         [np.mean(widths[:min_fields]), np.mean(heights[:min_fields]), mean_angle]))

            # quit the loop early if the white ratio gets too high and 'enough' fields are found or intensity is maxed
            if white_field_ratio >= .7 and max_len > min_fields:
                stop = True
            elif onset_intensity > 254:
                if max_len > min_fields:
                    stop = True
                else:
                    onset_intensity = 0
                    intensity_span += mask_settings['IterativeFidelity']
                    supports.tprint('Thresholding has restarted with intensity span %d for file %s%s.'
                                    % (intensity_span, self.metadata['FileName'], self.unc))
            elif intensity_span > 100:
                raise RuntimeError(
                    'Threshold intensity span exceeded for %s%s. Try lowering the Fields or Fidelity option.'
                                   % (self.metadata['FileName'], self.unc))
            else:
                onset_intensity += mask_settings['IterativeFidelity']

        # define field dimensions along with all unique contours
        contour_grid = sorted(contour_grid, key=operator.itemgetter(-1))
        contour_sets = list(zip(*contour_sets))

        try:
            (width, height, angle) = np.mean(contour_sets[1], axis=0)
        except IndexError:
            raise RuntimeError('No contours found for %s%s. Try lowering the Fields option.' %
                               (self.metadata['FileName'], self.unc))

        if mask_settings['Align']:
            angle = mask_settings['Align']

        true_angle = angle  # save the true angle
        angle = angle if angle > 0 else 90 + angle  # convert the true angle to the opencv angle

        _ = [contour_grid[0][:-1]]  # save only the best-fitting unique contours starting with best field
        for c in contour_grid:  # add contour to grid if there does not exist a contour there
            add_cnt = True
            for (cont, cent) in _:
                if np.sqrt((cent[0] - c[1][0]) ** 2 + (cent[1] - c[1][1]) ** 2) < np.sqrt(width * height):
                    add_cnt = False
                    break

            if add_cnt:
                _.append(c[:-1])
        contour_grid = list(zip(*_))

        # transform all contours to a vector space with an arbitrary origin
        stop = False; i = 0
        while not stop:
            try:
                x0, y0 = contour_grid[1][i]
            except IndexError:
                raise RuntimeError(
                    f'Not enough fields found to determine mask for %s%s. Try changing the Fields option value.'
                    % (self.metadata['FileName'], self.unc))

            contour_vectors = []
            contour_magnitudes = []
            for (x, y) in contour_grid[1]:
                contour_vectors.append((x - x0, y - y0))
                contour_magnitudes.append(np.sqrt((x - x0) ** 2 + (y - y0) ** 2))

            # create functions to determine x-shift for row changes and y-shift for column changes, and x,y separations
            ideal_x_sep = self.mask_pars['FieldSpacingX'] + width
            ideal_y_sep = self.mask_pars['FieldSpacingY'] + height
            column0_shifts, row0_shifts, column0_mags, row0_mags, x_sep, y_sep = [], [], [], [], [], []
            for v, m in zip(contour_vectors, contour_magnitudes):
                vxp, vxm = v[0] + m, v[0] - m
                vyp, vym = v[1] + m, v[1] - m
                if -2 < vxm < 2 or -2 < vxp < 2:
                    _id = int(np.round(v[0] / ideal_y_sep, 0))
                    row0_shifts.append((_id, v[1]))
                    y_sep.append((_id, v[0]))
                if -2 < vym < 2 or -2 < vyp < 2:
                    _id = int(np.round(v[1] / ideal_x_sep, 0))
                    column0_shifts.append((_id, v[0]))
                    x_sep.append((_id, v[1]))

            row_x_shift_popt = self._attempt_fit(column0_shifts)
            column_y_shift_popt = self._attempt_fit(row0_shifts)
            y_sep_popt = self._attempt_fit(y_sep)
            x_sep_popt = self._attempt_fit(x_sep)

            if any(i is None for i in (row_x_shift_popt, column_y_shift_popt, y_sep_popt, x_sep_popt)):
                i += 1
            elif i > len(contour_grid[1]):
                raise RuntimeError(
                    'Not enough contours found for %s%s. Try lowering the Fields option.' %
                    (self.metadata['FileName'], self.unc))
            else:
                stop = True

        def point_to_space(_r, _c, npa=False) -> np.ndarray | tuple:  # create transformer function
            vector_x = linear_regression(_c, *x_sep_popt) + linear_regression(_r, *row_x_shift_popt)
            vector_y = linear_regression(_r, *y_sep_popt) + linear_regression(_c, *column_y_shift_popt)
            if npa is True:  # return coordinates as a numpy array if prompted
                return np.array([vector_x, vector_y])
            else:
                return vector_x, vector_y

        # transform all existing points to the point grid
        point_grid = {}
        for (x, y) in contour_grid[1]:
            x -= x0; y -= y0  # adjust points to fit with vectors
            axs, bxs = row_x_shift_popt
            ays, bys = column_y_shift_popt
            ax, bx = x_sep_popt
            ay, by = y_sep_popt

            # solved convolutions x = ax*c+bx + axs*r+bxs and y = ay*r+by + ays*c+bys
            column = (x - bx - axs * ((y + bys + by) / ay) - bxs) / (ax - (ays / ay))
            row = (y - ays * column + bys - by) / ay

            point_grid[(int(np.round(row, 0)), int(np.round(column, 0)))] = (x, y)

        min_r, min_c = np.min(list(point_grid.keys()), axis=0)
        max_r, max_c = np.max(list(point_grid.keys()), axis=0)
        _rows, _columns = max_r - min_r + 1, max_c - min_c + 1

        # check for missing bounderies and attempt to add appropriately
        if _columns < self.mask_pars['Columns']:
            c_diff = self.mask_pars['Columns'] - _columns
            for i in range(1, c_diff + 1):  # generate a sample field at min(c) - 1 and check for boundary cross
                _x, _ = point_to_space(min_r, min_c - i)
                if _x + x0 - ideal_x_sep / 2 < 0:  # check proposed field for boundary cross in x-direction
                    max_c += 1
                else:
                    min_c -= 1
        if _rows < self.mask_pars['Rows']:
            r_diff = self.mask_pars['Rows'] - _rows
            for i in range(1, r_diff + 1):  # generate a sample field at min(c) - 1 and check for boundary cross
                _, _y = point_to_space(min_r - i, min_c)
                if _y + y0 - ideal_y_sep / 2 < 0:  # check proposed field for boundary cross in y-direction
                    max_r += 1
                else:
                    min_r -= 1

        # construct a grid in the image space according to the adjusted point grid limits
        index_mask = {}
        real_indices = []
        for r in range(min_r, max_r + 1, 1):
            for c in range(min_c, max_c + 1, 1):
                vector_x, vector_y = point_to_space(r, c)
                if mask_settings['MaskingMethod'] == 'Hybrid':
                    if (r, c) in point_grid:
                        vector_x, vector_y = point_grid[(r, c)]
                        real_indices.append((r - min_r, c - min_c))
                index_mask[(r - min_r, c - min_c)] = np.int64(cv2.boxPoints((
                    (vector_x + x0, vector_y + y0), (width, height), angle)))  # save contours and reset origin to zero

        if mask_settings['MaskShift'] == 'Auto':
            _coords = np.vstack(list(index_mask.values()))  # flatten contour array to contour coordinates
            mask_cX, mask_cY = cv2.minAreaRect(_coords)[0]  # fit field mask to contour
            img_h, img_w = self.img.shape[:2]; img_cX, img_cY = img_w // 2, img_h // 2  # get image parameters

            # determine distance between image center and mask center for direction x and y
            x_dist = img_cX - mask_cX; y_dist = img_cY - mask_cY
            ideal_x_dist = point_to_space(0, 1, npa=True) - point_to_space(0, 0, npa=True)
            ideal_y_dist = point_to_space(1, 0, npa=True) - point_to_space(0, 0, npa=True)

            index_mask = self.grid_shift(index_mask, point_to_space, x=round(x_dist / ideal_x_dist[0]),
                                         y=round(y_dist / ideal_y_dist[1]))

        # write control image for manual evaluation
        _elementary_masking_path = rf'{self.dirs["OutputFolder"]}\_misc\elementary_masking'
        if not os.path.exists(_elementary_masking_path):
            try:  # catch overlapping file path creation from simultaneous processes
                os.makedirs(_elementary_masking_path)
            except FileExistsError:
                pass

        _img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        font_scale = 3 / 8700 * _img.shape[1]
        cv2.drawContours(_img, list(index_mask.values()), -1, (0, 255, 255), 10)

        # non-angular-corrected surrounding rows and columns from arbitrary origin
        cv2.putText(_img, 'p0', (int(x0), int(y0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 10)
        cv2.putText(_img, 'c+1', (int(x0 + ideal_x_sep), int(y0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 10)
        cv2.putText(_img, 'c-1', (int(x0 - ideal_x_sep), int(y0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 10)
        cv2.putText(_img, 'r+1', (int(x0), int(y0 + ideal_y_sep)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 10)
        cv2.putText(_img, 'r-1', (int(x0), int(y0 - ideal_y_sep)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 10)

        for center, vector, mag, pg in zip(contour_grid[1], contour_vectors, contour_magnitudes, point_grid.keys()):
            cv2.putText(_img, str(np.int64(mag)), (int(center[0] - 200), int(center[1] + 150)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 255), 10)
            cv2.putText(_img, str(np.int64(vector)), (int(center[0] - 380), int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 123, 255), 10)
            cv2.putText(_img, str(pg), (int(center[0] - 200), int(center[1] - 150)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 10)
        for k, v in index_mask.items():
            x, y = contour_center(v)
            cv2.putText(_img, str(k), (int(x - 200), int(y + 300)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 10)
        cv2.drawContours(_img, [markSquare(c) for c in contour_grid[0]], -1, (0, 123, 255), 25)
        cv2.putText(_img, 'Yellow: Calculated fields with absolute index positions', (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 10)
        cv2.putText(_img, 'Orange: Found fields with relative real positions', (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 123, 255), 10)
        cv2.putText(_img, 'Red: Found fields with relative index positions and origin-field distances', (60, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 10)
        _img = criterion_resize(_img)  # downscale to MaxSize pixels
        cv2.imwrite(r'{}\{}.png'.format(_elementary_masking_path, self.metadata['FileName']), _img)

        c_area = width * height
        c_ratio = width / height
        norm_rel_area = abs(c_area / self.mask_pars['FieldArea'] - 1)
        norm_ratio = abs(c_ratio - 1)

        # correct for input rescale
        width = width * self.scale_r ** -1; height = height * self.scale_r ** -1
        index_mask = {k: np.int64(v * self.scale_r ** -1) for k, v in index_mask.items()}

        # write determined parameters to the settings file
        update_dict = {'PreprocessedParameters': {self.channels.file_name: {
            'FieldParameters': {
                'ScaleBar': self.metadata['ImageData']['ScaleBarRMS'] ** -1,  # in units: real / pix
                'EffectiveScaleBar': self.scale ** -1,
                'MinFields': min_fields,
                'Align': true_angle,
                'Width': width,
                'Height': height,
                'MaskingMethod': mask_settings['MaskingMethod'],
                'MaskShift': mask_settings['MaskShift'],
                'IterativeFidelity': mask_settings['IterativeFidelity'],
                'OnsetIntensitySpan': trimmed_onset_intensity_span
            },
            'QualityParameters': {
                'Level': norm_rel_area * norm_ratio * 1e5,  # this is an arbitrary scalar (lower is better)
                'AreaError': norm_rel_area * 100,  # conversion to %
                'RatioError': norm_ratio * 100  # conversion to %
            },
            'MiscParameters': {
                'UsedRealFields': len(real_indices),
                'UsedComputedFields': len(index_mask.values()) - len(real_indices),
            }
        }}}

        self.draw_parameters = update_dict

        self.draw_parameters['PreprocessedParameters'][self.channels.file_name]['DrawParameters'] = {
            'RealFieldIndices': real_indices,
        }

        if kwargs['return_settings'] is False:
            return index_mask, update_dict
        else:
            settings = update_dict['PreprocessedParameters'][self.channels.file_name]
            return index_mask, settings

    def grid_shift(self, mask, cfunc, x=0, y=0) -> dict:
        """Shifts the index mask dict according to the specified shift and the conversion function.
        :param mask: the index mask dict.
        :param cfunc: the point-to-vector function/conversion function for the grid positions to real coordinates."""

        _ = {}; x_shift = 0; y_shift = 0
        if x != 0:
            x_shift = cfunc(0, x, npa=True) - cfunc(0, 0, npa=True)
            supports.tprint('Shifting mask for %s%s with %d columns.' % (self.metadata['FileName'], self.unc, x))
        if y != 0:
            y_shift = cfunc(y, 0, npa=True) - cfunc(0, 0, npa=True)
            supports.tprint('Shifting mask for %s%s with %d rows.' % (self.metadata['FileName'], self.unc, y))

        for k, v in mask.items():
            _[k] = np.int64(v + x_shift + y_shift)
        return _

    @staticmethod
    def pack_best_field(cnt):
        if len(cnt) > 1:
            contour = sorted(cnt, key=operator.itemgetter(-1))[0]
        else:
            contour = cnt[0]

        (_cX, _cY), (_w, _h), _a = cv2.minAreaRect(contour[0])

        # determine the true angle for the field
        bp = cv2.boxPoints(((_cX, _cY), (_w, _h), _a))
        if bp[0][1] < _cY:
            _a -= 90
        field = (contour[0], _cX, _cY, _w, _h, _a, contour[-1])
        return field

    def _single_field_iterator(self, error):
        _if = self.__settings['IndividualPreprocessSettings'][self.channels.file_name]['IterativeFidelity']
        method_prefix = 'Otsu'
        thresh_img = cv2.threshold(self.blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        stop = False; _k = 3 - _if; field = None
        while not stop:  # attempt masking on otsu threshold
            _k += _if
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_k, _k))  # define morphology kernel
            morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, morph_kernel,
                                         iterations=2)  # apply morphology to threshold
            cont = cv2.findContours(morph_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]  # find contours
            cnt = self.__contour_filter(cont, pae=error, pre=error)

            if cnt:
                stop = True
                field = self.pack_best_field(cnt)
                self.TP = _k

            elif _k > 100:
                stop = True
                supports.tprint(
                    'Otsu threshold failed for %s%s; attempting adaptive thresholding.' %
                    (self.metadata['FileName'], self.unc))

        if field is None:
            method_prefix = 'Adaptive'
            stop = False; _b, _c = 11 - _if, 2
            while not stop:  # attempt masking with adaptive thresholding
                _b += _if; _c += 0
                thresh_img = cv2.adaptiveThreshold(self.blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, _b, _c)
                cont = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]  # find contours
                cnt = self.__contour_filter(cont, pae=error, pre=error)

                if cnt:
                    stop = True
                    field = self.pack_best_field(cnt)
                    self.TP = _b

                elif _b > 100:
                    stop = True
                    supports.tprint(
                        'Adaptive threshold failed for %s%s; attempting iterative thresholding.' % (self.metadata[
                            'FileName'], self.unc))

        if field is None:
            method_prefix = 'Iterative'
            stop = False; _oi = -_if
            while not stop:  # attempt masking with adaptive thresholding
                _oi += _if
                cont, _ = self.__contour_detect(_oi, 200)  # detect contours
                cnt = self.__contour_filter(cont, pae=error, pre=error)

                if cnt:
                    stop = True
                    field = self.pack_best_field(cnt)
                    self.TP = _oi

                elif _oi > 253:
                    stop = True
                    supports.tprint(
                        'Iterative threshold failed for %s%s; increasing error allowance by .01.' % (self.metadata[
                            'FileName'], self.unc))
        return field, method_prefix

    def single_field_identification(self):
        mask_settings = self.__settings['IndividualPreprocessSettings'][self.channels.file_name]
        mask_settings['MaskingMethod'] = mask_settings['MaskingMethod'].split(' ')[-1]  # ensure correct choice load

        # find the field contour
        field = None; stop = False; error = .01; method_prefix = None
        while not stop:
            field, method_prefix = self._single_field_iterator(error)
            if field is not None: stop = True
            if error > .1: raise RuntimeError('Mask field could not be identified for %s%s, as error limit was reached.'
                                               % (self.metadata['FileName'], self.unc))
            error += .01

        # define field dimensions along with all unique contours
        cX, cY, width, height, angle = field[1:-1]
        if mask_settings['MaskingMethod'] == 'Fixed':
            width = self.mask_pars['FieldWidth']; height = self.mask_pars['FieldHeight']

        if mask_settings['Align']:
            angle = mask_settings['Align']

        true_angle = angle  # save the true angle
        angle = angle if angle > 0 else 90 + angle  # convert the true angle to the opencv angle

        contour = np.int64(cv2.boxPoints(((cX, cY), (width, height), angle)))  # pack the contour

        c_area = width * height
        c_ratio = width / height
        norm_rel_area = abs(c_area / self.mask_pars['FieldArea'] - 1)
        norm_ratio = abs(c_ratio - 1)

        # correct for input rescale
        width = width * self.scale_r ** -1; height = height * self.scale_r ** -1

        # write determined parameters to the settings file
        update_dict = {'PreprocessedParameters': {self.channels.file_name: {
            'FieldParameters': {
                'ScaleBar': self.metadata['ImageData']['ScaleBarRMS'] ** -1,  # in units: real / pix
                'EffectiveScaleBar': self.scale ** -1,
                'Align': true_angle,
                'Width': width,
                'Height': height,
                'MaskingMethod': ' '.join([method_prefix, mask_settings['MaskingMethod']]),
                'TP': self.TP,
                'IterativeFidelity': mask_settings['IterativeFidelity'],
            },
            'QualityParameters': {
                'Level': norm_rel_area * norm_ratio * 1e5,  # this is an arbitrary scalar (lower is better)
                'AreaError': norm_rel_area * 100,  # conversion to %
                'RatioError': norm_ratio * 100  # conversion to %
            },
        }}}

        self.draw_parameters = update_dict

        return [contour], update_dict


class ProcessingHandler:
    def __init__(self):
        self.__settings = None
        self.dirs = supports.setting_cache()['DirectorySettings']

    def process(self, file):
        file_repeat, file_name = file.split(':')[1:]; file_repeat = f'Repeat:{file_repeat}'
        self.dirs = self.dirs[file_repeat]
        self.__settings = supports.json_dict_push(rf'{self.dirs["OutputFolder"]}\Settings.json', behavior='read')
        _cps = self.__settings['CollectionProcessSettings']  # set collection settings

        scale_bar = self.__settings['PreprocessedParameters'][file_name]['FieldParameters']['ScaleBar']
        _fr = self.__settings['CollectionProcessSettings']['ForceWorkResolution']
        if _fr == '':
            scale_bar, scale_ratio = define_scalars(scale_bar, 1.7, 2.5, True)
        elif _fr > scale_bar ** -1:
            scale_bar, scale_ratio = define_scalars(scale_bar, None, _fr, True)
        else:
            scale_bar = scale_bar; scale_ratio = 1

        cell_params = SampleParameters((scale_bar / scale_ratio) ** -1, configure=_cps['CellType'])
        mask_data = supports.json_dict_push(r'{}\{}\maskdata.json'.format(self.dirs['OutputFolder'], file_name),
                                            behavior='read')

        # fetch previous data if possible
        json_path = rf'{self.dirs["OutputFolder"]}\{file_name}\data.json'
        if os.path.exists(json_path):
            prev_data = supports.json_dict_push(json_path, behavior='read')
        else:
            prev_data = {'NucleiFieldData': {}, 'MorphologyFieldData': {}, 'CellData': {}}

        nuclei_data_dfs, cell_data_dfs, morph_data_dfs = [], [], []  # placeholder for dataframes
        write_json = False
        cell_count = None
        if _cps['NucleiChannel'] is not None:
            write_json = True
            field_path = r'{}\{}\{}'.format(self.dirs['OutputFolder'], file_name, _cps['NucleiChannel'])
            fields = [f.removeprefix('_').removesuffix('.tiff') for f in os.listdir(field_path) if f.endswith('.tiff')]
            _ips = self.__settings['IndividualProcessSettings'][file_name]  # set individual settings

            # check output directory validity and ensure that it is empty
            directory_checker(rf'{field_path} (Processed)', clean=True)
            sc_path = r'{}\{}\Seeding Scores'.format(self.dirs['OutputFolder'], file_name)
            directory_checker(sc_path, clean=True)

            for field in fields:
                img = read_channel(filename=rf'{field_path}\_{field}.tiff', scalar=scale_ratio)
                h, w = img.shape[0:2]
                _edge = int(_cps['EdgeProximity'] * scale_bar / scale_ratio)  # save edge filter in pixels

                # draw field contour and permitted edge on image
                rect = mask_data['ContourMask'][field]['MinAreaRect']
                cv2.drawContours(img, [np.int64(cv2.boxPoints(rect) * scale_ratio)], -1, (0, 0, 255), 1)
                edge_box = [rect[0], [rect[1][0] - _edge * 2, rect[1][1] - _edge * 2], rect[2]]
                edge_field = np.int64(cv2.boxPoints(edge_box) * scale_ratio)
                cv2.drawContours(img, [edge_field], -1, (0, 255, 0), 1)

                _binary = None
                if _ips['CountingMethod'] == 'Hough':
                    _count, _binary = hough_circles_cell_counting(img, cell_params)
                elif _ips['CountingMethod'] == 'Classic':
                    _count, _binary = threshold_cell_counting(img, cell_params)
                elif _ips['CountingMethod'] == 'CCA':
                    _cd = analysis.CellDetector(img)
                    _cd.connected_component_counting(_ips['SliceSize'])
                    _count = _cd.points
                elif _ips['CountingMethod'] == 'Black-Out':
                    _cd = analysis.CellDetector(img)
                    _cd.image_iterator(_ips['SliceSize'], expand_cycles=_ips['Cycles'], expand_onset=_ips['Filter'],
                                       expand_step_size=_ips['Step'])
                    _count = _cd.points
                else:
                    raise ValueError('Counting Method {!r} is invalid.'.format(_ips['CountingMethod']))

                # filter away cells if their center is within the outer edges of the field
                valid_cells = []  # placeholder list
                for c in _count:
                    if _ips['CountingMethod'] == 'Black-Out':
                        (cX, cY), (axL1, axL2), angle = cv2.fitEllipse(c)
                    elif _ips['CountingMethod'] == 'CCA':
                        cX, cY = c
                    else:
                        cX, cY, r = c

                    if cv2.pointPolygonTest(edge_field, (cX, cY), False) > 0:
                        if _ips['CountingMethod'] == 'Hough':
                            cv2.circle(img, (int(cX), int(cY)), int(r), (0, 255, 0), 1)
                        elif _ips['CountingMethod'] == 'Classic':
                            cv2.circle(img, (int(cX), int(cY)), int(r), (0, 0, 255), 1)
                        elif _ips['CountingMethod'] == 'CCA':
                            cv2.circle(img, (int(cX), int(cY)), int(5), (255, 0, 255), 1)
                        elif _ips['CountingMethod'] == 'Black-Out':
                            cv2.ellipse(img, (int(cX), int(cY)), (int(axL1), int(axL2)), angle, 0, 360, (0, 255, 255), 1)

                        valid_cells.append((cX, cY))  # append valid cell to list

                # count the number of valid cells and write it on the image
                cell_count = len(valid_cells)
                cell_density = int(cell_count / (cv2.contourArea(edge_field) / (scale_bar * 1e4) ** 2))

                # estimate area-based cell count
                area_cell_count = None
                if _binary is not None:
                    mean_nuclei_area = np.mean(np.square(list(zip(*_count))[-1]) * np.pi)
                    cell_coverage = np.where(_binary[_edge:w - _edge, _edge:h - _edge].flatten() == 255)[0].size
                    area_cell_count = cell_coverage // mean_nuclei_area

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cell_image = np.zeros_like(gray)
                seeding_mask = np.zeros_like(gray)
                cv2.drawContours(seeding_mask, [edge_field], -1, (255, 255, 255), -1)

                # estimate nearest neighbour distances for found valid cells and write the average on image
                ideal_nnd = np.sqrt(1 / self.__settings['CollectionProcessSettings']['SeedingDensity'] * 1e4 ** 2)
                if cell_count > 1:  # analysis can only proceed with 2 or more cells, otherwise NNDs will be invalid
                    nearest_cell_distances = nearest_neighbors(valid_cells)
                    for i in range(cell_count):
                        cX, cY = valid_cells[i]
                        r = nearest_cell_distances[i]

                        if _ips['CountingMethod'] == 'Hough':
                            cv2.circle(img, (int(cX), int(cY)), int(r), (0, 127, 0), 1)
                        elif _ips['CountingMethod'] == 'Classic':
                            cv2.circle(img, (int(cX), int(cY)), int(r), (0, 0, 127), 1)
                        elif _ips['CountingMethod'] == 'Black-Out':
                            cv2.circle(img, (int(cX), int(cY)), int(r), (0, 127, 127), 1)
                        elif _ips['CountingMethod'] == 'CCA':
                            cv2.circle(img, (int(cX), int(cY)), int(r), (127, 0, 127), 1)

                        # add NND area to cell_image
                        cv2.circle(cell_image, (int(cX), int(cY)), int(ideal_nnd * scale_bar), (255, 255, 255), -1)
                else:
                    supports.tprint(f'No cells could be identified for {file_name} field {field}.')
                    nearest_cell_distances = []

                # mask NND cell_image and score seeding
                masked_img = cv2.bitwise_and(cell_image, cell_image, mask=seeding_mask)
                cell_area = masked_img[masked_img == 255].size  # count white pixels
                seeding_score = cell_area / cv2.contourArea(edge_field)
                seeding_score = seeding_score if seeding_score <= 1 else 1  # prevent overflow values
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(masked_img, [np.int64(cv2.boxPoints(rect))], -1, (0, 0, 255), 1)
                cv2.drawContours(masked_img, [edge_field], -1, (0, 255, 0), 1)

                # draw data on the seeding score image and write
                text_overlay = Image.fromarray(masked_img)
                text_font = ImageFont.truetype(rf'{supports.__cwd__}\__misc__\Arial.ttf', 60 / 3850 * w)
                text_text = (f'Maximum-Ideal NND: {ideal_nnd:.2f} µm\n'
                             f'Seeding Score: {seeding_score * 100:.2f}%')
                text_draw = ImageDraw.Draw(text_overlay)
                if seeding_score > .9:
                    text_draw.text((w // 50, h // 50 + 12), text_text, font=text_font, fill=(0, 200, 0))
                elif .75 < seeding_score <= .9:
                    text_draw.text((w // 50, h // 50 + 12), text_text, font=text_font, fill=(0, 200, 200))
                else:
                    text_draw.text((w // 50, h // 50 + 12), text_text, font=text_font, fill=(0, 0, 200))
                cv2.imwrite(r'{}\_{}_ss.tiff'.format(sc_path, field), np.array(text_overlay))

                # Note that the std is the sample std not the population std
                _mean_distribution = np.mean(nearest_cell_distances) / scale_bar
                _distribution_std = np.std(nearest_cell_distances, ddof=1) / scale_bar

                # put data text on nuclei image
                text_overlay = Image.fromarray(img)
                text_text = (f'Cells: {cell_count}     Density: {cell_density} c/cm²\n'
                             f'Distribution: {_mean_distribution:.2f} ± {_distribution_std:.2f}')
                text_draw = ImageDraw.Draw(text_overlay)
                text_draw.text((w // 50, h // 50 + 12), text_text, font=text_font, fill=(255, 255, 255))
                img = np.array(text_overlay)

                _nd_df = pd.DataFrame([[cell_count, area_cell_count, _mean_distribution, _distribution_std,
                                        cell_density, seeding_score]],
                                      index=[field],
                                      columns=['Cell Count', 'Area Cell Count', 'Mean Cell Distribution (µm)',
                                               'Std Cell Distribution (µm)', 'Cell Density', 'Seeding Score'])
                nuclei_data_dfs.append(_nd_df)

                _cd_df = pd.DataFrame([[i / scale_bar for i in nearest_cell_distances], nearest_cell_distances],
                                      index=['NN Distances (µm)', 'NN Distances (pix)'])
                cell_data_dfs.append(_cd_df)

                # write output image
                cv2.imwrite(r'{} (Processed)\_{}_{}_processed.tiff'.format(
                    field_path, field, _cps['NucleiChannel']), img)

        if _cps['MorphologyChannel'] is not None:
            write_json = True
            field_path = r'{}\{}\{}'.format(self.dirs['OutputFolder'], file_name, _cps['MorphologyChannel'])
            fields = [f.removeprefix('_').removesuffix('.tiff') for f in os.listdir(field_path) if f.endswith('.tiff')]

            # check output directory validity and ensure that it is empty
            directory_checker(rf'{field_path} (Processed)', clean=True)

            for field in fields:
                img = read_channel(filename=rf'{field_path}\_{field}.tiff', scalar=scale_ratio)
                h, w = img.shape[0:2]
                _edge = int(_cps['EdgeProximity'] * scale_bar)  # save edge filter in pixels

                # draw field contour and determine permitted edge on image
                rect = mask_data['ContourMask'][field]['MinAreaRect']
                cv2.drawContours(img, [np.int64(cv2.boxPoints(rect))], -1, (0, 0, 255), 1)
                edge_box = [rect[0], [rect[1][0] - _edge * 2, rect[1][1] - _edge * 2], rect[2]]
                edge_field = np.int64(cv2.boxPoints(edge_box))

                # mask input field image with the permitted field edge
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [edge_field], -1, (255, 255, 255), -1)
                masked_img = cv2.bitwise_and(gray, gray, mask=mask)

                if _cps['ThresholdIntensity'] > 0:
                    thresh = cv2.threshold(masked_img, _cps['ThresholdIntensity'], 255, cv2.THRESH_BINARY)[1]
                else:
                    thresh = cv2.threshold(masked_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                cell_area = thresh[thresh == 255].size
                cell_coverage = cell_area / cv2.contourArea(edge_field)
                text_text = f'Cell Coverage: {cell_coverage * 100:.2f}%\n'

                cell_spreading = None
                if cell_count in (None, 0) and 'Cell Count' in prev_data['NucleiFieldData']:
                    cell_count = prev_data['NucleiFieldData']['Cell Count'][field]

                if cell_count:
                    cell_spreading = cell_area / scale_bar ** 2 / cell_count
                    text_text += f'Average Cell Area: {cell_spreading:.2f} µm²/c'

                # put data text on the image
                text_overlay = Image.fromarray(img)
                text_font = ImageFont.truetype(rf'{supports.__cwd__}\__misc__\Arial.ttf', 60 / 3850 * w)
                text_draw = ImageDraw.Draw(text_overlay)
                text_draw.text((w // 50, h // 50 + 12), text_text, font=text_font, fill=(255, 255, 255))
                img = np.array(text_overlay)

                col_thresh = supports.colorize(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGBA), '#16f1f5', 'white')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA); img = cv2.add(img, col_thresh)  # add cell area to img
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # revert to old color type
                cv2.drawContours(img, [edge_field], -1, (0, 255, 0), 1)  # draw permitted edge

                _md_df = pd.DataFrame([[cell_coverage, cell_spreading]],
                                      index=[field],
                                      columns=['Cell Coverage', 'Average Cell Area (µm^2/c)'])
                morph_data_dfs.append(_md_df)

                # write output image
                cv2.imwrite(r'{} (Processed)\_{}_{}_processed.tiff'.format(
                    field_path, field, _cps['MorphologyChannel']), img)

        # collect and store the determined variables in a json file at the data position
        if write_json:
            nucleiDF = pd.concat(nuclei_data_dfs, axis=0) if nuclei_data_dfs else pd.DataFrame()
            morphDF = pd.concat(morph_data_dfs, axis=0) if morph_data_dfs else pd.DataFrame()
            cellDF = pd.concat(cell_data_dfs, axis=1, ignore_index=True).T if cell_data_dfs else pd.DataFrame()

            data_dict = {
                'NucleiFieldData': nucleiDF.to_dict() if not nucleiDF.empty else prev_data['NucleiFieldData'],
                'MorphologyFieldData': morphDF.to_dict() if not morphDF.empty else prev_data['MorphologyFieldData'],
                'CellData': cellDF.to_dict() if not cellDF.empty else prev_data['CellData']
            }

            supports.json_dict_push(json_path, params=data_dict, behavior='mutate')


class AnalysisHandler:
    def __init__(self):
        self.cache = _ = supports.setting_cache()
        self._dirs = _['DirectorySettings']; self.rdata = _['RepeatData']
        self._dom_out = self._dirs[self.cache['DominantRepeat']]['OutputFolder']
        self.__settings = None
        self.data = None; self.collection_data = None

    def __getitem__(self, item):
        return self.__settings[item]

    def analyze(self):
        repeats = [i for i in self._dirs.keys() if i != 'CommonOutputFolder']  # fetch repeats
        for r in repeats:
            self._analyze(r)

        if 'CommonOutputFolder' in self._dirs:
            supports.tprint(f'Initialized Collection Analysis')
            self.collect_field_data()
            _da = analysis.DataAnalysis(self.collection_data, None)
            if self['CollectionAnalysisSettings']['AnalyzeData']['NucleiAnalysis']:
                supports.tprint(f'Started Collection Nuclei Analysis', branch=1)
                if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
                    _da.multi_field_nuclei_analysis()
                elif self['CollectionPreprocessSettings']['SampleType'] in ('Single-Field', 'Zero-Field'):
                    _da.single_field_nuclei_analysis()
                supports.tprint(f'Finished Collection Nuclei Analysis', branch=1)
            if self['CollectionAnalysisSettings']['AnalyzeData']['StatisticalAnalysis']:
                supports.tprint(f'Started Collection Statistical Analysis', branch=1)
                _da.statistical_analysis()
                supports.tprint(f'Finished Collection Statistical Analysis', branch=1)
            supports.tprint(f'Finalized Collection Analysis')

    def _analyze(self, repeat):
        rname = self.rdata[repeat]['Name']; rname = f' {rname}' if rname is not None else ''

        if rname: supports.tprint(f'Initialized Analysis of{rname}');
        else: supports.tprint(f'Initialized Analysis')

        self.__out_dir = self._dirs[repeat]['OutputFolder']
        self.__settings = supports.json_dict_push(rf'{self.__out_dir}\Settings.json', behavior='read')  # load settings

        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            self.normalize_multi_field_data()
        elif self['CollectionPreprocessSettings']['SampleType'] in ('Single-Field', 'Zero-Field'):
            self.normalize_single_field_data()

        _da = analysis.DataAnalysis(self.data, repeat)
        _da.group_data()
        if self['CollectionAnalysisSettings']['AnalyzeData']['NucleiAnalysis']:
            supports.tprint(f'Started{rname} Nuclei Analysis', branch=1)
            if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
                _da.multi_field_nuclei_analysis()
            elif self['CollectionPreprocessSettings']['SampleType'] in ('Single-Field', 'Zero-Field'):
                _da.single_field_nuclei_analysis()
            supports.tprint(f'Finished{rname} Nuclei Analysis', branch=1)
        if self['CollectionAnalysisSettings']['AnalyzeData']['NearestNeighbourHistogram']:
            supports.tprint(f'Started{rname} Nearest Neighbour Evaluation', branch=1)
            _da.nearest_neighbour_analysis()
            supports.tprint(f'Finished{rname} Nearest Neighbour Evaluation', branch=1)
        if self['CollectionAnalysisSettings']['AnalyzeData']['StatisticalAnalysis']:
            supports.tprint(f'Started{rname} Statistical Analysis', branch=1)
            _da.statistical_analysis()
            supports.tprint(f'Finished{rname} Statistical Analysis', branch=1)

        if rname: supports.tprint(f'Finalized Analysis of{rname}');
        else: supports.tprint('Finalized Analysis');

    def collect_field_data(self):
        groups = self.__settings['CollectionAnalysisSettings']['DataGroups']
        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            factors = ['Cell Count', 'Normalized Cell Count', 'Area Cell Count', 'Mean Cell Distribution (µm)',
                       'Seeding Score', 'SC Normalized Cell Count',
                       'Average Cell Area (µm^2/c)', 'Normalized Cell Coverage',
                       'SC Normalized Cell Coverage']
        else:
            factors = ['Cell Count', 'Normalized Cell Count', 'Area Cell Count', 'Mean Cell Distribution (µm)',
                       'Cell Density', 'Seeding Score', 'SC Cell Count', 'SC Normalized Cell Count', 'SC Cell Density',
                       'Average Cell Area (µm^2/c)', 'Cell Coverage', 'Normalized Cell Coverage', 'SC Cell Coverage',
                       'SC Normalized Cell Coverage']

        factors_std = [f'{f} Std' for f in factors]

        mean_dict = {f: {g: [] for g in groups} for f in factors + factors_std}
        collection_dict = {f: {} for f in factors}

        group_dict = {g: [] for g in groups}
        set_dict = {g: {f: {} for f in factors} for g in groups}

        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            control_keys = ('ControlCount', 'ControlArea', 'ControlDensity')
            control_pars = {p: {g: [] for g in groups} for p in control_keys}

        # collect data in a single dict
        for repeat in self.rdata.keys():
            rn = repeat.split(':')[-1]
            try: out = self._dirs[repeat]['OutputFolder'];
            except KeyError: continue;  # catch DominantRepeat slip
            anres = supports.json_dict_push(rf'{out}\AnalysisResults.json', behavior='read')
            adata = anres['GroupAnalysis']

            for g in groups:
                group_dict[g].append(f'{g}_{rn}')
                if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
                    try:
                        for ck in control_keys:
                            control_pars[ck][g].append(anres['ControlParameters'][ck][g])
                    except KeyError: pass;
                for f in factors:
                    try:  # catch differences between sample types
                        mean_dict[f][g].append(adata[f][g])
                        collection_dict[f][f'{g}_{rn}'] = adata[f][g]
                        set_dict[g][f][f'{g}_{rn}'] = adata[f][g]
                    except KeyError: pass;

        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            for g in groups:
                for ck in control_keys:
                    control_pars[ck][g] = np.mean(control_pars[ck][g])

        # determine averages and stds
        for f in factors:
            for g in groups:
                try:
                    if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
                        try:
                            _std = dict(zip(mean_dict[f][g][0].keys(), np.std([list(i.values()) for i in mean_dict[f][g]], axis=0)))
                            _mean = dict(zip(mean_dict[f][g][0].keys(), np.mean([list(i.values()) for i in mean_dict[f][g]], axis=0)))
                        except IndexError: continue  # catch missing values
                    else:
                        _std = np.std(mean_dict[f][g], ddof=1)
                        _mean = np.mean(mean_dict[f][g])

                    mean_dict[f'{f} Std'][g] = _std
                    mean_dict[f][g] = _mean
                except KeyError: pass  # catch differences between sample types

        self.collection_data = {'CollectionData': collection_dict, 'MeanData': mean_dict, 'DataGroups': group_dict,
                                'SetData': set_dict}
        if self['CollectionPreprocessSettings']['SampleType'] == 'Multi-Field':
            self.collection_data['ControlParameters'] = control_pars
        c_out = self._dirs['CommonOutputFolder']
        supports.json_dict_push(rf'{c_out}\AnalysisResults.json', self.collection_data, behavior='mutate')

    def normalize_multi_field_data(self):
        """Internal method that normalizes multi-field data."""

        push_sheets = ['Cell Count', 'Normalized Cell Count', 'Area Cell Count', 'Mean Cell Distribution (µm)',
                       'Std Cell Distribution (µm)', 'Cell Density', 'Seeding Score', 'SC Normalized Cell Count',
                       'NN Distances (µm)', 'NN Distances (pix)']  # define sheet names
        if self['CollectionAnalysisSettings']['MorphologyAnalysis'] is True:
            push_sheets += ['Average Cell Area (µm^2/c)', 'Cell Coverage', 'Normalized Cell Coverage',
                            'Normalized Cell Area', 'SC Normalized Cell Coverage']

        # load processed image data
        datas = []
        for file in self['IndividualProcessSettings'].keys():  # iterate through processed files
            json_data_dir = rf'{self.__out_dir}\{file}\data.json'  # set data directory
            file_name_columns = dict(zip(push_sheets, [file, ] * len(push_sheets)))  # set column names
            if os.path.isfile(json_data_dir):  # if stored data exists, read it and construct dataframes
                json_data = supports.json_dict_push(json_data_dir, behavior='read')
                nucleiDF = pd.DataFrame(json_data['NucleiFieldData']).rename(columns=file_name_columns)
                cell_data_df = pd.DataFrame(json_data['CellData']).rename(columns=file_name_columns)
                cell_counts = nucleiDF.iloc[:, 0:1]; seeding_score_df = nucleiDF.iloc[:, 5:6]
                norm_cell_count_df = cell_counts.copy()  # default normalized cell counts to the cell counts
                sc_norm_cell_count_df = cell_counts.copy()  # default normalized cell counts to the cell counts

                # normalize cell count data according to control fields
                control_fields = self['CollectionAnalysisSettings']['SampleTypeSettings']['ControlFields']
                control_counts = [cell_counts.loc[cf, file] for cf in control_fields]
                _mean, _std = np.mean(control_counts), np.std(control_counts, ddof=1)
                _ = {'AdditionalData': {'InternalControlCountMean': _mean, 'InternalControlCountStd': _std}}
                norm_cell_count_df[file] = norm_cell_count_df[file].div(_mean)
                sc_norm_cell_count_df[file] = norm_cell_count_df[file].div(seeding_score_df[file], axis=0).replace(np.inf, 0)

                # append dataframes to list
                push_dfs = [cell_counts, norm_cell_count_df, nucleiDF.iloc[:, 1:2], nucleiDF.iloc[:, 2:3],
                            nucleiDF.iloc[:, 3:4], nucleiDF.iloc[:, 4:5], seeding_score_df,
                            sc_norm_cell_count_df, cell_data_df.iloc[:, 0:1], cell_data_df.iloc[:, 1:2]]

                # add morphology data normalization if prompted
                if self['CollectionAnalysisSettings']['MorphologyAnalysis'] is True:
                    morphDF = pd.DataFrame(json_data['MorphologyFieldData']).rename(columns=file_name_columns)
                    cell_coverages = morphDF.iloc[:, 0:1]; norm_cell_coverage_df = cell_coverages.copy()
                    cell_areas = morphDF.iloc[:, 1:2]; norm_cell_area_df = cell_areas.copy()
                    sc_norm_cell_coverage_df = cell_coverages.copy()

                    # normalize cell coverage to control fields
                    control_coverages = [cell_coverages.loc[cf, file] for cf in control_fields]
                    _mean, _std = np.mean(control_coverages), np.std(control_coverages, ddof=1)
                    _['AdditionalData']['InternalControlCoverageMean'] = _mean
                    _['AdditionalData']['InternalControlCoverageStd'] = _std
                    norm_cell_coverage_df[file] = norm_cell_coverage_df[file].div(_mean)
                    sc_norm_cell_coverage_df[file] = norm_cell_coverage_df[file].div(seeding_score_df[file], axis=0).replace(np.inf, 0)

                    # normalize cell area to control fields
                    control_areas = [cell_areas.loc[cf, file] for cf in control_fields]
                    _mean, _std = np.mean(control_areas), np.std(control_areas, ddof=1)
                    _['AdditionalData']['InternalControlAreaMean'] = _mean
                    _['AdditionalData']['InternalControlAreaStd'] = _std
                    norm_cell_area_df[file] = norm_cell_area_df[file].div(_mean)

                    push_dfs += [cell_areas, cell_coverages, norm_cell_coverage_df, norm_cell_area_df,
                                 sc_norm_cell_coverage_df]

                supports.json_dict_push(json_data_dir, params=_, behavior='update')
                datas.append(push_dfs)

        data_sets = [pd.concat(i, axis=1) for i in zip(*datas)]  # transform data sets and merge
        self.data = dict(zip(push_sheets, data_sets))  # construct data dict for analysis

        if self['CollectionAnalysisSettings']['ExcelExport'] is True:
            excel_df_push(rf'{self.__out_dir}\Results.xlsx', dataframes=data_sets, sheets=push_sheets,
                          behavior='replace')  # export result dataframes Excel

    def normalize_single_field_data(self):
        """Internal method that normalizes single-field data."""

        sheets = ('Collection Data', 'NN Distances (µm)', 'NN Distances (pix)')  # define sheet names
        collection_data = ['Cell Count', 'Normalized Cell Count', 'Area Cell Count', 'Mean Cell Distribution (µm)',
                           'Std Cell Distribution (µm)', 'Cell Density', 'Seeding Score', 'SC Cell Count',
                           'SC Normalized Cell Count', 'SC Cell Density']
        _erg = self['CollectionAnalysisSettings']['SampleTypeSettings']['ExternalReferenceGroup']; _ergs = []
        if _erg != 'None':
            _ = [_erg]; mg_options = [self['CollectionAnalysisSettings']['DataGroups']]
            if self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup'] != 'None':

                options = supports.json_dict_push(
                    rf'{self._dom_out}\_misc\multi_group.json', behavior='read')[
                    self['CollectionAnalysisSettings']['SampleTypeSettings']['MultiGroup']]['MultiGroup']
                _ = []

                mg_options = []
                for elem in options.values():
                    temp = []
                    for k, v in elem.items():
                        temp.append(k)
                        if v == _erg:
                            _.append(k)
                    mg_options.append(temp)
            _ergs = _


        _entries = ['SC Cell Count', 'Normalized Cell Count', 'SC Normalized Cell Count', 'SC Cell Density']
        if self['CollectionAnalysisSettings']['MorphologyAnalysis'] is True:
            collection_data += ['Average Cell Area (µm^2/c)', 'Cell Coverage', 'Normalized Cell Coverage',
                                'SC Cell Coverage', 'SC Normalized Cell Coverage']
            _entries += ['SC Cell Coverage', 'Normalized Cell Coverage', 'SC Normalized Cell Coverage']

        # load processed image data
        datas = []; reference_group = {}
        for file, state in self['IndividualAnalysisSettings'].items():  # iterate through processed files
            json_data_dir = rf'{self.__out_dir}\{file}\data.json'  # set data directory

            if os.path.isfile(json_data_dir) and state['State'] is True:  # if stored data exists, read it and construct dataframes
                json_data = supports.json_dict_push(json_data_dir, behavior='read')

                _skip_file = False
                if (json_data['NucleiFieldData']['Seeding Score'][file] <
                        self['CollectionAnalysisSettings']['SampleTypeSettings']['SeedingScoreCriterion']):
                    _skip_file = True
                    supports.tprint(rf'Seeding score for {file} is below the criterion. Skipping analysis for file.')
                elif json_data['NucleiFieldData']['Cell Count'][file] < 2:
                    _skip_file = True
                    supports.tprint(rf'Nuclei count for {file} is below  2. Skipping analysis for file.')

                if _skip_file is True:
                    _ = {'IndividualAnalysisSettings': {file: {'State': False}}}
                    supports.json_dict_push(rf'{self.__out_dir}\Settings.json', params=_, behavior='update')
                    continue  # skip file if it falls below the seeding score criterion

                cd_DF = pd.DataFrame(columns=collection_data, index=[file])  # placeholder dataframe

                nnd_DFs = []
                for k in ('µm', 'pix'):
                    _ = json_data['CellData'][f'NN Distances ({k})']
                    nnd_DFs.append(pd.DataFrame(_.values(), index=_.keys(), columns=[file]).T)

                # add data entries
                for de in _entries:
                    for _end in ('Cell Count', 'Cell Density', 'Cell Coverage'):
                        if de.endswith(_end):
                            if _end == 'Cell Coverage':
                                json_data['MorphologyFieldData'][de] = json_data['MorphologyFieldData'][_end]
                            else:
                                json_data['NucleiFieldData'][de] = json_data['NucleiFieldData'][_end]

                for i in ('Morphology', 'Nuclei'):
                    if f'{i}FieldData' in json_data:
                        for k, v in json_data[f'{i}FieldData'].items():
                            cd_DF[k] = v
                _dg = self['IndividualAnalysisSettings'][file]['DataGroup']
                if _erg != 'None' and _dg in _ergs:
                    if _dg not in reference_group:
                        reference_group[_dg] = []

                    # note that at this point the SC values are not yet SC corrected, and so this is done simultaneously
                    _ = [cd_DF['Cell Count'].values[0],
                         cd_DF['SC Cell Count'].values[0] / cd_DF['Seeding Score'].values[0]]
                    if self['CollectionAnalysisSettings']['MorphologyAnalysis'] is True:
                        _ += [cd_DF['Cell Coverage'].values[0],
                              cd_DF['SC Cell Coverage'].values[0] / cd_DF['Seeding Score'].values[0]]
                    reference_group[_dg].append(_)
                datas.append((cd_DF, *nnd_DFs))

        data_sets = [pd.concat(i, axis=0) for i in zip(*datas)]  # transform data sets and merge
        for k in data_sets[0].keys():
            if k.startswith('SC'):  # perform seeding correction on data set
                data_sets[0][k] = data_sets[0][k].div(data_sets[0]['Seeding Score'], axis=0)
        if _erg != 'None':
            _means, _stds, _dict = [], [], {}
            for k, rg in reference_group.items():
                _m = np.mean(rg, axis=0); _s = np.std(rg, ddof=1, axis=0)
                _dict[k] = (_m, _s); _means.append(_m); _stds.append(_s)

            _ = {'AdditionalData': {  # add additional data to file
                'ExternalReferenceGroup': list(reference_group.keys()),
                'ExternalReferenceCountMean': [i[0] for i in _means],
                'ExternalReferenceCountStd': [i[0] for i in _stds],
                'ExternalReferenceSCCountMean': [i[1] for i in _means],
                'ExternalReferenceSCCountStd': [i[1] for i in _stds],
            }}

            if self['CollectionAnalysisSettings']['MorphologyAnalysis'] is True:
                _['AdditionalData']['ExternalReferenceCoverageMean'] = [i[2] for i in _means]
                _['AdditionalData']['ExternalReferenceCoverageStd'] = [i[2] for i in _stds]
                _['AdditionalData']['ExternalReferenceSCCoverageMean'] = [i[3] for i in _means]
                _['AdditionalData']['ExternalReferenceSCCoverageStd'] = [i[3] for i in _stds]

            supports.json_dict_push(rf'{self.__out_dir}\AnalysisResults.json', params=_, behavior='update')

            _sheets = ['Normalized Cell Count', 'SC Normalized Cell Count']
            if self['CollectionAnalysisSettings']['MorphologyAnalysis'] is True:
                _sheets += ['Normalized Cell Coverage', 'SC Normalized Cell Coverage']

            warnings.filterwarnings('ignore', category=FutureWarning)  # supress pandas bug
            for n, ds in enumerate(_sheets):
                for mg in mg_options:

                    # set correct normalization value
                    _mean = None
                    for k, v in _dict.items():
                        if k in mg:
                            _mean = v[0]

                    for k, v in self['IndividualAnalysisSettings'].items():
                        if v['DataGroup'] in mg:
                            data_sets[0].loc[k, ds] = data_sets[0].loc[k, ds] / _mean[n]
            warnings.filterwarnings('default', category=FutureWarning)

        data_sets[1] = data_sets[1].T; data_sets[2] = data_sets[2].T  # flip axis on nnd data

        self.data = dict(zip(sheets, data_sets))  # construct data dict for analysis

        if self['CollectionAnalysisSettings']['ExcelExport'] is True:
            excel_df_push(rf'{self.__out_dir}\Results.xlsx', dataframes=data_sets, sheets=sheets,
                          behavior='replace')  # export result dataframes Excel


class FieldCodeHandler:
    def __init__(self, codes):
        self.codes = codes
        self.sep_codes = None
        self.sor_codes = None
        self.id_codes = None

    def split_codes(self, delimiters, **kwargs):
        """Method that splits field codes according to delimiters and kwargs."""

        pars = {
            'regex': False,
            'white_space': False,
            'separator': None,
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        if kwargs['white_space'] is not True:
            delimiters = delimiters.replace(' ', '')

        if kwargs['separator']:
            delimiters = delimiters.split(kwargs['separator'])
        else:
            delimiters = [*delimiters]

        if kwargs['regex'] is True:
            regex = delimiters
        else:
            escape_dict = {
                '[': r'\[', ']': r'\]', '.': r'\.', '^': r'\^', '$': r'\$', '*': r'\*', '+': r'\+',
                '?': r'\?', '{': r'\{', '}': r'\}', '|': r'\|', '(': r'\(', ')': r'\)'
            }

            _ = []
            for d in delimiters:
                for k, v in escape_dict.items():
                    if k in d:
                        d = d.replace(k, v)
                _.append(d)
            delimiters = _

            regex = r'|'.join(delimiters)

        self.sep_codes = [re.split(regex, c) for c in self.codes]

    def sort_codes(self, indices, **kwargs):
        """Method that sorts field codes according to the input indices and separated codes."""

        pars = {
            'reverse_order': False,
            'field_codes': None
        }

        for k, v in pars.items():
            if k not in kwargs:
                kwargs[k] = v

        _ = []
        for i in indices:  # shift negative indices to match the code-appended list set
            if i < 0:
                if kwargs['field_codes'] is not None:
                    _.append(i - 2)
                else:
                    _.append(i - 1)
            else:
                _.append(i)
        indices = _

        if kwargs['field_codes'] is None:
            sort_codes = [i + [j] for i, j in zip(self.sep_codes, self.codes)]
        else:
            sort_codes = [i + [k] + [j] for i, j, k in zip(self.sep_codes, self.codes, kwargs['field_codes'])]
        _ = sorted(sort_codes, key=operator.itemgetter(*indices))

        self.sor_codes = [i[-1] for i in _]
        if kwargs['field_codes'] is not None:
            self.id_codes = [i[-2] for i in _]

        if kwargs['reverse_order'] is True:
            self.sor_codes.reverse()


class MultiGroupHandler(FieldCodeHandler):
    def __init__(self, groups):
        self.multi_groups = None
        super().__init__(groups)

    def group_codes(self, index):
        # map codes to index
        groups = {}
        for c, code in zip(self.sep_codes, self.codes):
            _g = c[index]  # set multi-group name
            if _g not in groups:
                groups[_g] = {}
            c.pop(index)
            groups[_g][code] = ' '.join(c)
        self.multi_groups = groups


def load_mask(mask_selection):
    """Function that loads in a mask file to a readable pandas dataframe."""
    try:  # check for existence of mask file
        mask_file = [i for i in os.listdir(f'{supports.__cwd__}\__masks__') if mask_selection in i][0]
    except IndexError:
        raise ValueError(f'Select a valid mask {mask_selection!r} is invalid.')

    # load in the mask
    mask = load_mask_file(mask_selection)

    return mask


def hough_circles_cell_counting(img, npar):
    """
    Function that counts the number of cells on a given image through Hough transform.
    :param img: opencv formatted image
    :param scale: scalebar for image in pix/µm
    :param remove_edge: remove cells counted in a proximity of remove_edge µm of the edge
    :return: the number of cells counted, along with the number of cells counted based on nuclei area
    """

    _img = background_subtraction(img)  # subtract background from image

    # set image threshold and convert image to binary with that threshold
    binary = cv2.threshold(_img, np.mean(_img.flatten(), dtype=int) + 5, 255, cv2.THRESH_BINARY)[1]
    _canny = cv2.Canny(binary, 100, 200)  # use Canny edge detection on image

    int_dp = 1  # set initial inverse ratio of accumulator resolution
    cell_counts = []
    int_dps = []
    while int_dp < 10:  # iterate over increasing dp until limit
        circles = cv2.HoughCircles(_canny, cv2.HOUGH_GRADIENT, int_dp, 1, param1=50, param2=28,
                                   minRadius=int(npar['nuclei_radius'][0]), maxRadius=int(npar['nuclei_radius'][1]))
        if circles is not None:
            cells = len(circles[0])
            cell_counts.append(cells)
            int_dps.append(int_dp)
        int_dp += 0.05

    if cell_counts:
        max_cell_count = max(cell_counts)  # find max cell count from iterations
    else:
        max_cell_count = 0
    found_dp = int_dps[cell_counts.index(max_cell_count)]  # determine optimum dp from cell count max

    # re-calculate from the found parameters to get cell positions
    found_cells = cv2.HoughCircles(_canny, cv2.HOUGH_GRADIENT, found_dp, 1, param1=50, param2=28,
                                   minRadius=int(npar['nuclei_radius'][0]), maxRadius=int(npar['nuclei_radius'][1]))[0]
    return found_cells, binary


def threshold_cell_counting(img, npar):
    _img = background_subtraction(img)  # subtract background from image
    img_ints = _img.flatten()  # create a list consisting of all intensities for all pixels in the image
    img_int_max = max(img_ints)  # find the maximum pixel intensity in the image
    t_min = np.mean(img_ints, dtype=int) + 5  # set the starting intensity for cell counting to match the image noise

    na_min, na_max = npar['nuclei_area']

    cell_counts, contours, t_mins = [], [], []
    while t_min < img_int_max:
        if len(cell_counts) > 1:
            if max(cell_counts) == 0:
                pass
            elif cell_counts[-1] / max(cell_counts) < 0.5:
                break
            else:
                pass
        binary = cv2.threshold(_img, t_min, 255, cv2.THRESH_BINARY)[1]  # set threshold to convert to binary
        cnt = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # find contours
        cnt = [c for c in cnt if na_min < cv2.contourArea(c) < na_max]
        cell_counts.append(len(cnt)), contours.append(cnt), t_mins.append(t_min)
        t_min += 1
        na_min *= 0.965  # lower nuclei min area by 3.5% for each iteration to account for apparent cell shrinkage
        na_max *= 0.985  # lower nuclei max area by 1.5% for each iteration to account for apparent cell shrinkage

    max_count_id = np.where(np.array(cell_counts) == max(cell_counts))[0][-1]
    found_cells_dim = [cv2.minEnclosingCircle(c) for c in contours[max_count_id]]  # approximate contours to circles
    found_cells = [[cX, cY, r] for ((cX, cY), r) in found_cells_dim]  # fix list structure

    # determine the optimal threshold image and cut it down to 2x2 mm^2
    binary = cv2.threshold(_img, t_mins[max_count_id], 255, cv2.THRESH_BINARY)[1]
    return found_cells, binary


def background_subtraction(img, ksize=101):
    """Function that subtracts the image background with a medium blur kernel with ksize"""
    try:
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # fix image color interpretation
    except cv2.error:
        grayscale = img
    background = cv2.medianBlur(grayscale, ksize=ksize)  # determine the background with a large kernel
    subtraction = cv2.subtract(grayscale, background)  # subtract the background from the grayscale image
    return subtraction

# @supports.timer
def nearest_neighbors(positions):
    """
    For a set of cell coordinates, finds nearest neighbour distances for all cells in the set.
    :param positions: coordinates with shape (x-coordinate, y-coordinate, *_).
    :return: A list of nearest neighbour distances with same order as input coordinates.
    """

    cXs, cYs, *_ = [np.array(k) for k in list(zip(*positions))]  # unpack positions

    _id = 0  # set iterative index
    dist_vectors = []  # placeholder list for cell vector magnitudes
    for cX, cY in zip(cXs, cYs):  # iterate over each cell position

        # create a vector space and find their magnitudes based on the iterative cell position
        vector_lengths = np.sqrt(np.square(cX - cXs) + np.square(cY - cYs))
        vector_lengths[_id] = np.inf  # remove |vector| = 0 from the distance space
        dist_vectors.append(np.int32(vector_lengths.min()))  # find nearest neighbour and save
        _id += 1  # update iterative index

    return dist_vectors


def vector_length(p1, p2):
    """For a set of coordinates, finds the length of the vector between two coordinates."""
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))


def _excel_df_push(path, dataframes, sheets, behavior, **kwargs):
    """Function that pushes a dataframe to an Excel sheet."""

    # delete dead kwargs
    dead_kwargs = ('sheet_name', 'axis')
    for n in dead_kwargs:
        if n in kwargs:
            del kwargs[n]

    if os.path.isfile(path):  # check for existence of Excel file
        xlsx_file = pd.ExcelFile(path, engine='openpyxl')
        if sheets is not None:  # convert to dict catching sheet-induced import dict
            excel_dict = {}
            excel_df = {}
            for s in sheets:
                try:  # catch new sheets
                    excel_df[s] = pd.read_excel(xlsx_file, sheet_name=s, index_col=0)  # load file as dataframe
                    excel_dict[s] = excel_df[s].to_dict()  # convert each dataframe from each sheet to a dict and pack it
                except ValueError:
                    excel_dict[s] = {}
        else:
            excel_df = pd.read_excel(xlsx_file, index_col=0)
            excel_dict = excel_df.to_dict()  # convert to dict if no sheets
    else:
        excel_df = pd.DataFrame()
        excel_dict = {}
        if sheets is not None:
            for s in sheets:
                excel_dict[s] = {}

    if behavior != 'read':
        if behavior == 'update':
            if sheets is None:
                input_dict = dataframes.to_dict()
                output_df = pd.DataFrame(supports.dict_update(excel_dict, input_dict))
            else:
                input_dict, output_df = {}, {}
                for s, df in zip(sheets, dataframes):  # convert passed dataframes to dict with Excel dict structure
                    input_dict[s] = df.to_dict()
                    output_df[s] = pd.DataFrame(supports.dict_update(excel_dict[s], df.to_dict()))
        elif behavior == 'replace':
            if sheets is None:
                output_df = dataframes
            else:
                output_df = {}
                for s, df in zip(sheets, dataframes):
                    output_df[s] = df
        elif behavior == 'clear':
            output_df = {}
        else:
            raise ValueError(f'Behaviour {behavior!r} is invalid.')

        # write dataframes to Excel, respecting sheets
        if sheets is None:
            output_df.to_excel(path, **kwargs)
        else:
            options = {'strings_to_formulas': False,  # prevent the Excel interpreter from corrupting data accidentally
                       'strings_to_urls': False}
            with pd.ExcelWriter(path, engine='xlsxwriter', engine_kwargs={'options': options}) as writer:
                for s in sheets:
                    output_df[s].to_excel(writer, sheet_name=s)
    else:
        return excel_df


def excel_df_push(path, dataframes=None, sheets=None, behavior='update', **kwargs):
    """Wrapper for _excel_df_push that catches the PermissionError arising from an open document."""
    start = time.time()
    while True:
        _runtime = time.time() - start
        try:
            _ = _excel_df_push(path, dataframes, sheets, behavior, **kwargs)
            return _
        except PermissionError:
            if _runtime < supports.__defaults__['FilePollingTimeOut']:
                supports.tprint(
                    f'Failed to create Excel file at {path} due to open file. Close the file to proceed. Reattempting '
                    f'Excel save in 10 seconds.')
                time.sleep(10)
            else:
                raise RuntimeError(f'Failed to create Excel file at {path} due to open file. Close the file and redo the '
                                   f'analysis.')


def contour_center(cnt):
    """Function that determines the center of a contour.
    :param cnt: an opencv contour
    :return: the center of the contour as (cX, cY)"""
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    else:
        cX, cY = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        return cX, cY


def strip_dataframe(dataframe):
    """Function that deconstructs a dataframe into a dictionary of indices and values."""
    _ = {}
    size = dataframe.shape
    for r in range(size[0]):
        for c in range(size[1]):
            _[(r, c)] = dataframe.iloc[r][c]
    return _


def craft_collection_dataframe(dictionary):
    """Function that creates a dataframe from a collection-style dictionary."""
    dfs = []
    for c, rs in dictionary.items():
        dfs.append(pd.DataFrame(rs.values(), index=rs.keys(), columns=(c, )))
    df = pd.concat(dfs, axis=1)
    return df


def craft_dataframe(dictionary):
    """Function that creates a dataframe from a dictionary of indices and values."""
    rows, columns = np.max(list(dictionary.keys()), axis=0) + 1  # fetch number of rows and columns in dataframe
    _ = pd.DataFrame(index=np.arange(rows), columns=np.arange(columns))  # create dummy dataframe
    for k, v in dictionary.items():  # replace indexed values in dummy dataframe
        _.iloc[k] = v
    return _


def load_mask_file(mask, relative_path=True, strip=False):
    """Function that loads a mask file from a given path."""
    if relative_path:
        mask_path = rf'{supports.__cwd__}\__masks__\{mask}.mask'  # set mask file path
    else:
        mask_path = mask
    if os.path.isfile(mask_path):  # read and return mask if it exists, otherwise return None
        mask = pd.read_csv(mask_path, sep='\t', header=None, encoding='utf-8', dtype='string')
        if strip:
            return strip_dataframe(mask)
        else:
            return mask
    else:
        return None


def directory_checker(directory, clean=True):
    """Function that checks the existance of a directory path.
    :params directory: the path to the directory to be checked.
    :params clean: if True, the script removes all files from the folder if it exists. The default is True."""
    if os.path.exists(directory):  # check existence of output path
        if clean:
            existing_files = os.listdir(directory)
            for ef in existing_files:
                os.remove(rf'{directory}\{ef}')
    else:
        os.makedirs(directory)


def import_mask(path):
    """Function that imports a mask from a path and yields a DataFrame."""
    if path.endswith('.txt') or path.endswith('.mask'):
        mask_df = pd.read_csv(path, sep='\t', header=None, encoding='utf-8', dtype='string')
    elif path.endswith('.xlsx'):
        mask_df = pd.read_excel(path, header=None, encoding='utf-8', dtype='string')
    else:
        raise ValueError('Mask file extension must be ".txt", ".mask", or ".xlsx"')
    return mask_df


def str_to_tuple(string, datatype=int):
    """Function that converts a tuple/list caught in a string to a plain tuple/list. Note that if the datatype is str
    the stringified tuple/list must not contain strings with ',' elements."""
    if string[0] == '(':  # if the string is a tuple
        string = string.removeprefix('(').removesuffix(')')
        return tuple(map(datatype, string.split(',')))
    elif string[0] == '[':
        string = string.removeprefix('[').removesuffix(']')
        return list(map(datatype, string.split(',')))


def linear_regression(x, a, b):
    return a * x + b


def sign(a):
    return math.copysign(1, a)


def criterion_resize(image, criterion: bool | int=True):
    """Function that scales an image while preserving aspect ratio such that the largest image dimension is the
    criterion.
    :param image: the opencv image to scale.
    :param criterion: the criterion to scale the image to."""

    if criterion is True:  # default to the application default
        criterion = supports.json_dict_push(rf'{supports.__cache__}\application.json',
                                            behavior='read')['ApplicationSettings']['AuditImageResolution']

    max_size = max(image.shape[:2])
    if max_size > criterion:
        scalar = criterion / max_size
        image = cv2.resize(image, None, fx=scalar, fy=scalar, interpolation=cv2.INTER_AREA)
    return image


def markSquare(cnt):
    """
    This will define the minimum square for a set of contour coordinates
    :param cnt: Contour list
    :return: Coordinates of the square
    """
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    return box


def configure_directory(tkID=None, ptkID=None, **kwargs):
    """
    An extension of the filedialog.askdirectory that configures the directory cache, and adds forbidden folder
    configurations.
    :param tkID: tkID of the calling widget.
    :param ptkID: ptkID of the calling widget parent.
    """

    kwargs['forbidden'] = [] if 'forbidden' not in kwargs else kwargs['forbidden']
    if 'forbidden_message' not in kwargs:
        kwargs['forbidden_message'] = 'Selected directory is restricted.'

    _dir = None
    if tkID is not None and ptkID is not None:
        _dir_memory = supports.json_dict_push(rf'{supports.__cache__}\directory_memory.json', behavior='read')
        if ptkID + tkID in _dir_memory:
            _dir = _dir_memory[ptkID + tkID]
    select_directory = filedialog.askdirectory(initialdir=_dir)

    # catch cancelled selection
    if select_directory:
        if select_directory in kwargs['forbidden']:
            supports.tprint(kwargs['forbidden_message'])
        else:
            supports.tprint(f'Selected directory: {select_directory}')

            if tkID is not None and ptkID is not None:  # update memory if plausible
                _memory = {ptkID + tkID: select_directory}
                supports.json_dict_push(rf'{supports.__cache__}\directory_memory.json', params=_memory,
                                        behavior='update')
            return select_directory
    else:
        supports.tprint('Directory selection was cancelled.')

    return None


def read_channel(*args, **kwargs):
    """Modified cv image reader, with added automatic parameter-based functionality."""
    _kwargs = {
        'scalar': 1,
        'img': None
    }

    for k, v in _kwargs.items():
        if k in kwargs:
            _kwargs[k] = kwargs[k]
            del kwargs[k]

    if _kwargs['img'] is None:
        img = cv2.imread(*args, **kwargs)
    else:
        img = _kwargs['img']

    if _kwargs['scalar'] != 1:
        img = cv2.resize(img, None, fx=_kwargs['scalar'], fy=_kwargs['scalar'], interpolation=cv2.INTER_AREA)

    return img


def define_scalars(scalebar, criterion=1.7, forced_scale=2.5, inverted=False):
    """
    Support function that defines a rescaling factor from the native scalebar and a forced scale.
    :param scalebar: the native scalebar.
    :param criterion: the criterion for when the scalebar is small enough for a rescale to occur. If set to None, the
    scale will always be forced to the defined scale. The default is 1.7 µm/pix.
    :param forced_scale: the forced scale.
    :param inverted: whether the native scale is inverted. The default scale is from pixels to physical unit.
    :return: the new scalebar and the rescaling factor.
    """
    scalebar = scalebar if not inverted else scalebar ** -1
    if criterion is None or scalebar < criterion:  # ensure analyzable scale
        scale_ratio = scalebar / forced_scale
        scalebar = forced_scale
    else: scale_ratio = 1;
    scalebar = scalebar if not inverted else scalebar ** -1
    return scalebar, scale_ratio

