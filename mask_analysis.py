import os
import cv2
import nanoscipy.util as nsu
import feature_recognition as fr
import bright_spot_recognition as bsr
import pandas as pd
import time
import tqdm
import numpy as np


def __time_print__(text):
    """
    Print the time in front of the printed message.
    :param text: The text to be printed.
    :return: Prints the text along with the current time.
    """
    time_stamp = time.localtime()
    print()
    print(f'{time.strftime("%H:%M", time_stamp)} - {text}')


# ::::: PROCESS : SEPARATING STRUCTURES
def scaleBarGetter(folder):
    """
    For a .vsi folder, extracts scale bar data for associated metadata from microscoper.
    :param folder: .vsi file folder
    :return: The mean scale bar for the image as pixel/µm
    """
    # define metadata ids to look for and extract
    metadata_ids = ['PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeXUnit', 'PhysicalSizeYUnit']
    metadata_scale = nsu.xml_extract(rf'{folder}\metadata.xml', metadata_ids)  # extract metadata
    mds_list = [metadata_scale.get(s) for s in metadata_scale]  # get values
    metadata_scale_tuple = ((1 / float(mds_list[0][0]), mds_list[2][0]),
                            (1 / float(mds_list[1][0]), mds_list[3][0]))  # redefine result

    unit_to_pixel = []  # has shape (height, width)
    for md in metadata_scale_tuple:  # fix potential mm unit to µm
        if md[1] == 'mm':
            unit_to_pixel.append(md[0] * 10 ** 3)
        elif md[1] == 'cm':
            unit_to_pixel.append(md[0] * 10 ** 4)
        elif md[1] == 'm':
            unit_to_pixel.append(md[0] * 10 ** 6)
        else:
            unit_to_pixel.append(md[0])

    # take the mean of the scale bars in x and y, and set as the image scale bar
    mean_scalebar = np.mean(unit_to_pixel)

    return mean_scalebar


def cutImgs(tbox, MF, GE, UV, out_dir):

    # sort the tbox coordinates prioritizing x-coordinate
    coordinates = []
    for c in tbox:
        y, x, w, h = cv2.boundingRect(c)  # by swapping x and y coordinates, the desired sorting is found
        coordinates.append([x, y])
    sorted_coordinates = sorted(coordinates, key=lambda k: [k[0], k[1]])
    sorted_id = [coordinates.index(i) for i in sorted_coordinates]
    sorted_tbox = [tbox[i] for i in sorted_id]

    StructureMapList = ['56_0.3;0.3', '55_0.5;0.3', '54_0.8;0.3', '53_1;0.3', '52_E2', '51_E2', '50_E2', '49_E2',
                        '48_0.3;0.5', '47_0.5;0.5', '46_0.8;0.5', '45_1;0.5', '44_1;2', '43_0.8;2', '42_0.5;2',
                        '41_0.3;2', '40_0.3;0.8', '39_0.5;0.8', '38_0.8;0.8', '37_1;0.8', '36_1;1', '35_0.8;1',
                        '34_0.5;1', '33_0.3;1', '32_0.3;1', '31_0.5;1', '30_0.8;1', '29_1;1', '28_1;0.8', '27_0.8;0.8',
                        '26_0.5;0.8', '25_0.3;0.8', '24_0.3;2', '23_0.5;2', '22_0.8;2', '21_1;2', '20_1;0.5',
                        '19_0.8;0.5', '18_0.5;0.5', '17_0.3;0.5', '16_E1', '15_E1', '14_E1', '13_E1', '12_1;0.3',
                        '11_0.8;0.3', '10_0.5;0.3', '09_0.3;0.3', '08_C', '07_4;1', '06_2;2', '05_2;1', '04_2;0.8',
                        '03_2;0.5', '02_2;0.3', '01_C', '64_C', '63_2;0.3', '62_2;0.5', '61_2;0.8', '60_2;1', '59_2;2',
                        '58_1;4', '57_C']
    StructureMapList = nsu.list_sorter(StructureMapList, stype='int_size')  # sort structure map

    # check path validity
    out_folders = [r'\Multi Fluo', r'\Green Ex', r'\UV']
    for i in out_folders:
        if not os.path.exists(out_dir + i):
            os.makedirs(out_dir + i)

    # cut rotated images and save each cut
    sbar = tqdm.tqdm(total=64, leave=False)
    for c, n in zip(sorted_tbox, StructureMapList):
        sbar.set_description(f' >>>   {n.replace("_", " ")}')
        # map squares
        x,y,w,h = cv2.boundingRect(c)
        MF_cut, GE_cut, UV_cut = MF[y:y+h, x:x+w], GE[y:y+h, x:x+w], UV[y:y+h, x:x+w]

        # construct true file path
        MF_path = out_dir + r'\Multi Fluo\_{}.tiff'.format(n)
        GE_path = out_dir + r'\Green Ex\_{}.tiff'.format(n)
        UV_path = out_dir + r'\UV\_{}.tiff'.format(n)

        cv2.imwrite(MF_path, MF_cut)
        cv2.imwrite(GE_path, GE_cut)
        cv2.imwrite(UV_path, UV_cut)
        sbar.update(1)
    sbar.close()
    sbar.refresh()
    return


def removeCommonWords(*snippets):
    # fix list input to expected structure
    if not isinstance(snippets[0], str):
        snippets = snippets[0]

    # separate all words into a list from the strings
    snips = [list(s.split()) for s in snippets]

    # use first string-list as word-matching check-list
    check_list = snips[0]

    # check the list and remove the common elements
    for i in check_list:
        if all(i in j for j in snips):
            for e in snips:
                del e[e.index(i)]
    resolved_snippets = [nsu.list_to_string(i, ' ') for i in snips]

    return resolved_snippets


def structureCellCount(img_path, df_col, scale_bar):
    img = cv2.imread(img_path)  # load image from path
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # fix image color interpretation
    bg_img = cv2.medianBlur(gs_img, ksize=101)  # determine the background with a large kernel
    b_img = cv2.subtract(gs_img, bg_img)  # subtract the background from the grayscale image

    # optimize each img for maximum cell count
    h, w = img.shape[0:2]
    img_ints = b_img.flatten()  # create a list consisting of all intensities for all pixels in the image
    img_int_max = max(img_ints)  # find the maximum pixel intensity in the image
    t_min = np.mean(img_ints, dtype=int) + 5  # set the starting intensity for cell counting to match the image noise

    # set a range for the nuclei radius in pixels and area in pixels^2
    nuclei_radius = np.array([5, 27]) * scale_bar  # the nuclei real size is assumed to be from 5 µm to 27 µm
    nuclei_area = np.square(nuclei_radius) * np.pi
    na_min, na_max = nuclei_area[0], nuclei_area[-1]  # unpack min and max nuclei area
    edge = int(50 * scale_bar)  # define the 50 µm field edge size in pixels

    # determine method employment by an initial cell count
    thresh_img = cv2.threshold(b_img, t_min, 255, cv2.THRESH_BINARY)[1]  # set threshold to convert to binary
    cnt = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # find contours
    init_cell_count = len([c for c in cnt if na_min < cv2.contourArea(c) < na_max])

    if not init_cell_count:  # if the initial count is above 250 cells, used the HoughCircles method
        thresh_img = cv2.threshold(b_img, t_min, 255, cv2.THRESH_BINARY)[1]  # set threshold to convert to binary
        _ones = np.where(thresh_img.flatten() == 255)[0].size  # find all binary ones in the image
        _canny = cv2.Canny(thresh_img, 100, 200)  # use Canny edge detection on image

        int_dp = 1  # set initial inverse ratio of accumulator resolution
        cell_counts = []
        int_dps = []
        while int_dp < 10:  # iterate over increasing dp until limit
            circles = cv2.HoughCircles(_canny, cv2.HOUGH_GRADIENT, int_dp, 1,
                                       param1=50, param2=28, minRadius=nuclei_radius[0], maxRadius=nuclei_radius[1])
            if circles is not None:
                cells = len(circles[0])
                cell_counts.append(cells)
                int_dps.append(int_dp)
            int_dp += 0.05

        max_cell_count = max(cell_counts)  # find max cell count from iterations
        found_dp = int_dps[cell_counts.index(max_cell_count)]  # determine optimum dp from cell count max

        # re-calculate from the found parameters to get cell positions
        found_cells = cv2.HoughCircles(_canny, cv2.HOUGH_GRADIENT, found_dp, 1,
                                   param1=50, param2=28, minRadius=nuclei_radius[0], maxRadius=nuclei_radius[1])[0]

    else:
        cell_counts, contours, t_mins = [], [], []
        while t_min < img_int_max:
            if len(cell_counts) > 1:
                if cell_counts[-1] / max(cell_counts) < 0.5:
                    break
                else:
                    pass
            thresh_img = cv2.threshold(b_img, t_min, 255, cv2.THRESH_BINARY)[1]  # set threshold to convert to binary
            cnt = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # find contours
            cnt = [c for c in cnt if na_min < cv2.contourArea(c) < na_max]
            cell_counts.append(len(cnt)), contours.append(cnt), t_mins.append(t_min)
            t_min += 1
            na_min *= 0.965  # lower nuclei min area by 3.5% for each iteration to account for apparent cell shrinkage
            na_max *= 0.985  # lower nuclei max area by 1.5% for each iteration to account for apparent cell shrinkage

        max_count_id = np.where(np.array(cell_counts) == max(cell_counts))[0][-1]
        found_cnt = contours[max_count_id]  # find contours for max cell count
        found_cells_dim = [cv2.minEnclosingCircle(c) for c in found_cnt]  # approximate contours to circles
        found_cells = [[cX, cY, r] for ((cX, cY), r) in found_cells_dim]  # fix list structure

        # determine the optimal threshold image and cut it down to 2x2 mm^2
        thresh_img = cv2.threshold(b_img, t_mins[max_count_id], 255, cv2.THRESH_BINARY)[1][edge:w-edge, edge:h-edge]
        _ones = np.where(thresh_img.flatten() == 255)[0].size  # find all binary ones in the image

    filter_cell_count = 0
    for cX, cY, r in found_cells:

        # filter away cells if their center is within the outer 50 µm edges of the field
        if edge < cY < h - edge and edge < cX < w - edge:
            if not init_cell_count:
                cv2.circle(img, (int(cX), int(cY)), int(r), (0, 255, 0), 1)
            else:
                cv2.circle(img, (int(cX), int(cY)), int(r), (0, 0, 255), 1)
            filter_cell_count += 1

    # mean_nuclei_area = np.mean(nuclei_area)  # set the mean nuclei area from scale bar
    mean_nuclei_area = np.mean(np.square(list(zip(*found_cells))[-1]) * np.pi)  # set the mean nuclei area from mean contour
    area_cell_count = int(_ones / mean_nuclei_area)  # estimate cell count by using binary area

    # construct output path
    split_path = img_path.split('\\')
    out_dir_path = nsu.list_to_string(split_path[:-1], r'\ '.replace(' ', '')) + ' (Processed)'

    # check path validity
    if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

    # construct specific file path
    out_file_path = out_dir_path + r'\ '.replace(' ', '') + split_path[-1].split('.tiff')[0] + '_UV_processed.tiff'
    cv2.imwrite(out_file_path, img)

    # pack result in DataFrame
    row_name = ' '.join(split_path[-1].split('.tiff')[0].split('_')[1:])
    df, df_area = pd.DataFrame({df_col:[filter_cell_count]}), pd.DataFrame({df_col:[area_cell_count]})
    df.index = [row_name]
    df_area.index = [row_name]
    return df, df_area


# start processing the files in the given folder
def loadImgs(img_dir, img_names, gs_out=True):  # define image loading script

    # load the image and convert to grayscale

    loaded_images = []
    sbar = tqdm.tqdm(total=3, leave=False)
    for i in img_names:
        sbar.set_description(f' >>>   {i[1:]}')
        loaded_images.append(cv2.imread(img_dir + i))
        sbar.update(1)
    sbar.close()
    sbar.refresh()

    MF_img, GE_img, UV_img = loaded_images
    gs_img = cv2.cvtColor(MF_img, cv2.COLOR_BGR2GRAY)

    if gs_out:
        return gs_img, MF_img, GE_img, UV_img
    else:
        return MF_img, GE_img, UV_img


# create compiler function for feature recognition method
def featureRecognition(cv2_images, out_dir, metadata, **kwargs):
    global pbar

    MF_img, GE_img, UV_img = cv2_images[1:]
    pbar.set_description('Identifying Large Feature Set')
    try:
        cnt = fr.identifyMarkerCluster(MF_img, metadata)
    except fr.GaussianMixtureFailure:
        pbar.close()
        pbar.refresh()
        raise fr.GaussianMixtureFailure
    pbar.update(1)
    pbar.set_description('Rotating Images')
    MF_img, GE_img, UV_img = fr.rotateImgs(cnt, MF_img, GE_img, UV_img)
    pbar.update(1)
    pbar.set_description('Finding Ideal Array for Masking')
    if 'angle_max_fields' in kwargs.keys():
        img, tbox = fr.arrayIdentification(MF_img, md_scalar=metadata, angle_max_fields=kwargs.get('angle_max_fields'))
    elif 'fixed_best_angle' in kwargs.keys():
        img, tbox = fr.arrayIdentification(MF_img, md_scalar=metadata, fixed_best_angle=kwargs.get('fixed_best_angle'))
    else:
        img, tbox = fr.arrayIdentification(MF_img, md_scalar=metadata)
    pbar.update(1)
    pbar.set_description('Creating Mask Image')
    fr.createMaskImg(img, out_dir)
    pbar.update(1)
    pbar.set_description('Cutting Images with Mask')
    cutImgs(tbox, MF_img, GE_img, UV_img, out_dir)
    pbar.update(1)
    pbar.close()
    pbar.refresh()


# create compiler function for bright spot recognition method
def brightSpotRecognition(cv2_images, out_dir, metadata, force_square=False):
    global pbar

    gs_img, MF_img, GE_img, UV_img = cv2_images
    pbar.set_description('Adjusting Grayscale')
    cnt, morph_kernel, low_lim, it_thresh = bsr.adjustGrayscale(gs_img, metadata)
    pbar.update(1)
    pbar.set_description('Rotating Images')
    cnt, MF_img, GE_img, UV_img = bsr.rotateImgs(cnt, MF_img, GE_img, UV_img)
    pbar.update(1)
    if force_square:
        pbar.set_description('Extending Square to Mask')
        tbox = bsr.extendSquare(cnt, metadata)
        pbar.update(1)
        pbar.set_description('Creating Mask Image')
        bsr.createMaskImg(MF_img, tbox, out_dir)
        pbar.update(1)
    else:
        pbar.set_description('Finding Ideal Array for Masking')
        img, tbox = fr.arrayIdentification(MF_img, metadata)
        pbar.update(1)
        pbar.set_description('Creating Mask Image')
        fr.createMaskImg(img, out_dir)
        pbar.update(1)

    pbar.set_description('Cutting Images with Mask')
    cutImgs(tbox, MF_img, GE_img, UV_img, out_dir)
    pbar.update(1)
    pbar.close()
    pbar.refresh()


# create compiler function for collective image processing
def preProcessImgs(data_folders, data_path, out_path, images, force_pro, ana_method, angle_max_fields,
                                     fixed_best_angle):
    global pbar

    out_dirs = []
    ana_method = str(ana_method) if not isinstance(ana_method, str) else ana_method
    methods = {'0':'LFSR', '0.1':'LFSR-SMF', '0.2':'LFSR-SA', '0.3':'LFSR-LEGACY', '1':'BSR', '2':'BSR-FS'}
    ANA_method = methods.get(ana_method)

    scale_bars = []
    for file in data_folders:
        vsi_dir = rf'{data_path}\_{file}_'  # set .vsi folder directory
        _scale_bar = scaleBarGetter(vsi_dir)
        scale_bars.append(_scale_bar)

        out_dir = rf'{out_path}\{file}'
        out_dirs.append(out_dir)
        if not all(os.path.isfile(out_dir + r'{}\_64_C.tiff'.format(i.split('.')[0])) for i in
                   images) or file in force_pro:
            __time_print__(f'PRE-PROCESSING ({ANA_method}): {file}')

            pbar = tqdm.tqdm(total=6, desc='Loading Images')
            gs_img, MF_img, GE_img, UV_img = loadImgs(vsi_dir, images, gs_out=True)
            cv2_images = gs_img, MF_img, GE_img, UV_img
            pbar.update(1)
            if ana_method in ('1', '2'):
                if ana_method == '2':
                    force_square = True
                else:
                    force_square = False
                brightSpotRecognition(cv2_images, out_dir, _scale_bar, force_square)
            elif ana_method in ('0', '0.1', '0.2', '0.3'):
                try:
                    if angle_max_fields:
                        featureRecognition(cv2_images, out_dir, _scale_bar, angle_max_fields=angle_max_fields)
                    elif fixed_best_angle:
                        featureRecognition(cv2_images, out_dir, _scale_bar, fixed_best_angle=fixed_best_angle)
                    else:
                        featureRecognition(cv2_images, out_dir, _scale_bar)
                except fr.GaussianMixtureFailure:
                    print('LFSR clustering failed, attempting BSR instead')
                    pbar = tqdm.tqdm(total=6, desc='Loading Images')
                    pbar.update(1)
                    brightSpotRecognition(cv2_images, out_dir, _scale_bar)
    return out_dirs, scale_bars


# count cells in the cut DAPI images (DATA EXTRACTION)
def processImgs(data_folders, out_dirs, out_path, scale_bars):
    sample_set_df, sample_set_df_area = pd.DataFrame(), pd.DataFrame()  # create empty DataFrame to append to
    __time_print__('DETERMINING CELL COUNT FOR STRUCTURES')

    N_files = len(data_folders)
    pbar = tqdm.tqdm(total=N_files)
    for op, sample_name, _sb in zip(out_dirs, data_folders, scale_bars):
        pbar.set_description(sample_name)

        # check files in UV file path
        UV_img_path = rf'{op}\UV'
        UV_files = [i for i in os.listdir(UV_img_path) if i.split('.')[-1] == 'tiff']

        # load and analyze files
        time.sleep(0.5)

        loop_df, loop_df_area = pd.DataFrame(), pd.DataFrame()  # create dummy DataFrame for appending
        sbar = tqdm.tqdm(total=64, leave=False)
        for UV_file in UV_files:
            sbar.set_description(f' >>>   {UV_file[1:-5].replace("_", " ")}')
            UV_path = rf'{UV_img_path}\{UV_file}'  # find true file path
            cell_df, cell_df_area = structureCellCount(UV_path, sample_name, _sb)
            loop_df, loop_df_area = pd.concat([loop_df, cell_df]), pd.concat([loop_df, cell_df_area])
            sbar.update(1)
        sample_set_df, sample_set_df_area = pd.concat([sample_set_df, loop_df], axis=1), pd.concat([
            sample_set_df_area, loop_df_area], axis=1)
        sbar.close()
        sbar.refresh()
        pbar.update(1)
    pbar.close()
    pbar.refresh()
    sample_set_df.sort_index()  # sort DataFrame
    sample_set_df_area.sort_index()  # sort DataFrame
    sample_set_df.to_excel(rf'{out_path}\cellCountData.xlsx')  # save DataFrame
    sample_set_df_area.to_excel(rf'{out_path}\areaCellCountData.xlsx')  # save DataFrame
