import os
import cv2
import nanoscipy.util as nsu
import feature_recognition as fr
import bright_spot_recognition as bsr
import pandas as pd
import time
import tqdm
import numpy as np

# ::::: PROCESS : SEPARATING STRUCTURES
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


def structureCellCount(img_path, df_col):
    img = cv2.imread(img_path)  # load image from path
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # fix image color interpretation
    b_img = cv2.GaussianBlur(gs_img, (3, 3), 2, 2)

    # optimize each img for maximum cell count
    h, w = img.shape[0:2]
    img_area = h * w
    int_fact = np.mean(b_img.flatten()) + 5  # set the starting intensity for cell counting to match the image noise

    # determine method employment by an initial cell count
    thresh_img = cv2.threshold(b_img, int_fact, 255, cv2.THRESH_BINARY)[1]  # set threshold to convert to binary
    cnt = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # find contours
    init_cell_count = len([c for c in cnt if img_area * 0.00002 < cv2.contourArea(c) < img_area * 0.0005])

    # ------ ILLUSTRATION
    if '_04_2;0.8.tiff' in img_path:
        __illustration__ = False
    else:
        __illustration__ = False
    __ill_folder__ = r'C:\Users\nicho\OneDrive - Aarhus Universitet\8SEM\Project in Nanoscience\PowerPoint\Python and NTSAs\cellCount'

    fr.__write_img__(img, 'init', img_folder=__ill_folder__, write=__illustration__)
    fr.__write_img__(b_img, 'blur', img_folder=__ill_folder__, write=__illustration__)
    fr.__write_img__(thresh_img, 'thresh', img_folder=__ill_folder__, write=__illustration__)
    fr.__write_img__(img, 'init_count', img_folder=__ill_folder__, write=__illustration__, c=[c for c in cnt if img_area * 0.00002 < cv2.contourArea(c) < img_area * 0.0005], cc='cyan', ct=2)

    if __illustration__:
        print(f'Initial Cell Count: {init_cell_count}')
    # ------ ILLUSTRATION

    if init_cell_count > 250:  # if the initial count is above 250 cells, used the HoughCircles method
        thresh_img = cv2.threshold(b_img, int_fact + 3, 255, cv2.THRESH_BINARY)[1]  # set threshold to convert to binary
        _ones = sum([len([j for j in i if j == 255]) for i in thresh_img])  # find all binary ones in the image
        _canny = cv2.Canny(thresh_img, 100, 200)  # use Canny edge detection on image

        # ------ ILLUSTRATION
        fr.__write_img__(_canny, 'canny', img_folder=__ill_folder__, write=__illustration__)
        # ------ ILLUSTRATION

        # determine radii range for HoughCircles detection through the image size
        radii_range = [int((img_area * i / np.pi) ** 1 / 2) for i in (0.00001, 0.000085)]

        int_dp = 1  # set initial inverse ratio of accumulator resolution
        cell_counts = []
        int_dps = []
        while int_dp < 10:  # iterate over increasing dp until limit
            circles = cv2.HoughCircles(_canny, cv2.HOUGH_GRADIENT, int_dp, 1,
                                       param1=50, param2=28, minRadius=radii_range[0], maxRadius=radii_range[1])
            if circles is not None:
                cells = len(circles[0])
                cell_counts.append(cells)
                int_dps.append(int_dp)
            int_dp += 0.05

        if __illustration__:
            print(int_dps)
            print(cell_counts)

        max_cell_count = max(cell_counts)  # find max cell count from iterations
        found_dp = int_dps[cell_counts.index(max_cell_count)]  # determine optimum dp from cell count max

        # re-calculate from the found parameters to get cell positions
        found_cells = cv2.HoughCircles(_canny, cv2.HOUGH_GRADIENT, found_dp, 1,
                                   param1=50, param2=28, minRadius=radii_range[0], maxRadius=radii_range[1])[0]

    else:
        cell_counts = []
        contours = []
        while int_fact < 255:
            if len(cell_counts) > 1:
                if max(cell_counts) - cell_counts[-1] > 100:
                    break
                else:
                    pass
            thresh_img = cv2.threshold(b_img, int_fact, 255, cv2.THRESH_BINARY)[1]  # set threshold to convert to binary
            cnt = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # find contours
            cnt = [c for c in cnt if img_area * 0.00002 < cv2.contourArea(c) < img_area * 0.0005]
            cell_counts.append(len(cnt)), contours.append(cnt)
            int_fact += 1

        _ones = sum([len([j for j in i if j == 255]) for i in thresh_img])  # find all binary ones in the image
        found_cnt = contours[cell_counts.index(max(cell_counts))]  # find contours for max cell count
        found_cells_dim = [cv2.minEnclosingCircle(c) for c in found_cnt]  # approximate contours to circles
        found_cells = [[cX, cY, r] for ((cX, cY), r) in found_cells_dim]  # fix list structure

        # ------ ILLUSTRATION
        if __illustration__:
            print(cell_counts)
            fr.__write_img__(thresh_img, 'final_thresh', img_folder=__ill_folder__, write=__illustration__)
        # ------ ILLUSTRATION


    filter_cell_count = 0
    for cX, cY, r in found_cells:

        # filter away cells near edges (cuts away if the cell is an entire cell radius or less outside the border)
        cSize = r / 0.22
        if cY - cSize > 0 and cX - cSize > 0 and cY + cSize < h and cX + cSize < w:
            if init_cell_count > 250:
                cv2.circle(img, (int(cX), int(cY)), int(r), (0, 255, 0), 1)
            else:
                cv2.circle(img, (int(cX), int(cY)), int(r), (0, 0, 255), 1)
            filter_cell_count += 1

    rel_cell_size = 1e-04  # set relative cell area compared to entire image

    avr_area = rel_cell_size * img_area
    area_cell_count = int(_ones / avr_area)  # estimate cell count by using binary area

    # construct output path
    split_path = img_path.split('\\')
    out_dir_path = nsu.list_to_string(split_path[:-1], r'\ '.replace(' ', '')) + ' (Processed)'

    # check path validity
    if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

    # construct specific file path
    out_file_path = out_dir_path + r'\ '.replace(' ', '') + split_path[-1].split('.tiff')[0] + '_UV_processed.tiff'
    cv2.imwrite(out_file_path, img)

    # ------ ILLUSTRATION
    if __illustration__:
        print(f'Final Cell Count: {filter_cell_count}')
    # ------ ILLUSTRATION

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
    ana_method = str(ana_method) if not isinstance(ana_method, str) else ana_method  # ensure the method is always keyed as a string
    methods = {'0':'LFSR', '0.1':'LFSR-SMF', '0.2':'LFSR-SA', '0.3':'LFSR-LEGACY', '1':'BSR', '2':'BSR-FS'}
    ANA_method = methods.get(ana_method)
    metadata_ids = ['PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeXUnit', 'PhysicalSizeYUnit']
    for file in data_folders:
        vsi_dir = rf'{data_path}\_{file}_'  # set .vsi folder directory
        metadata_scale = nsu.xml_extract(rf'{vsi_dir}\metadata.xml', metadata_ids)
        mds_list = [metadata_scale.get(s) for s in metadata_scale]
        metadata_scale_tuple = ((1 / float(mds_list[0][0]), mds_list[2][0]),
                                (1 / float(mds_list[1][0]), mds_list[3][0]))

        # normalize scalebar to meters
        norm_scale = []
        for t in metadata_scale_tuple:
            if t[1] == 'µm':
                norm_scale.append(t[0] / 10e-6)
            elif t[1] == 'mm':
                norm_scale.append(t[0] / 10e-3)
            else:
                norm_scale.append(t[0])

        out_dir = rf'{out_path}\{file}'
        out_dirs.append(out_dir)
        if not all(os.path.isfile(out_dir + r'{}\_64_C.tiff'.format(i.split('.')[0])) for i in
                   images) or file in force_pro:
            time_stamp = time.localtime()
            print()
            print(f'{time.strftime("%H:%M", time_stamp)} - PRE-PROCESSING ({ANA_method}): {file}')

            pbar = tqdm.tqdm(total=6, desc='Loading Images')
            gs_img, MF_img, GE_img, UV_img = loadImgs(vsi_dir, images, gs_out=True)
            cv2_images = gs_img, MF_img, GE_img, UV_img
            pbar.update(1)
            if ana_method in ('1', '2'):
                if ana_method == '2':
                    force_square = True
                else:
                    force_square = False
                brightSpotRecognition(cv2_images, out_dir, norm_scale, force_square)
            elif ana_method in ('0', '0.1', '0.2', '0.3'):
                try:
                    if angle_max_fields:
                        featureRecognition(cv2_images, out_dir, norm_scale, angle_max_fields=angle_max_fields)
                    elif fixed_best_angle:
                        featureRecognition(cv2_images, out_dir, norm_scale, fixed_best_angle=fixed_best_angle)
                    else:
                        featureRecognition(cv2_images, out_dir, norm_scale)
                except fr.GaussianMixtureFailure:
                    print('LFSR clustering failed, attempting BSR instead')
                    pbar = tqdm.tqdm(total=6, desc='Loading Images')
                    pbar.update(1)
                    brightSpotRecognition(cv2_images, out_dir, norm_scale)
    return out_dirs


# count cells in the cut DAPI images (DATA EXTRACTION)
def processImgs(data_folders, out_dirs, out_path):
    # sample_names = [i.replace(' -', '') for i in removeCommonWords(data_folders)]
    sample_set_df, sample_set_df_area = pd.DataFrame(), pd.DataFrame()  # create empty DataFrame to append to
    time_stamp = time.localtime()
    print()
    print(f'{time.strftime("%H:%M", time_stamp)} - DETERMINING CELL COUNT FOR STRUCTURES')
    N_files = len(data_folders)
    pbar = tqdm.tqdm(total=N_files)
    for op, sample_name in zip(out_dirs, data_folders):
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
            cell_df, cell_df_area = structureCellCount(UV_path, sample_name)
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
