import time

from numpy import unique
from numpy import where
import numpy as np
from sklearn.mixture import GaussianMixture
import cv2
import os
import itertools

__BGR_colors__ = {'blue':(255,0,0), 'pastelblue':(255,122,122), 'navy':(122,0,0), 'turquoise':(122,122,0),

                  'red':(0,0,255), 'pastelred':(122,122,255), 'rose':(0,0,122),

                  'green':(0,255,0), 'pastelgreen':(122,255,122), 'forest':(0,122,0), 'lime':(0,255,122),

                  'cyan':(255,255,0), 'pastelcyan':(255,255,122), 'sky':(255,122,0),

                  'magenta':(255,0,255), 'pastelpurple':(255,122,255), 'neonpurple':(255,0,122), 'pink':(122,0,255),
                  'purple':(122,0,122),

                  'yellow':(0,255,255), 'pastelyellow':(122,255,255), 'orange':(0,122,255), 'neongreen':(0,255,122),
                  'olive':(0,122,122),

                  'white':(255,255,255), 'gray':(122,122,122), 'black':(0,0,0)}

class GaussianMixtureFailure(Exception):
    """
    Exception raised when there is not enough components to perform the Gaussian mixture clustering.
    """
    pass

def __range_check__(check_value, range_mean, error, err_type='relative'):
    """
    Helper function that checks a value against a range defined from a value and an error
    :param check_value: the value that is to be checked against the range.
    :param range_mean: the value of which the error is range is to be checked.
    :param error: the size of the error.
    :param err_type: either 'relative' or 'absolute'. Determines whether the error will be computed relative or absolute.
    :return: Either False if the value is not in the range or True if the value is in the range.
    """

    # check if the check value is in the given error range
    if err_type in ('rel', 'relative'):
        if error > 1:  # account for relative error in percentage
            error /= 100
        if range_mean * (1 - error) < check_value < range_mean * (1 + error):
            return True
    elif err_type in ('abs', 'absolute'):
        if range_mean - error < check_value < range_mean + error:
            return True
    else:
        raise ValueError(f'Wrong error type "{err_type}".')

    return False  # return default value if value check is untrue

def __write_img__(img, img_name: str, img_folder, img_type='tif', write=True, **kwargs):
    """
    Saves illustration image to defined path at the top of the document
    :param img: loaded image name from function
    :param img_name: image name
    :param img_type: image extension
    :param overwrite_save_param: overwrite the global save parameter
    :return: image saved to path
    """

    def __kwarg_setter__(name, default):
        if name in kwargs.keys():
            return kwargs.get(name)
        else:
            return default

    kwarg_defs = (False, __BGR_colors__.get('green'), 50)
    # kwarg_ext = c:contours, cc:contour_color, ct:contour_thickness
    kwarg_list = ('c', 'cc', 'ct')
    kwarg_vals = [__kwarg_setter__(i, d) for i, d in zip(kwarg_list, kwarg_defs)]


    if write:
        if kwarg_vals[0]:
            if not isinstance(img[0][0], np.ndarray):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img, kwarg_vals[0], -1, __BGR_colors__.get(kwarg_vals[1]), kwarg_vals[2])
        cv2.imwrite(rf'{img_folder}\{img_name}.{img_type}', img)


def markSquare(cnt):
    """
    This will define the minimum square for a set of contour coordinates
    :param cnt: Contour list
    :return: Coordinates of the square
    """
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    return box

def gaussian_kernel(size, sigma=1):
    """
    Sets up a gaussian kernel for image processing
    :param size:
    :param sigma:
    :return: Gaussian kernel
    """
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def identifyMarkerCluster(img, md_scalar):
    """
    This script will identify and mark the cluster of large features on the NTSA surface to orient the image correctly.
    :param img: the image loaded in with cv2.imread()
    :return: Contours mapping for the cluster
    """

    mean_scale = np.mean(md_scalar)  # find mean scale from x and y scalebar
    field_len = 2.1 * 10e-3 * mean_scale
    field_area = field_len ** 2

    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale

    # sharpen the image to enhance features
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    s_img = cv2.filter2D(g_img, -1, sharpen_kernel)

    # ------ ILLUSTRATION
    __illustration__ = False
    __ill_folder__ = r'C:\Users\nicho\OneDrive - Aarhus Universitet\8SEM\Project in Nanoscience\PowerPoint\Python and NTSAs\LFSR_imgs'
    __write_img__(img, 'init', img_folder=__ill_folder__, write=__illustration__)
    __write_img__(s_img, 'sharp1', img_folder=__ill_folder__, write=__illustration__)
    # ------ ILLUSTRATION

    sharpen_kernel = np.array([[-1, -1, -1], [-1, 7, -1], [-1, -1, -1]])
    s_img = cv2.filter2D(s_img, -1, sharpen_kernel)

    # find maximum grayscale intensity to use for initial threshold
    int_max = max([i for l in s_img for i in l])
    t_img = cv2.threshold(s_img, int_max * 0.5, int_max, cv2.THRESH_BINARY)[1]

    # set up a morphing kernel and apply to smoothen structures
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_img = cv2.morphologyEx(t_img, cv2.MORPH_CLOSE, morph_kernel, iterations=4)

    # ------ ILLUSTRATION
    __write_img__(s_img, 'sharp2', img_folder=__ill_folder__, write=__illustration__)
    __write_img__(t_img, 'thresh', img_folder=__ill_folder__, write=__illustration__)
    __write_img__(morph_img, 'morph', img_folder=__ill_folder__, write=__illustration__)
    # ------ ILLUSTRATION

    # find the large marker
    cont = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # ------ ILLUSTRATION
    __write_img__(morph_img, 'all_count', img_folder=__ill_folder__, write=__illustration__, c=cont, cc='green', ct=50)
    # ------ ILLUSTRATION

    # filter all contours for squares that fit the array dimensions
    # cont = [markSquare(c) for c in cont if 5 * 10 ** 6 > cv2.contourArea(c) > 3 * 10 ** 5]
    cont = [markSquare(c) for c in cont if __range_check__(cv2.contourArea(c), field_area, 15, 'rel')]

    # ------ ILLUSTRATION
    __write_img__(morph_img, 'all_squares', img_folder=__ill_folder__, write=__illustration__, c=cont, cc='yellow',
                  ct=50)
    # ------ ILLUSTRATION

    # filter for the most filled squares
    cnt, cnt_prop = [], []
    for n, c in enumerate(cont):
        x, y, w, h = cv2.boundingRect(c)  # define square properties
        if x > 0 and y > 0:
            square = morph_img[y:y + h, x:x + w]

            # find avg color for each square and isolate the brightest only
            avg_color_per_row = np.average(square, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            if avg_color > 0.35 * 255:
                cnt.append(c)
                cnt_prop.append((x, y, w, h))

    # ------ ILLUSTRATION
    __write_img__(morph_img, '35pct_squares', img_folder=__ill_folder__, write=__illustration__, c=cont, cc='cyan',
                  ct=50)
    # ------ ILLUSTRATION

    # attempt to identify the amount of clusters in image
    # this step should not make any sense
    (h, w) = img.shape[:2]
    clust_count = 0
    if any(p[0] < w * 0.4 and p[1] < h * 0.6 for p in cnt_prop):
        clust_count += 1
    if any(p[0] > w * 0.4 and p[1] < h * 0.4 for p in cnt_prop):
        clust_count += 1
    if any(p[0] > w * 0.6 and p[1] > h * 0.4 for p in cnt_prop):
        clust_count += 1
    if any(p[0] < w * 0.6 and p[1] > h * 0.6 for p in cnt_prop):
        clust_count += 1

    # perform connectivity based clustering
    ''' 
    Gaussian machine learning algorithm
    The script uses connectivity based clustering. This will attempt to collect all squares that are in close proximity 
    of each other. The expected amount of clusters is based on, whether a contour is found in each quadrant of 
    the image. 
    Once the squares in the area have been collected, we find the largest cluster as this is almost certainly the main
    cluster. Then the ids for the squares in the cluster are extracted and the corresponding contours are isolated.
    '''

    # if clust_count > 1:  # if there are indeed multiple clusters, find the largest one

    model = GaussianMixture(n_components=clust_count)  # define Gaussian ML model
    pos = [i[:2] for i in cnt_prop]  # isolate the contour bondingRect (x, y) positions
    try:
        model.fit(pos)  # fit the model to the positions
    except ValueError:
        raise GaussianMixtureFailure
    yhat = model.predict(pos)  # predict the clustering
    clusters = [where(yhat == c)[0] for c in unique(yhat)]  # isolate square ids
    clust_size = [len(i) for i in clusters]  # find cluster sizes
    clust_marker = clusters[clust_size.index(max(clust_size))]  # find square ids for the largest cluster
    cnt = [cnt[i] for i in clust_marker]  # filter the contours to only be the largest cluster

    # temp_path = r"C:\Users\nicho\OneDrive - Aarhus Universitet\6SEM\Bachelor\Report\v2\graphics\raster"
    # out_img = cv2.cvtColor(morph_img.copy(), cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(out_img, cnt, -1, (0, 0, 255), 40)
    # scale = 50
    # resized = cv2.resize(out_img, (int(out_img.shape[1] * scale / 100), int(out_img.shape[0] * scale / 100)),
    #                      interpolation=cv2.INTER_AREA)
    # cv2.imwrite(temp_path + r'\gaussianClustering.png', resized)

    return cnt


def rotateImgs(contour, img, GE_img, UV_img):
    """
    Rotate the image appropriately, depending on whether the cluster was identified, compared to where it should be.
    :param GE_img: layered image to also be rotated
    :param UV_img: layered image to also be rotated
    :param contour: gaussian cluster contours
    :param img: image where the clusters are identified
    :return: Rotated images
    """

    # get shape params for rotating the image correctly
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    if contour[0][0][0] < w * 0.4 and contour[0][0][1] < h * 0.6:
        M = cv2.getRotationMatrix2D((cX, cY), 270, 1.0)
    elif contour[0][0][0] > w * 0.4 and contour[0][0][1] < h * 0.4:
        M = cv2.getRotationMatrix2D((cX, cY), 0, 1.0)
    elif contour[0][0][0] > w * 0.6 and contour[0][0][1] > h * 0.4:
        M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
    else:
        M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)

    # rotate all images at the same time
    img = cv2.warpAffine(img, M, (w, h))
    GE_img = cv2.warpAffine(GE_img, M, (w, h))
    UV_img = cv2.warpAffine(UV_img, M, (w, h))

    return img, GE_img, UV_img


def extendSquareX(cont, field_space, box_params, dupes):
    """
    This function will extend a square defined from 4 (x,y) coordinates, respecting its angle according to the norm.

    Parameters
        box_xy : list
            The 4 coordinates packed into a list.
        field_space : float
            The spacing between adjacent squares in pixels.
        dupes : int
            The amount of duplicates wanted. Thus, for a row of 4 squares total, dupes = 3.
    """

    # define single-box parameters
    x_box, y_box = np.transpose(cont)
    x_len, y_len, angle = box_params

    # x_peri =
    # x_peri_shift = best_x * np.sin(best_angle)
    # y_peri = best_y * np.cos(best_angle)
    # y_peri_shift = best_y * np.sin(best_angle)
    # box_par = x_peri, x_peri_shift, y_peri, y_peri_shift



    # check extension direction
    if dupes < 0:
        ext_dir = -1
        dupes *= -1
    else:
        ext_dir = 1

    # find absolute x spacing and define shifts
    x_shift = field_space + x_len * np.cos(angle)
    y_shift = x_shift * np.sin(angle)
    # if angle < 0:
    #     y_shift *= -1
    # y_shift = y_peri_shift / x_peri * x_shift

    # construct new list with all extensions
    n_box = [np.transpose([x_box - int(x_shift * i * ext_dir), y_box - int(y_shift * i * ext_dir)])
             for i in range(dupes + 1)]

    return n_box

def extendSquareRowY(ext_box_coords, field_space, box_params, dupes):
    """
    This function will extend a square row defined from a set of 4 (x,y) coordinates, respecting its angle according to
    the norm.

    Parameters
        ext_box_coords : list
            The 4 coordinates packed into a list of multiple coordinates.
        field_space : float
            The spacing between adjacent squares in pixels.
        box_params : list
            The x perimeter, its shift, and the y perimeter and its shift packed into a list in that order.
        dupes : int
            The amount of duplicates wanted. Thus, for a column of 4 rows total, dupes = 3.
    """

    # define single-box parameters
    # x_peri, x_peri_shift, y_peri, y_peri_shift = box_params
    x_len, y_len, angle = box_params

    # find absolute y spacing and define shifts
    y_shift = field_space + y_len * np.cos(angle)
    x_shift = y_shift * np.sin(angle)
    # if angle < 0:
    #     x_shift *= -1
    # x_shift = x_peri_shift / y_peri * y_shift

    # check extension direction
    if dupes < 0:
        ext_dir = -1
        dupes *= -1
    else:
        ext_dir = 1

    # construct new list with all extensions
    n_box = []
    for i in range(dupes + 1):
        # construct mathematical operation matrix
        opr_list = np.repeat([np.transpose([[- int(x_shift * ext_dir * i)] * 4, [int(y_shift * ext_dir * i)] * 4])],
                             len(ext_box_coords), axis=0)
        n_box.append(ext_box_coords - opr_list)

    # fix output list shape
    shape_box = []
    for i in n_box:
        for j in i:
            shape_box.append(j)

    return shape_box

def arrayIdentification(img, md_scalar, **kwargs):
    """
    Identify an array (or a few) to map to mask and extend to all arrays with proper dimensions and angles.
    :param img: image for array identification
    :param md_scalar: 2D tuple consisting of the scalebar (pixel/length) and the length unit in x and y directions
    :return:
    """

    # define NTSA parameters from metadata scalar
    mean_scale = np.mean(md_scalar)  # find mean scale from x and y scalebar
    NTSA_len = 20 * 10e-3 * mean_scale
    field_len = 2.1 * 10e-3 * mean_scale
    field_area = field_len ** 2
    field_sep = .4 * 10e-3 * mean_scale

    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert grayscale
    blur_img = cv2.medianBlur(g_img, 5)  # blur the image to enhance entire-array identification

    # ------ ILLUSTRATION
    __illustration__ = False
    __ill_folder__ = r'C:\Users\nicho\OneDrive - Aarhus Universitet\8SEM\Project in Nanoscience\PowerPoint\Python and NTSAs\arrayID'
    __write_img__(img, 'init', img_folder=__ill_folder__, write=__illustration__)
    __write_img__(blur_img, 'blur', img_folder=__ill_folder__, write=__illustration__)
    # ------ ILLUSTRATION

    def __contour_detect__(gs_range):
        thresh_img = cv2.threshold(blur_img, gs_range[0], gs_range[1], cv2.THRESH_BINARY_INV)[1]  # convert to binary
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # define morphology kernel
        morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, morph_kernel,
                                     iterations=2)  # apply morphology to threshold
        cont = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # find contours

        # determine binary coverage with the current threshold
        avg_color_per_row = np.average(morph_img, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        col_ratio = avg_color / g_range[1]

        # find a rough estimate of the fields in the image
        print_cnt = []
        cnt = []
        for c in cont:
            x, y, w, h = cv2.boundingRect(c)
            # cArea = cv2.contourArea(c)  # it should be more optimal to take the bounding box area
            c_area = w * h
            c_ratio = w / h

            # find squares by matching height and width ratio
            if __range_check__(c_area, field_area, 15, 'rel') and __range_check__(c_ratio, 1, 0.02, 'abs'):
                cnt.append((c, c_area, c_ratio))
                print_cnt.append(c)
        # ------ ILLUSTRATION
        __write_img__(morph_img, f'_whites_{gs_range[0]}_{gs_range[1]}', img_folder=__ill_folder__, write=__illustration__, c=print_cnt,
                      cc='yellow',
                      ct=50)
        # ------ ILLUSTRATION

        return cnt, col_ratio

    # def detectArrays(img, gs_range, ratio_extend=0.001, area_extend=0.5):
    #     """
    #     This sub-function is designed to detect the arrays
    #     :param img: image for detection
    #     :param gs_range: grayscale range to use for detection
    #     :return:
    #     """
    #
    #     thresh_img = cv2.threshold(img, gs_range[0], gs_range[1], cv2.THRESH_BINARY_INV)[1]  # set initial threshold
    #     morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # define morphology kernel
    #     morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, morph_kernel,
    #                                  iterations=2)  # apply morphology to threshold
    #     cont = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #
    #     cnt = []
    #     for c in cont:
    #         x, y, w, h = cv2.boundingRect(c)
    #         cArea = cv2.contourArea(c)
    #
    #         # find squares by matching height and width ratio
    #         if __range_check__(cArea, field_area, area_extend, 'rel') and __range_check__(w / h, 1, ratio_extend, 'abs'):
    #             cnt.append(markSquare(c))
    #     return cnt, morph_img, thresh_img

    g_range = np.array([0, 15])  # set initial grayscale range fraction
    if 'angle_max_fields' in kwargs.keys():
        angle_max_fields = kwargs.get('angle_max_fields')
    else:
        angle_max_fields = 6

    # first, find all ranges where more than 10% and less than 90% of the image is marked in squares
    _whites = 0
    _contours = []
    while _whites <= .9:
        curr_cont, _whites = __contour_detect__(g_range)  # find contours in the image with the current range
        if _whites > .1:  # if threshold is true, save the contours to list
            _contours.append(curr_cont)
        g_range += 1  # update threshold range

    # second, go through the found contours, and determine the best fits
    _stop_iterator = False
    fa_error, r_error = 3, .001  # define initial acceptance for field_area (rel) and ratio (abs)
    tolerance = 0
    _field_maps = []
    while not _stop_iterator:  # while no more than 20 squares has been found, update the error tolerance
        _field_set = []
        for cs in _contours:
            _fields = []
            _field_cont = 0
            for c, ca, cr in cs:  # find squares by matching height and width ratio
                if __range_check__(ca, field_area, fa_error, 'rel') and __range_check__(cr, 1, r_error, 'abs'):
                    _fields.append(markSquare(c))
                    _field_cont += 1

            if _field_cont > angle_max_fields:  # stop iterating when sufficient fields have been found
                _stop_iterator = True

            # if there are 1 or more fields in the current field list, save them
            if _field_cont > 1:
                _field_set.append(_fields)

        if _field_set:  # if any fields were found with the current tolerance, save them with the tolerance
            _field_maps.append([tolerance, _field_set])

        if not _stop_iterator:  # if iterations should continue, update tolerance
            tolerance += 1
            fa_error += .5
            r_error += .001

    # # optimize contours till arrays are identified
    # while not contour:
    #     # cv2.imshow("pros_img", cv2.resize(pros_img, (int(pros_img.shape[1] * 10 / 100), int(pros_img.shape[0] * 10 / 100)),
    #     #                   interpolation=cv2.INTER_AREA))
    #     # cv2.waitKey(0)
    #
    #     # redo array detection
    #     start = time.time()
    #     contour, pros_img, thresh_img = detectArrays(blur_img, g_range, area_extend=4.5 + 0.5 * ext_factor,
    #                                                  ratio_extend=0.001 * ext_factor)
    #     end = time.time()
    #     print(f'{end - start=}')
    #
    #     # ------ ILLUSTRATION
    #     if g_range[0] in (0, 5, 10, 15, 50, 100, 150, 200, 240):
    #         __write_img__(pros_img, f'morph_range{g_range[0]}_{g_range[1]}', img_folder=__ill_folder__,
    #                       write=__illustration__)
    #     # ------ILLUSTRATION
    #
    #     # determine amount of coloured pixels compared to black pixels
    #
    #     avg_color_per_row = np.average(pros_img, axis=0)
    #     avg_color = np.average(avg_color_per_row, axis=0)
    #     _whites = avg_color / g_range[1]
    #
    #     # if the end is reached and no contours have been found, try again with broader thresholds
    #     if g_range[1] > 255 or _whites > .9:
    #         ext_factor += 1  # extend ratio range
    #         g_range = np.array([0, 15])  # reset intensity scale
    #     else:
    #         g_range += 1

    # # ------ ILLUSTRATION
    # __write_img__(pros_img, f'morph_range{g_range[0]}_{g_range[1]}', img_folder=__ill_folder__, write=__illustration__)
    # __write_img__(pros_img, f'morph_cont_range{g_range[0]}_{g_range[1]}', img_folder=__ill_folder__,
    #               write=__illustration__, c=contour, cc='green', ct=50)
    # __write_img__(thresh_img, f'thresh_range{g_range[0]}_{g_range[1]}', img_folder=__ill_folder__,
    #               write=__illustration__)
    # ill_img = img.copy()
    # # ------ ILLUSTRATION

    def __put_contour_text__(text, cnt, col, type, fs=4, tt=3):
        x, y = cv2.boundingRect(cnt)[:2]
        if type == 'best':
            coords = (x + 320, y + 350)
        else:
            coords = (x + 390, y + 500)
        cv2.putText(img=img, text=text, org=coords, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fs, color=col,
                    thickness=tt)

    def __put_guide_text__(text, coords, col, fs=3, tt=3):
        cv2.putText(img=img, text=text, org=coords, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fs, color=col,
                    thickness=tt)


    # draw best and worst contour tolerances and tolerance guide in the corner
    best_tolerance, worst_tolerance = _field_maps[0][0], _field_maps[-1][0]

    # define best contours as all contours found in the lowest tolerance run
    best_contours = list(itertools.chain(*_field_maps[0][1]))

    # define worst contours as the maximum contours found in the highest tolerance run
    high_tol_conts = _field_maps[-1][1]
    high_tol_count = [len(i) for i in high_tol_conts]
    worst_contours = high_tol_conts[high_tol_count.index(max(high_tol_count))]

    # ------ ILLUSTRATION
    __write_img__(img.copy(), f'_best', img_folder=__ill_folder__,
                  write=__illustration__, c=best_contours,
                  cc='pastelgreen',
                  ct=50)
    __write_img__(img.copy(), f'_worst', img_folder=__ill_folder__, write=__illustration__,
                  c=worst_contours,
                  cc='pastelred',
                  ct=50)
    # ------ ILLUSTRATION

    best_guide_text = f'T{best_tolerance}: +/-{fa_error - worst_tolerance * 0.5:.1f}% ' \
                      f'Area; +/-{r_error / (worst_tolerance + 1):.3f} Ratio'
    if best_tolerance == worst_tolerance:
        best_contours_text = [c for c in best_contours if __range_check__(c, c[0][0], 100, 'abs') and
                              __range_check__(c, c[0][1], 100, 'abs')]
        for c in best_contours_text:
            __put_contour_text__(f'T{best_tolerance}', c, __BGR_colors__.get('pastelpurple'), 'best')
        __put_guide_text__(best_guide_text, (50, 120), __BGR_colors__.get('pastelpurple'))
    else:
        for c in best_contours:
            __put_contour_text__(f'T{best_tolerance}', c, __BGR_colors__.get('pastelgreen'), 'best')
        __put_guide_text__(best_guide_text, (50, 120), __BGR_colors__.get('pastelgreen'))
        for c in worst_contours:
            __put_contour_text__(f'T{worst_tolerance}', c, __BGR_colors__.get('pastelred'), 'worst')
        worst_guide_text = f'T{worst_tolerance}: +/-{fa_error:.1f}% Area; +/-{r_error:.3f} Ratio'
        __put_guide_text__(worst_guide_text, (2000, 120), __BGR_colors__.get('pastelred'))

        # [cv2.circle(img, cv2.boundingRect(c)[:2], 40, __BGR_colors__.get('magenta'), 3) for c in _field_maps[0][1]]

    # # ------ ILLUSTRATION
    # __write_img__(ill_img, 'found_square', img_folder=__ill_folder__, write=True, c=contour, cc='magenta', ct=50)
    # # ------ ILLUSTRATION

    def __find_box_parameters__(contour):
        x_box, y_box = np.transpose(contour)  # separate box coordinates

        # find tilt direction and determine parameters
        if y_box[0] > y_box[2]:  # if first (x, y) is bottom-left corner (right tilt)
            x_peri = x_box[-1] - x_box[0]
            x_peri_shift = x_box[0] - x_box[1]
            y_peri = y_box[0] - y_box[1]
            y_peri_shift = y_box[-1] - y_box[0]
            angle = np.arctan(y_peri_shift / x_peri)
        else:  # if first (x, y) is top-left corner (left tilt)
            x_peri = x_box[1] - x_box[0]
            x_peri_shift = x_box[-1] - x_box[0]
            y_peri = y_box[-1] - y_box[0]
            y_peri_shift = y_box[0] - y_box[1]
            angle = - np.arctan(y_peri_shift / x_peri)

        x_len = np.sqrt(np.power(x_peri, 2) + np.power(y_peri_shift, 2))
        y_len = np.sqrt(np.power(y_peri, 2) + np.power(x_peri_shift, 2))
        box_par = x_len, y_len, angle  # pack square coordinates

        return box_par

    # find the most likely field parameters from an average of the best tolerance
    best_dims = [__find_box_parameters__(c) for c in best_contours]
    box_x, box_y = np.transpose(best_dims)[:2]
    best_x, best_y = np.mean(box_x), np.mean(box_y)

    # find the most likely angular offset from the average of the worst tolerance (most accepted/found fields)
    worst_dims = [__find_box_parameters__(c) for c in worst_contours]
    box_angle = np.transpose(worst_dims)[2]

    if 'fixed_best_angle' in kwargs.keys():
        best_angle = kwargs.get('fixed_best_angle')
    else:
        best_angle = np.mean(box_angle)

    box_par = best_x, best_y, best_angle

    box_par_guide_text = f'Determined Field Parameters: H{int(np.round(best_x))}, W{int(np.round(best_y))} with ' \
                         f'{len(best_contours)} T{best_tolerance} Fields, and A{np.rad2deg(best_angle):.3f} with ' \
                         f'{len(worst_contours)} T{worst_tolerance} Fields'
    __put_guide_text__(box_par_guide_text, (50, 240), __BGR_colors__.get('white'))


    # find single-array position
    x, y, w, h = cv2.boundingRect(best_contours[0])  # define single-array parameters

    # find x-position in arbitrary coordinate array
    x_pos_change = x
    x_pos_id = 0
    while x_pos_change > 0:
        x_pos_change -= w + field_sep
        x_pos_id += 1

    # find y-position in arbitrary coordinate array
    y_pos_change = y
    y_pos_id = 0
    while y_pos_change > 0:
        y_pos_change -= h + field_sep
        y_pos_id += 1

    # find initial row of squares based on initial parameters
    pos_row_box = extendSquareX(best_contours[0], field_sep, box_par, x_pos_id - 1)
    neg_row_box = extendSquareX(best_contours[0], field_sep, box_par, - (8 - x_pos_id))
    row_box = pos_row_box + neg_row_box[1:]
    cv2.drawContours(img, row_box, -1, __BGR_colors__.get('pastelcyan'), 2)  # draw the masked row

    # # ------ ILLUSTRATION
    # __write_img__(ill_img, 'init_row', img_folder=__ill_folder__, write=True, c=row_box, cc='cyan', ct=50)
    # # ------ ILLUSTRATION

    # compute field separation correction from the actual mask row length and redo the linear mask
    """Note here that the true dimensions of the NTSA are the following. Each field is 2.1 mm x 2.1 mm with a .2 mm 
    padding correspnding to the blank field between samples. Therefore, the true dimensions of each field is 2.3 mm x 
    2.3 mm, and 8 of these makes up the 20 mm x 20 mm NTSA survey area. Consequently, the length of the linear mask will 
    always be the NTSA_field_len - NTSA_field_sep."""
    x_vals = np.array([])
    for arr in row_box:  # separate all x values
        xs = arr[:, 0]
        x_vals = np.concatenate((x_vals, xs), 0)
    x_vals.sort()  # sort from low to high
    masked_NTSA_len = x_vals[-2] - x_vals[0]  # find initial linear mask length
    mask_scalar = masked_NTSA_len / (NTSA_len - field_sep)
    field_sep /= mask_scalar

    # redo the mapping based on the new parameters in both x and y directions
    pos_row_box = extendSquareX(best_contours[0], field_sep, box_par, x_pos_id - 1)
    neg_row_box = extendSquareX(best_contours[0], field_sep, box_par, - (8 - x_pos_id))
    row_box = pos_row_box + neg_row_box[1:]

    pos_col_box = extendSquareRowY(row_box, field_sep, box_par, y_pos_id - 1)
    neg_col_box = extendSquareRowY(row_box, field_sep, box_par, - (8 - y_pos_id))
    tbox = pos_col_box + neg_col_box[8:]

    # draw contours on the image for mapping
    cv2.drawContours(img, tbox, -1, __BGR_colors__.get('pastelyellow'), 2)

    # # ------ ILLUSTRATION
    # __write_img__(ill_img, 'final_row', img_folder=__ill_folder__, write=True, c=row_box, cc='yellow', ct=50)
    # __write_img__(ill_img, 'full_mask', img_folder=__ill_folder__, write=True, c=tbox, cc='yellow', ct=50)
    # # ------ ILLUSTRATION

    return img, tbox

def createMaskImg(img, out_dir):

    # check path validity
    dir_split = out_dir.split('\\')
    man_ctrl_dir = r'{}\_masks for manual control'.format('\\'.join(dir_split[:-1]))
    file_name = dir_split[-1]
    writing_dirs = (out_dir, man_ctrl_dir)
    for d in writing_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # write initial mask
    cv2.imwrite(rf"{out_dir}\StructureMask.tiff", img)

    # create and write control mask
    ctrl_mask = cv2.resize(img, (int(img.shape[1] * .13), int(img.shape[0] * .13)), interpolation=cv2.INTER_AREA)
    ctrl_mask = cv2.convertScaleAbs(ctrl_mask, alpha=2, beta=20)  # contrast enhance and brighten ctrl image
    cv2.imwrite(rf"{man_ctrl_dir}\{file_name}.png", ctrl_mask)