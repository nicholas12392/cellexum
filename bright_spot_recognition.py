import cv2
import numpy as np
import nanoscipy.util as nsu
import nanoscipy.functions as nsf
import time
import imutils
import os
import tqdm
import pandas as pd
import feature_recognition as fr



# ::::: PROCESS : ADJUSTING GRAYSCALE MINIMUM INTENSITY
def adjustGrayscale(img, md_scalar):

    mean_scale = np.mean(md_scalar)  # find mean scale from x and y scalebar
    field_len = 2.1 * 10e-3 * mean_scale
    field_area = field_len ** 2

    blur_img = cv2.medianBlur(img, 5) # blur image to remove noise


    # ------ ILLUSTRATION
    __illustration__ = False
    __ill_folder__ = r'C:\Users\nicho\OneDrive - Aarhus Universitet\8SEM\Project in Nanoscience\PowerPoint\Python and NTSAs\BSR_imgs'
    fr.__write_img__(img, 'init', img_folder=__ill_folder__, write=__illustration__)
    fr.__write_img__(blur_img, 'medianBlur', img_folder=__ill_folder__, write=__illustration__)
    # ------ ILLUSTRATION

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))  # define morphology kernel

    it_thresh = []
    low_lim = 255
    skip_ratio_constraint = False  # start iterating with a constraint of a square
    while len(it_thresh) != 1:  # iterate while there is more (or less) than one area of 500 000 pixels
        low_lim -= 1  # update lower limit, until criteria is reached
        if low_lim == 0:  # if the lower limit is reached, then it needs to restart with no ratio constraint
            skip_ratio_constraint = True
            low_lim = 255

        thresh_img = cv2.threshold(blur_img, low_lim, 255, cv2.THRESH_BINARY)[1]  # create threshold image from range
        morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, morph_kernel, iterations=2)  # morph the image

        # find contours based on the morphed image and create them as squares
        cnt = [fr.markSquare(c) for c in cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]

        # sort and filter the found contours based on area constraint and ratio constraint if set
        it_thresh = []
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            cArea = cv2.contourArea(c)
            if not skip_ratio_constraint:
                if fr.__range_check__(cArea, field_area, 15, 'rel') and fr.__range_check__(w / h, 1, .005, 'abs'):
                    it_thresh.append(cArea)
            else:
                if fr.__range_check__(cArea, field_area, 15, 'rel'):
                    it_thresh.append(cArea)

        # ------ ILLUSTRATION
        if __illustration__ and low_lim in (254, 45, 42):
            fr.__write_img__(morph_img, f'int_{low_lim}', img_folder=__ill_folder__, write=__illustration__, c=cnt,
                             cc='green', ct=50)
        # ------ ILLUSTRATION

    # set the contour to be the threshold value

    # ------ ILLUSTRATION
    fr.__write_img__(morph_img, f'int_{low_lim}_all', img_folder=__ill_folder__, write=__illustration__, c=cnt, cc='green',
                  ct=50)
    fr.__write_img__(thresh_img, f'int_{low_lim}_tresh', img_folder=__ill_folder__, write=__illustration__)
    fr.__write_img__(morph_img, f'int_{low_lim}_morph', img_folder=__ill_folder__, write=__illustration__)
    # ------ ILLUSTRATION

    cnt = cnt[[cv2.contourArea(c) for c in cnt].index(it_thresh[0])]
    # temp_path = r"C:\Users\nicho\OneDrive - Aarhus Universitet\6SEM\Bachelor\Report\v3\graphics\raster"
    # out_img = cv2.cvtColor(morph_img.copy(), cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(out_img, [cnt], -1, (255, 0, 0), 40)
    # scale = 10
    # resized = cv2.resize(img, (int(img.shape[1] * scale / 100), int(img.shape[0] * scale / 100)),
    #                      interpolation=cv2.INTER_AREA)
    # cv2.imwrite(temp_path + r'\rawNTSA.png', resized)

    # ------ ILLUSTRATION
    fr.__write_img__(morph_img, f'int_{low_lim}', img_folder=__ill_folder__, write=__illustration__, c=cnt,
                     cc='blue', ct=50)
    # ------ ILLUSTRATION

    return cnt, morph_kernel, low_lim, it_thresh


def rotateCoords(origin, point, angle):
    """
    Rotate a point clockwise by a given angle around a given origin.
    """

    ox, oy = origin  # define origin point coords
    px, py = point  # define target point coords

    angle = np.deg2rad(-angle)  # fix degrees to radians

    # morph the coordinates according to the rotation
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return int(qx), int(qy)


def rotateMultiCoords(origin, points, angle):
    """
    Rotate multiple points clockwise by a given angle around a given center point.
    :param origin: origin coordinates
    :param points: target coordinates to be rotated packed in a list
    :param angle: angle in degrees
    :return: Rotated coordinated in the same structure as input
    """
    rot_coords = []
    for i in points:
        rot_coords.append(rotateCoords(origin, i, angle))

    return rot_coords


# ::::: PROCESS : ROTATE IMAGE AND CONTOUR
def rotateImgs(contour, MF_img, GE_img, UV_img):
    (h, w) = MF_img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    cnt_x, cnt_y, cnt_w, cnt_h = cv2.boundingRect(contour)
    # get shape params for rotating the image correctly
    if cnt_x > 0.5 * w or cnt_y > 0.5 * h:
        if cnt_x > 0.5 * w and cnt_y > 0.5 * h:
            M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
            angle = 180
        elif cnt_x > 0.5 * w and 0.5 * h > cnt_y:
            M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
            angle = 90
        else:
            M = cv2.getRotationMatrix2D((cX, cY), 270, 1.0)
            angle = 270

        # rotate all images at the same time
        MF_img = cv2.warpAffine(MF_img, M, (w, h))
        GE_img = cv2.warpAffine(GE_img, M, (w, h))
        UV_img = cv2.warpAffine(UV_img, M, (w, h))
        contour = rotateMultiCoords((cX, cY), contour, angle)

    return contour, MF_img, GE_img, UV_img


def markSquare(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    return box


def extendSquareX(box_coords, field_space, dupes=-7):
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

    # define x and y coordinates of initial square
    x_box, y_box = np.transpose(box_coords)

    # determine the perimeters and the perimeter shift from angle tilt
    x_box_sorted = nsu.list_sorter(x_box, stype='int_size', reverse=True)
    y_box_sorted = nsu.list_sorter(y_box, stype='int_size', reverse=True)
    x_peri, x_peri_shift = x_box_sorted[0] - x_box_sorted[2], x_box_sorted[0] - x_box_sorted[1]
    y_peri, y_peri_shift = y_box_sorted[0] - y_box_sorted[2], y_box_sorted[0] - y_box_sorted[1]

    # find absolute x spacing and define shifts
    x_shift = field_space + x_peri
    y_shift = y_peri_shift / x_peri * x_shift

    # check extension direction
    if dupes < 0:
        ext_dir = 1
        dupes *= -1
    else:
        ext_dir = -1

    # construct new list with all extensions
    n_box = [np.transpose([x_box - int(x_shift * i * ext_dir), y_box - int(y_shift * i * ext_dir)])
             for i in range(dupes + 1)]

    # add square parameters to output
    box_par = x_peri, x_peri_shift, y_peri, y_peri_shift

    return n_box, box_par


def extendSquareRowY(ext_box_coords, field_space, box_params, dupes=6):
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
    x_peri, x_peri_shift, y_peri, y_peri_shift = box_params

    # find absolute y spacing and define shifts
    y_shift = field_space + y_peri
    x_shift = x_peri_shift / y_peri * y_shift

    # check extension direction
    if dupes < 0:
        ext_dir = 1
        dupes *= -1
    else:
        ext_dir = -1

    # construct new list with all extensions
    n_box = []
    for i in range(dupes + 1):
        # # construct mathematical operation matrix
        opr_list = np.transpose([[int(x_shift * ext_dir * i)] * 4, [int(y_shift * ext_dir * i)] * 4])
        n_box.append(ext_box_coords + opr_list)

    # fix output list shape
    shape_box = []
    for i in n_box:
        for j in i:
            shape_box.append(j)

    return shape_box


# ::::: PROCESS : EXTENDING SQUARES IN X
def extendSquare(box, md_scalar):

    # define NTSA dimensions
    mean_scale = np.mean(md_scalar)  # find mean scale from x and y scalebar
    NTSA_len = 20 * 10e-3 * mean_scale
    field_len = 2.1 * 10e-3 * mean_scale
    field_sep = .4* 10e-3 * mean_scale

    row_box, box_prm = extendSquareX(box, field_sep, dupes=7)

    # compute field separation correction from the actual mask row length and redo the linear mask
    x_vals = np.array([])
    for arr in row_box:  # separate all x values
        xs = arr[:, 0]
        x_vals = np.concatenate((x_vals, xs), 0)
    x_vals.sort()  # sort from low to high
    masked_NTSA_len = x_vals[-2] - x_vals[0]  # find initial linear mask length
    mask_scalar = masked_NTSA_len / (NTSA_len - field_sep)
    field_sep *= mask_scalar

    row_box, box_prm = extendSquareX(box, field_sep, dupes=7)

    pos_col_box = extendSquareRowY(row_box, field_sep, box_prm, dupes=1)
    neg_col_box = extendSquareRowY(row_box, field_sep, box_prm, dupes=-6)
    tbox = pos_col_box + neg_col_box[8:]

    return tbox


# ::::: PROCESS : CREATING STRUCTURE MASK IMAGE
def createMaskImg(img, cont, out_dir):

    # check path validity
    dir_split = out_dir.split('\\')
    man_ctrl_dir = r'{}\_masks for manual control'.format('\\'.join(dir_split[:-1]))
    file_name = dir_split[-1]
    writing_dirs = (out_dir, man_ctrl_dir)
    for d in writing_dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # draw contours onto the image
    cv2.drawContours(img,cont,-1,(0,255,255),2)
    cv2.imwrite(out_dir + r"\StructureMask.tiff", img)

    # create and write control mask
    ctrl_mask = cv2.resize(img, (int(img.shape[1] * .13), int(img.shape[0] * .13)), interpolation=cv2.INTER_AREA)
    ctrl_mask = cv2.convertScaleAbs(ctrl_mask, alpha=2, beta=20)  # contrast enhance and brighten ctrl image
    cv2.imwrite(rf"{man_ctrl_dir}\{file_name}.png", ctrl_mask)
