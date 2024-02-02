# Cellexum
**Cell**a **ex**plicatio amminicul**um** or cell analysis tool, is a specialized low-level program made to analyze cell counts along with some supporting information on fluorescence images of Nanotopography Screening Array (NTSA) interfaces, which is a set of different surface topographies on a single sample developed by M. Foss et al. (personal communication, 2023).
The program is made to analyze images directly taken from an Olympus optical microscope with cellSens (i.e. in .vsi file format).

The detailed mechanics of the script can be found in [Technical Details](https://github.com/nicholas12392/cellexum#technical-details).

## Operating the Program
The script consists of six different files that each have a specific function.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/ea9a9e43-553a-4dd1-95f2-c831db4bd699"><br>
  Figure 1: <b>Program file overview.</b> All files will be explained later in more detail.
</p>

### Initializing Program
Once fluorescence images have been obtained and it is time to process them and get cell counts, simply launch the script by double-clicking the RUN.bat file. Once the script is running, it will start by adding the current path to PATH, whereafter it prompts for the location of the fluorescence images.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/bdc7ae82-3f90-4668-9537-dd02dec592d8"><br>
  Figure 2: <b>Initializing script with target path.</b>
</p>

### Optimal Preprocessing
When the image path is given, press enter. If there is currently no output directory, the script will create one.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/306cbb7d-237a-481d-81d7-fb42bdb6a320"><br>
  Figure 3: <b>Creating missing output path.</b>
</p>
Secondly, if the microscoper script has already been run, and the necessary files have been generated, the program will start preprocessing the images by rotating, masking, and slicing. Each major step in this process is marked along with some smaller steps, so that it is clear what the script does and when.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/01e21fc3-8459-4cd4-abc4-3740ead5e54f"><br>
  Figure 4: <b>Preprocessing image data.</b> Note that in newer versions sligtly more information shown than what is in the image, yet these are the major steps.
</p>
Note that this is simply a single bar being continuously updated in the program. Here each update is imaged to better illustrate what is going on. Furthermore, by default, the script will first attempt to employ LFSR with MOFM, and if that fails it moves to BSR with MOFM. In newer versions an additional bar marking smaller steps will sometimes be visible below the primary bar.

### Missing Files Generation (microscoper)
If on the other hand, files are missing, the microscoper script by pskeshu will be run, and the missing files will be generated.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/983698d3-0dd3-4d89-abf3-c323716c2828"><br>
  Figure 5: <b><a href="https://github.com/pskeshu/microscoper">microscoper</a> script creating missing files.</b>
</p>
Once the microscoper script has finished, the program will continue as expected.

### Select Images for Reprocessing or Continue
For each fluorescence image in the folder, the preprocessing will be done in the sequence from before. Once all images have 
been through, the script will note that it is done and it is then possible to redo preprocessing for specific images if they are not 
to a satisfactory quality.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/b1e10bda-2ab0-4b32-9ec1-1573c9b766ad"><br>
  Figure 6: <b>First manual control stop.</b>. At this point, check that all the masks are of sufficient quality.
</p>
In the example above, once you press enter, the images ‘0’, ‘3’, and ‘7’ will be redone.

Following, you will have to choose which method of preprocessing you wish to attempt. If you select only one image, you can 
specifically set the angular compensation with the LFSR method.
<p align="center">
  <img width="400vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/8c07c27d-b0dd-4e38-b2e4-36c695e29be8"><br>
  Figure 7: <b>List of analysis methods to choose from</b> if only a single image was selected.
</p>
Both options ‘0’ and ‘1’ will use MOFM for masking, whereas the last option will use the forced square method as emphasized. 
When this is done, the preprocessing will begin, and when done, the script will prompt again with the option to redo. If all 
mappings are good, simply press enter to continue.

If you select more than one image, you will only have the following options and the specific option will be applied to all images 
selected for reprocessing.
<p align="center">
  <img width="400vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/536cdcbe-8eee-4433-90e7-0c610da88a23"><br>
  Figure 8: <b>List of analysis methods to choose from</b> if multiple images were selected.
</p>
Note that both options ‘0’ and ‘1’ will use MOFM for masking, whereas the last option will use the forced square method as 
emphasized. When this is done, the preprocessing will begin, and when done, the script will prompt again with the option to 
redo. If all mappings are good, simply press enter to continue.

### Manual Mask Control
Note that a new directory will have been created in the parent directory named ‘_masks for manual control’, wherein there are
downscaled contrast-enhanced masks, which can be swiftly checked. If there is any doubt a high-resolution copy of the mask 
without enhancements is always saved in the child directory for the current image.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/06394c0b-d7a6-4db7-ba8a-7c6eb431c4b0"><br>
  Figure 9: <b>Created path and manual control file.</b>. Start by checking the overview mask, and if there is any doubt check the full resolution mask.
</p>
The meaning of T# is the measure of how strictly it was possible to fit the mask. This is measured in tolerance levels. The best tolerance is T0, which is the initial fit condition of an allowed error on the area of +/-3% and the height/width ratio of +/-0.001. Only contours that have more than 10% infill are saved. The T0 data will always be used to determine the angular correction of the masked arrays since the angular offset is most accurate the more precise the mask fits the arrays. If the T0 dataset contains less than 7 arrays, the script will up the tolerance with +/-0.5% area and +/-0.001 ratio and fit again. This will continue until the criteria is passed. The tolerance just above (or at) the criteria level, is then used to determine the average distance between arrays in the mask, to get the most accurate positional representation. Thus, in the control image, the green (left) tolerance is the best tolerance for which arrays could be fitted, which has been used to determine the angular offset, and the red (right) tolerance is the worst tolerance (or the criterium) tolerance used to fit the array positions.
The determined field parameters signify the average pixel height and width of each array and 'A', the horizontal angular offset.

### LFSR Failure
There are a few reasons for which LFSR might fail. If this occurs, the program will attempt to go through masking with BSR.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/8375986a-acab-4a64-bb5c-1be7eebc10e6"><br>
  Figure 10: <b>LFSR failure example.</b>
</p>
This will then be done in the sequence mentioned earlier. Note that if this occurs, it is very important to check that the rotation 
did indeed occur correctly since this method is less robust than the LFSR method. If the orientation has indeed failed, it is
advised to use enhanced BSR.

Currently, the options from the failed LFSR attempt are not inherited.

### Cell Counting
Once the mask has been done, when continuing the script will start counting the cells from all the cut UV images.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/f37de0ea-0b55-4061-9b4d-baa95a8a4faf"><br>
  Figure 11: <b>Cell count determination step.</b>
</p>
For each count, an additional UV image is saved so that if desired it is possible to check the cell count manually against the 
determined count. Note that if CTC is used the counts are red and if HCC they are green.
<p align="center">
  <img width="700vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/4110062a-a320-4ae5-b9ae-1456dbbfc99d"><br>
  Figure 12: <b>Cell count checking.</b>
</p>
At last, when all cells have been counted, an Excel file containing all the counts mapped to the field type is saved for further
processing. Additionally, an extra file is created. This file contains the cell count from area counting, which is the total cell area 
for a field, divided by the estimated size of a cell nucleus. The area count is usually unreliable, but can in very specific cases be 
the best option. The two counts may, however, in many cases be the same.
<p align="center">
  <img width="550vw" src="https://github.com/nicholas12392/cellexum/assets/87773847/81cc42a3-691f-4d95-8867-0601f6957356"><br>
  Figure 13: <b>Output cell count files</b>
</p>

## Technical Details 
