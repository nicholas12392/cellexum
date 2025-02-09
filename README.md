# Cellexum
**Cell**a **ex**plicatio amminicul**um** or cell analysis tool, is a program for analyzing cells on fluorescence images taken on e.g. Olympus microscopes.

If you have multiple python installations on your machine, you will have to edit the `python.exe` in the `RUN.bat` file to the path of your target python installation.

## Quick Start Guide
### Loading in image files
When the program has been loaded through the bat file it will automatically start in the 'File Selection' menu.
The program takes an entire folder with .vsi images. Click on the directory entry to select a local folder. The 'SELECT ALL' and 'DESELECT ALL' buttons allow quick toggling of all files. This selection menu acts to reduce clutter for later stages of the program. At any time can a deselected file be reselected and processed. 

### Before looking for cells (Preprocessing)
The program features methods for identifying specific rectangular fields that are important on the surface that has been imaged. For example, an imaged surface that contains unique fields that are of interest can be masked for those fields before counting the cells. This way, it is possible to get unique data analysis for each field for elaborate data analysis. 
The program will automatically set an output folder, once the input folder has been set. The output folder can be changed at any time, but analysis progress made in one output folder will not be transferred automatically to the new output folder.
It is possible to change all file settings related to a specific setting by changing the specific setting, and then control-clicking on the changed field. 

#### Sample types
This option is set depending on whether the image contains a surface that has more than one field or just a single field. Depending on the choice, the rest of the preprocessing settings will change.
The important change between the two options is that for multi-field surfaces, it is possible to create orientation configurations, along with custom masks that will be used for data presentation. This way, it is not necessary to rotate all images prior to analysis or exporting all the data generated through the program to do plotting afterward. 
If there exists no image mask, add a mask by pressing on the image mask selection menu and press 'Add ...'. This will open a mask creator window where the parameters for the field mask can be configured. After saving the mask, it will automatically be selected in the main frame. If the mask channel has also been set, it will now be possible to create an orientation reference. 

#### Mask channel
Here the color channel of the image for which the fields to be masked are visible is chosen. 

#### Orientation reference
The orientation reference holds information about how the image 'type' looks when it is rotated correctly. To use this feature, select a reference file and wait for the necessary elements to be generated. The window will automatically update once resources are ready. Click on the loaded image to zoom to native resolution, and use the +90 and -90 buttons to rotate the preview image. Once the image has the correct orientation, press on create mask, and wait for the program to map out the mask on the reference image. Once resources are ready, the masked image will be displayed. Check whether the mask has been applied correctly, and close the mask creator if so; otherwise, try altering the parameters or selecting a different file. 
Note that it is required to set up an orientation reference if the automatic rotation determination is to be used.

### Quality control (Mask Gallery)
After the field masks are created, they will start popping up in the mask gallery. Here it is possible to see different fit parameters for the mask and a low-res and hi-res version of the mask can be opened directly from the GUI (Windows only). If some of the masks are wrong, go back to the preprocessing menu, change some parameters, and retry the masking. 

### Identifying the cells (Processing)
Once the field masks have been properly configured, the images can be processed for identifying cells. The cell type is not too important and only acts as an upper bound for nuclei size. Currently, there are not many options, so simply leave it at fibroblast, regardless of the cell type on the image. The channel is the image color channel for which the cell nuclei are visible.
There are different counting methods, but usually, the CCA (connected component analysis) will be the most optimal. 

### Analysis
After processing, the images can be analyzed. The analysis menu features an auto-grouping function, which can automatically group the input data. Currently, the only two analysis options implemented are nuclei analysis and nearest neighbor evaluation. The nuclei analysis will use raw cell counts along with a presentation mask and sorting of the fields (Multi-Field only) to generate cell count plots. The nearest neighbor evaluation allows for a thorough analysis of the nearest neighbor distances on the images. 
If the sample type is Multi-Field, it is also possible to set up control fields for the particular field mask. Specifically, these fields will be used as internal controls. They will first have their cell counts averaged (if there are multiple control fields), whereafter cell counts for all fields will be divided with that average. 

### Application Settings
Clicking on the settings wheel in the bottom left corner will bring up the application settings menu. Here it is possible to set different parameters for the program along with deleting created masks. The important settings to note are the maximum CPU settings. This relates to the max_workers in concurrent.futures.ProcessPoolExecutor. Although your local system may have enough CPU power, depending on the size of the opened image RAM may become an issue. If memory errors occur, lower the maximum CPU.
