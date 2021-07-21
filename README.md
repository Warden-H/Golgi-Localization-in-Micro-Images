# Golgi-Localization-in-Micro-Images

There are 3 folders included.

Folder 'Pre-processing':
>Includes the code for image data preprocessing by steps as:
>>1. Image Generation: generate the sample image based on the process on ImageJ
>>2. ROI Extraction: to extract ROI data from the zip files extracted from ImageJ
>>3. Generate Objectness Image: to mark all the objects in the image
>>4. Update Golgi Coordinates: to stretch the ROI coordinates on Golgi image to perfectly fit the ROI
>>5. Extract Object Coordinates: to extract the ROI coordinates for all the objects in images.

Folder 'Model':
>This folder includes the code for model construction & training for different model version, and the prediction test code.

Folder 'FinalDelivery':
>This folder includes the final delivered files to lab for real application. It includes:
>>1. PredictROI.py: to extract all the objects & classify all the objects, and save all the predicted Golgi object into csv fils.
>>2. CSV_to_ROI.txt: the program to help on transforming the csv coordinates file into real ROI with the help of ImageJ
  
For more details, please refer to the report 'Developing Analytical Tool for Golgi Imaging - BMDSIS Project Report.pdf'.
