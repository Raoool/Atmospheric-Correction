import numpy as np
import pandas as pd
from osgeo import gdal
import math
import rasterio
from matplotlib import pyplot as plt

# Function to read Landsat metadata
def read_landsat_metadata(metadata_file):
    metadata = {}
    with open(metadata_file, 'r') as f:
        lines = f.readlines()
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('GROUP'):
                current_section = line.split('=')[1].strip()

                metadata[current_section] = {}
            elif line.startswith('END_GROUP'):
                current_section = None
            elif '=' in line and current_section is not None:
                key, value = line.split('=')
                metadata[current_section][key.strip()] = value.strip()
    return metadata

## Automated Dark Object Identification
# Define file paths
file_path_base = '/data/raw/LC08_L1TP_019031_20160612_20200906_02_T1/'
metadata_file = file_path_base + 'LC08_L1TP_019031_20160612_20200906_02_T1_MTL.txt'

# Read Landsat metadata
landsat_metadata = read_landsat_metadata(metadata_file)

# this code is able to find continous valid dark object
# Open the raster file using rasterio
file_path = file_path_base + 'LC08_L1TP_019031_20160612_20200906_02_T1_B4.tif'
break_threshold=100
with rasterio.open(file_path) as src:
    # Read the raster data as a NumPy array
    img = src.read(1)

# Mask zero values
masked_array = np.ma.masked_equal(img, 0)

# Calculate histogram of non-zero values
hist, bin_edges = np.histogram(masked_array.compressed(), bins=65535)

# Find the first bin index where the difference between consecutive bins exceeds the break_threshold
break_index = np.argmax(np.diff(hist) > break_threshold)

# Find the lowest valid DN by finding the minimum value within the range of the histogram before the break
if break_index > 0:
    # Find the minimum value within the range of the histogram before the break
    lowest_valid_dn = np.min(masked_array.compressed()[masked_array.compressed() <= bin_edges[break_index]])

    # Find the second lowest valid DN within a range after the minimum DN
    second_lowest_valid_dn = np.max(masked_array.compressed()[
        (masked_array.compressed() > lowest_valid_dn) &
        (masked_array.compressed() <= lowest_valid_dn + 500)  # Adjust this range as needed
    ])
    
    print('Break Found:', second_lowest_valid_dn)
else:
    # If no break is found, consider the entire histogram range
    lowest_valid_dn = np.min(masked_array.compressed())
    print('No Breaks',lowest_valid_dn)

# Find the indices where the array matches the second lowest valid DN
indices = np.where(masked_array == second_lowest_valid_dn)
row_indices, col_indices = indices
print(row_indices, col_indices)


darkest_pixel_position = (row_indices[0],col_indices[0])
print(darkest_pixel_position)
dark_pixel_values_all_bands = []

# Iterate over other bands
for band_number in range(1, 8):
    file_path_band = file_path_base + 'LC08_L1TP_019031_20160612_20200906_02_T1_B{}.tif'.format(band_number)
    with rasterio.open(file_path_band) as src:
        band_img = src.read(1)
    # Get the pixel value at the same position in this band
    darkest_pixel_value = band_img[darkest_pixel_position]
    dark_pixel_values_all_bands.append(darkest_pixel_value)

# Inserting 0 value at 0 place
dark_pixel_values_all_bands.insert(0, 0)

print("Dark Pixel Values of All Bands:", dark_pixel_values_all_bands)


## Converting TOA Reflectance to Surface Reflectance
# Surface Reflectance calulation - DOS (Dark Object Subtraction) method using Lowest DN level subtraction

Sol_Zen = 90 - float(landsat_metadata['IMAGE_ATTRIBUTES']['SUN_ELEVATION'])
print("Solar Zenith Angle:", Sol_Zen)

# Initializing an empty list to store Sur_ref arrays for all bands
sur_ref_list = []
L_Sat_list = []
L_sat_haze_deducted_list = []
TOA_Refl_list = []

# Initializing shape variable
shape = None

# Iterating over bands and perform calculations
for band_number in range(1, 8):
    # File path for the current band
    file_path_band = file_path_base + 'LC08_L1TP_019031_20160612_20200906_02_T1_B{}.tif'.format(band_number)
    #print(file_path_band)

    refl_mult_band = float(landsat_metadata['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_' + str(band_number)])
    refl_add_band = float(landsat_metadata['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_' + str(band_number)])
    #print('REFLECTANCE_MULT_BAND_{}:'.format(band_number), refl_mult_band) 
    #print('REFLECTANCE_ADD_BAND_{}:'.format(band_number), refl_add_band) 
    # Read GeoTIFF image for the current band
    with rasterio.open(file_path_band) as src:
        image_dn = src.read(1)
        band_shape = image_dn.shape
        
        # Ensure all bands have the same shape
        if shape is None:
            shape = band_shape
        elif shape != band_shape:
            raise ValueError("All bands must have the same shape")

    DN_1percent = ((0.01 * math.cos(Sol_Zen)) - refl_add_band) / refl_mult_band
    print('DN_1percent_{}:'.format(band_number), DN_1percent) 
    
    Haze_DN = dark_pixel_values_all_bands[band_number] - DN_1percent
    print('Haze_DN_{}:'.format(band_number), Haze_DN) 
    print('DO_DN_{}:'.format(band_number), dark_pixel_values_all_bands[band_number]) 

    # Create a mask where 0 represents 0 DN values and 1 represents non-zero DN values
    fill_mask = np.where(image_dn == 0, 0, 1)

    TOA_Refl = (((image_dn * refl_mult_band) + refl_add_band) / math.cos(Sol_Zen))  * fill_mask 
       
    # Surface Reflectance calculation - DOS (Dark Object Subtraction)
    if band_number <= 4:
        sur_ref = ((((image_dn - Haze_DN) * refl_mult_band) + refl_add_band) / math.cos(Sol_Zen))  * fill_mask 
    else:
        #LSAT_Deducted = L_Sat
        #sur_ref = (((pi * Distance ** 2) * L_Sat) / (Trans_sur_to_sen * ((getESUN(band_number) * math.cos(Sol_Zen) * Trans_sun_to_sur) + E_down))) * fill_mask
        sur_ref = TOA_Refl * fill_mask

        
    # Append Sur_ref array to the list
    sur_ref_list.append(sur_ref)
    #L_Sat_list.append(L_Sat)
    #L_sat_haze_deducted_list.append(LSAT_Deducted)
    #TOA_Refl_list.append(TOA_Refl)

  # Stack Sur_ref arrays along the band axis
stacked_sur_ref = np.stack(sur_ref_list, axis=0)

# Get metadata from the original GeoTIFF file for band 1
with rasterio.open(file_path_band) as src:
    meta = src.meta.copy()

# Update metadata for the stacked TIFF file
meta.update(count=stacked_sur_ref.shape[0], dtype='float32')

# Output file path for the stacked TIFF file
output_tiff_file = '/processed/LC08_L1TP_019031_L8SR_Dark_DN_Subtracted_Using_GreenBand.tif'

# Write stacked Sur_ref arrays to a new GeoTIFF file
with rasterio.open(output_tiff_file, 'w', **meta) as dst:
    for i, sur_ref_band in enumerate(stacked_sur_ref, 1):
        dst.write(sur_ref_band, i)

print("Stacked Sur_ref TIFF file saved successfully.")
