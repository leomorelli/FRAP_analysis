import matplotlib.pyplot as plt
from skimage.io import imread
from readlif.reader import LifFile
import scipy.ndimage as ndi
from scipy.stats import ttest_ind_from_stats
from skimage.filters.thresholding import threshold_otsu
from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage.morphology import area_opening
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation as register_translation
from scipy import signal
from decimal import Decimal
import numpy as np
import copy
from skimage import measure
import tifffile
import random
import os
import seaborn as sns
import matplotlib
import pandas as pd
from skimage import img_as_float
from skimage import exposure
from tifffile import imwrite
from skimage import io, color
from skimage.transform import AffineTransform
%matplotlib inline


def plot_mask(raw,mask,mask2=None,mask3=None):
  # create edges array
  edges1 = np.zeros_like(mask)
  edges1 = np.ma.masked_where(edges1 == 0, edges1)
  for ID in np.unique(mask)[1:]:
    cell_mask = mask==ID
    eroded_cell_mask = ndi.binary_erosion(cell_mask, iterations=1)
    edge_mask = np.logical_xor(cell_mask, eroded_cell_mask)
    edges1[edge_mask] = ID
  try:
    edges2 = np.zeros_like(mask2)
    edges2 = np.ma.masked_where(edges2 == 0, edges2)
    for ID in np.unique(mask2)[1:]:
      cell_mask = mask2==ID
      eroded_cell_mask = ndi.binary_erosion(cell_mask, iterations=1)
      edge_mask = np.logical_xor(cell_mask, eroded_cell_mask)
      edges2[edge_mask] = ID
    edges2 = np.ma.masked_where(edges2 == 0, edges2)
  except:
    pass
  try:
    edges3 = np.zeros_like(mask3)
    edges3 = np.ma.masked_where(edges3 == 0, edges3)
    for ID in np.unique(mask3)[1:]:
      cell_mask = mask3==ID
      eroded_cell_mask = ndi.binary_erosion(cell_mask, iterations=1)
      edge_mask = np.logical_xor(cell_mask, eroded_cell_mask)
      edges3[edge_mask] = ID
    edges3 = np.ma.masked_where(edges3 == 0, edges3)
  except:
    pass
  #plot

  plt.figure(figsize=(7,7))
  plt.imshow(raw, interpolation='none', cmap='gray')
  plt.imshow(edges1,interpolation='none',cmap='autumn',alpha=1)
  try:
      plt.imshow(edges2,interpolation='none',cmap='spring',alpha=1)
  except:
        pass
  try:
      plt.imshow(edges3,interpolation='none',cmap='cool',alpha=1)
  except:
    pass
  plt.show()

def generate_mask(side, top_left_x, top_left_y, image_shape):
    """
    Generate a binary mask with the specified region filled with 1s and the rest with 0s.

    Parameters:
        height (int): Height of the region.
        width (int): Width of the region.
        top_left_x (int): X-coordinate of the top-left corner of the region.
        top_left_y (int): Y-coordinate of the top-left corner of the region.
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        mask (ndarray): Binary mask with the specified region filled with 1s and the rest with 0s.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[top_left_y:top_left_y + side, top_left_x:top_left_x + side] = 1
    return mask

def generate_ROI(image,mask,ROI_area=0.25,n_ROIs=50,side=0):
    image=image
    mask=mask
    num_regions=n_ROIs
    fraction_area_ROI=ROI_area/num_regions
    
    roi_coords = np.argwhere(mask)
    roi_x_coords=list(set([x[1] for x in roi_coords]))
    roi_y_coords=list(set([y[0] for y in roi_coords]))

    mask_value=[x for x in np.unique(mask) if x!=0][0]
    area_mask=sum(sum(mask))/mask_value
    area_regions=area_mask*fraction_area_ROI
    if side==0:
        side=int(np.sqrt(area_regions))
    else:
        side=side
    
    selected_regions = []
    selected_regions_mask=np.zeros_like(image)

    while len(selected_regions)<num_regions:
        # Randomly select a point within the ROI
        x = np.random.randint(min(roi_x_coords),max(roi_x_coords))
        y = np.random.randint(min(roi_y_coords),max(roi_y_coords))
        #check if the region is inside the ROI
        mask_roi=generate_mask(side,x,y,image.shape)
        values=np.unique(mask[mask_roi==1])
        if len(values)==0:
            values=[0]
        if (len(values)==1)&(values[0]!=0):
        #  check if the new region overlaps with previously identified regions
            val_inside_region=np.unique(selected_regions_mask[y:y + side, x:x + side])
            if (len(val_inside_region)==1)&(val_inside_region[0]==0):
                selected_regions_mask[y:y + side, x:x + side] = 1
                selected_regions.append((x, y, side, side))
    return side,selected_regions,selected_regions_mask

def find_optimal_translation(img,idx):
    regions = regionprops(img)
    max_area = 0
    max_region = None
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            max_region = region
    
    # Calculate the center of the bounding box of the cell
    center_y, center_x = max_region.centroid
    
    # Get image dimensions
    height, width = img.shape
    
    # Calculate the offset needed to move the center to the image center
    offset_x = width // 2 - center_x
    offset_y = height // 2 - center_y

    if idx==0:
        # Translate the cell to the center using AffineTransform
        translation = AffineTransform(translation=(offset_x,offset_y))
    else:
        # Translate the cell to the center using AffineTransform
        translation = AffineTransform(translation=(-offset_x,-offset_y))
    return translation
