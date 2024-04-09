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
from skimage.transform import warp, AffineTransform
%matplotlib inline



class experiment:
    def __init__(self,path,time_res,pixel_res,nbleach,sigma):
        self.file_name = path
        self.nbleach = nbleach
        if type(path)==str:
            self.img = imread(self.file_name)
        else:
            self.img=path
        self.nframes = np.shape(self.img)[0]
        self.time_res,self.pixel_res,self.nbleach = time_res,pixel_res,(nbleach-1)
        self.bleach_img = ndi.gaussian_filter(self.img[self.nbleach,:,:],sigma)
        self.prebleach_img = ndi.gaussian_filter(self.img[(self.nbleach-1),:,:],sigma)

    #### IMAGE PREPARATION ####
    def image_preparation(self):
        #normalization
        self.img=self.img/self.img.max()
        #adjustment
        img_adapteq=[]
        img_gamma =[]
        for im in self.img:
            
            p2, p98 = np.percentile(im, (2, 98))
            img_rescale = exposure.rescale_intensity(im, in_range=(p2, p98))
            
            # Adaptive Equalization
            img_adapteq.append(exposure.equalize_adapthist(im, clip_limit=0.03))
            
            #Gamma adjustment --> on rescaled images
            img_gamma.append(exposure.adjust_gamma(img_rescale,gamma=2))
    
        self.img_cellseg=np.array(img_adapteq)
        self.img_nucseg=np.array(img_gamma)
  
    #### IMAGE SEGMENTATION ####
            
    def segment_nucleus(self,thr_nuc=1,nucsize_min=10000,nucsize_max=300000):  #thresholds based on empirical computation
        self.prebleach_img_nucseg=self.img_nucseg[(self.nbleach-1),:,:]
        
        #thresholding
        thresh = threshold_otsu(self.prebleach_img_nucseg)*thr_nuc
        self.nucmask = self.prebleach_img_nucseg > thresh
    
        #binary closing --> to close a little everything
        self.nucmask = ndi.binary_closing(self.nucmask,iterations=1)
    
        #removing objects with a low area
        self.nucmask=area_opening(self.nucmask,area_threshold=20000,connectivity=100)
    
        #binary closing
        self.nucmask = ndi.binary_closing(self.nucmask,iterations=4)
        self.nucmask = ndi.label(self.nucmask)[0]
        #sizes = [(self.nucmask == x).sum() for x in np.unique(self.nucmask)[1:]]
        labels = measure.label(self.nucmask)
        masks = [(labels == i) for i in np.unique(labels) if i != 0]
        #if multiple nuclei are present, select only the most decreasing after bleach
        ratio_min=1
        idx=100000
        for i in range(len(masks)):
            size=len(self.bleach_img[masks[i]!=0])
            ratio=np.mean(self.bleach_img[masks[i]!=0])/np.mean(self.prebleach_img[masks[i]!=0])
            if size>=nucsize_min:
                if ratio< ratio_min:
                    ratio_min=ratio
                    idx=i
        self.nucmask[~masks[idx]]=0
        self.nucmask[self.nucmask>0.5]=1

    def segment_cell(self,img=1,thr_cell=0.5,cellsize_min=60000,cellsize_max=1000000):    #thresholds based on empirical computation
        
        if type(img)!=int:
            self.prebleach_img_cellseg=img
        else:
            self.prebleach_img_cellseg=self.img_cellseg[(self.nbleach-1),:,:]
            
        thresh = threshold_otsu(self.prebleach_img_cellseg)*thr_cell
        self.cellmask = self.prebleach_img_cellseg > thresh
        #removing objects with a low area
        self.cellmask=area_opening(self.cellmask,area_threshold=30000,connectivity=100)
        #fill holes
        self.cellmask = ndi.binary_fill_holes(self.cellmask)
        #binary closing
        self.cellmask = ndi.binary_closing(self.cellmask,iterations=3)
        self.cellmask = ndi.label(self.cellmask)[0]
        #if identified cell is actually the nucleus, repeat the pipeline with a lower alpha
        if np.nonzero(self.nucmask)[0].shape[0]/np.nonzero(self.cellmask)[0].shape[0]>0.65:    #threshold defined empirically
            thr_cell=thr_cell-0.05
            self.segment_cell(thr_cell=thr_cell,cellsize_min=cellsize_min,cellsize_max=cellsize_max)
        #remove objects with area ower than cellsize min
        self.cellmask[self.cellmask>0.5]=1
        sizes = [np.nonzero(self.cellmask == x)[0].shape[0] for x in np.unique(self.cellmask)[1:]]
        for id,size in zip(np.unique(self.cellmask)[1:],sizes):
            if size < cellsize_min:
                self.cellmask[self.cellmask==id] = 0
        return self.cellmask

    
    def getBleachedmask(self,sigma_b=2,beta=0.5,extra_pix=2,shift_x=0,thr_bleach=2.5,thr_nonbleach=1.5):
        # loop through the mask of condensates
        self.bleachmask = np.zeros_like(self.nucmask)
        self.nonbleachmask = np.zeros_like(self.nucmask)
        alpha=np.mean(self.bleach_img)/np.mean(self.prebleach_img)
        for id in np.unique(self.nucmask):
            #alpha factor for minimizing effecto of a major mean intensity in the prebleached
            if np.mean(self.prebleach_img[self.nucmask == id])*alpha >= np.mean(self.bleach_img[self.nucmask == id]):  #only if the mean of the prebleach is higher than bleach
                # determine x position of bleach boundary
                temp_im_prebleach = copy.deepcopy(self.prebleach_img)
                temp_im_bleach = copy.deepcopy(self.bleach_img)
                
                temp_mask_prebleach = copy.deepcopy(self.nucmask)
                temp_mask_bleach = copy.deepcopy(self.nucmask)
                
                temp_im_prebleach[self.nucmask != id] = 0
                temp_im_bleach[self.nucmask != id] = 0
                temp_mask_prebleach[self.nucmask != id] = 0
                temp_mask_bleach[self.nucmask != id] = 0

                if len(regionprops(temp_mask_prebleach))==0: # let's say the background changes too much, I want to avoid that my program select the background
                    continue
                
                else:
                    # get radius and eccentricity
                    self.radius = ((regionprops(temp_mask_prebleach)[0].minor_axis_length+regionprops(temp_mask_prebleach)[0].major_axis_length)/2)*self.pixel_res
                    self.eccentricity = regionprops(temp_mask_prebleach)[0].eccentricity
                    #self.cellmask[self.nucmask == id] = 0
        
                    # if the prebleach/bleach has a FC of 2 or higher it will be considered bleach area
                    bleached_area=((temp_im_prebleach+1)/(temp_im_bleach+1))>thr_bleach
                    temp_bl_area=np.where(bleached_area, temp_im_prebleach, 0)
    
                    # if the prebleach/bleach has a FC of 1.5 or lower it will be considered non bleach area
                    non_bleached_area=((temp_im_prebleach+1)/(temp_im_bleach+1))<thr_nonbleach
                    temp_non_bl_area=np.where(non_bleached_area, temp_im_prebleach, 0)
        
                    self.bleachmask[temp_bl_area!=0] = id
                    self.nonbleachmask[temp_non_bl_area!=0] = id
                    return (temp_im_prebleach+1)/(temp_im_bleach+1)

    def ROI_generation(self,ROI_area=0.25,n_ROIs=50):
        side,coords_bl,mask_bl=generate_ROI(self.img[0],self.bleachmask,ROI_area=ROI_area,n_ROIs=n_ROIs)
        _,coords_nbl,mask_nbl=generate_ROI(self.img[0],self.nonbleachmask,ROI_area=ROI_area,n_ROIs=n_ROIs,side=side)
        cellmask_inverse=np.ones_like(self.cellmask)-self.cellmask
        _,coords_nc,mask_nc=generate_ROI(self.img[0],cellmask_inverse,ROI_area=ROI_area,n_ROIs=n_ROIs,side=side)

        self.bleached_roi=mask_bl
        self.nonbleached_roi=mask_nbl
        self.noncell_roi=mask_nc

        self.bleached_roi_coords=coords_bl
        self.nonbleached_roi_coords=coords_nbl
        self.noncell_roi_coords=coords_nc

    #### IMAGES ALIGNMENT
    def cells_identification(self,thr_cell=0.5,cellsize_min=80000,cellsize_max=20000000):
        '''identification of cell profile in each image of the series'''
        cells=[]
        for im in self.img_cellseg:
            cellmask=self.segment_cell(img=im,thr_cell=0.5,cellsize_min=60000,cellsize_max=1000000)
#            thresh = threshold_otsu(im)*thr_cell
#            cellmask = im > thresh
            #removing objects with a low area
#            cellmask=area_opening(cellmask,area_threshold=30000,connectivity=100)
            #fill holes
#            cellmask = ndi.binary_fill_holes(cellmask)
            #binary closing
#            cellmask = ndi.binary_closing(cellmask)
#            cellmask = ndi.label(cellmask)[0]
#            sizes = [(cellmask == x).sum() for x in np.unique(cellmask)[1:]]
            #remove objects with area ower than cellsize min
#            for id,size in zip(np.unique(cellmask)[1:],sizes):
#                if size < cellsize_min:
#                    cellmask[cellmask==id] = 0
#            im[cellmask == 0]=0
            cells.append(cellmask)
        self.iso_cells=np.array(cells)       
        return thr_cell

    
    def mask_alignment(self):
        aligned_bleached_mask=[]
        aligned_non_bleached_mask=[]
        aligned_nuc_mask=[]
        aligned_bleached_roi=[]
        aligned_nonbleached_roi=[]
        aligned_noncell_roi=[]
        
        for idx, img in enumerate(self.iso_cells):
            # Translate the cell to the center using AffineTransform
            if idx == 0:
                translation = find_optimal_translation(img,idx)
                centered_bm_0 = warp(self.bleachmask, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_nbm_0 = warp(self.nonbleachmask, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_nm_0 = warp(self.nucmask, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_br_0 = warp(self.bleached_roi, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_nbr_0 = warp(self.nonbleached_roi, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_nc_0 = warp(self.noncell_roi, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                aligned_bleached_mask.append(self.bleachmask)
                aligned_non_bleached_mask.append(self.nonbleachmask)
                aligned_nuc_mask.append(self.nucmask)
                aligned_bleached_roi.append(self.bleached_roi)
                aligned_nonbleached_roi.append(self.nonbleached_roi)
                aligned_noncell_roi.append(self.noncell_roi)
            else:
                translation = find_optimal_translation(img,idx)
                centered_bm = warp(centered_bm_0, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_nbm = warp(centered_nbm_0, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_nm = warp(centered_nm_0, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_br = warp(centered_br_0, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_nbr = warp(centered_nbr_0, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                centered_nc = warp(centered_nc_0, translation.inverse, output_shape=(img.shape[0], img.shape[1]),preserve_range=True)
                aligned_bleached_mask.append(centered_bm)
                aligned_non_bleached_mask.append(centered_nbm)
                aligned_nuc_mask.append(centered_nm)
                aligned_bleached_roi.append(centered_br)
                aligned_nonbleached_roi.append(centered_nbr)
                aligned_noncell_roi.append(centered_nc)
            
        aligned_bleached_mask=np.array(aligned_bleached_mask)
        aligned_non_bleached_mask=np.array(aligned_non_bleached_mask)
        aligned_nuc_mask=np.array(aligned_nuc_mask)
        aligned_bleached_roi=np.array(aligned_bleached_roi)
        aligned_nonbleached_roi=np.array(aligned_nonbleached_roi)
        aligned_noncell_roi=np.array(aligned_noncell_roi)
        self.bleached_aligned=aligned_bleached_mask
        self.nonbleached_aligned=aligned_non_bleached_mask
        self.nucmask_aligned=aligned_nuc_mask
        self.bleached_roi_aligned=aligned_bleached_roi
        self.nonbleached_roi_aligned=aligned_nonbleached_roi
        self.noncell_roi_aligned=aligned_noncell_roi
        

    ### IMAGE ANALYSIS
    def analysis_basics(self,norm=True):
        recovery_bleach=[]
        non_bleach=[]
        for i in range(self.img.shape[0]):
            recovery_bleach.append((np.mean(self.img[i][self.bleached_roi_aligned[i]!=0])-np.mean(self.img[i][self.noncell_roi_aligned[i]!=0]))/(np.mean(self.img[i][self.nonbleached_roi_aligned[i]!=0])-np.mean(self.img[i][self.noncell_roi_aligned[i]!=0])))
            non_bleach.append((np.mean(self.img[i][self.nonbleached_roi_aligned[i]!=0])-np.mean(self.img[i][self.noncell_roi_aligned[i]!=0])))
        if norm==True:
            recovery_norm=[x/max(recovery_bleach) for x in recovery_bleach]
            non_bleach_norm=[x/max(non_bleach) for x in non_bleach]
        else:
            recovery_norm=recovery_bleach
            non_bleach_norm=non_bleach
        time_points=[0]
        count=0
        for i in range(10):
            count=count+1.47
            time_points.append(count)
        for i in range(10,self.img.shape[0]-1):
            count=count+6.47
            time_points.append(count)
        time_points=time_points[1:]
        recovery_norm=recovery_norm[1:]
        non_bleach_norm=non_bleach_norm[1:]
        df=pd.DataFrame([recovery_norm+non_bleach_norm,time_points+time_points,['recovery' for x in range(len(recovery_norm))]+['non bleached' for x in range(len(non_bleach_norm))]],index=['FRAP (norm intensities)','t (s)','method'])
        return df.T
