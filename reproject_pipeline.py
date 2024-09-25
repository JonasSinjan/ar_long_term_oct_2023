# input folder of HRT files
# check if HMI blos and ic files exist (45s or 720s)
# raise error if not, and point to download script
# get wcs error in HRT maps using HMI as reference, using the continuum intensity images
# save all wcs errors in a directory to be reused if reprojecting continuum maps later
# reproject all HRT blos (or ic) maps to HMI frame
# save all reprojected maps in a directory (as pickled sunpy maps)

import os
from reproject_funcs import *
from download_all_hmi_files import get_hrt_earth_datetimes, get_list_files

class ReprojectHRT2HMIPipe:

    def __init__(self):
        self.hrt_input_folder
        self.hmi_input_folder
        self.output_folder
        self.hrt_input_file_series #'blos','icnt'
        self.hmi_ref_file_series #'hmi.m_45s, hmi.m_720s, hmi.ic_45s, hmi.ic_720s'

        if self.hrt_input_file_series is 'blos':
            try:
                self._check_all_files_for_icnt()
            except OSError as e:
                print(e)
                print('Download the missing files using the download script')

        elif self.hrt_input_file_series is 'icnt' and self.hmi_input_files_series.contains('ic_'):
            self.get_all_hrt_files()
            self.get_all_hmi_files()
            self.hrt_wcs_corr_files = [self.hrt_input_folder + f for f in self.hrt_files]
            self.hmi_wcs_corr_files = [self.hmi_input_folder + f for f in self.hmi_files]

    def _check_all_files_for_icnt(self):
        """check if the corresponding continuum intensity images for all input HRT files and HMI files exist"""
        self.get_all_hrt_files()
        self.get_all_hmi_files()
        self.hrt_wcs_corr_files = []
        self.hmi_wcs_corr_files = []
        for i,file in enumerate(self.hrt_files):
            hrt_file = self.hrt_input_folder+file
            hmi_file = self.hmi_input_folder+self.hmi_files[i]
            hrt_icfile,hmi_icfile = check_if_ic_images_exist(hrt_file, hmi_file)
            self.hrt_wcs_corr_files.append(hrt_icfile)
            self.hmi_wcs_corr_files.append(hmi_icfile)

    def get_all_hrt_files(self):
        """get list of desired HRT files in input folder"""
        self.hrt_files=get_list_files(self.hrt_input_folder,self.hrt_input_file_series, 'hrt')

    def get_all_hmi_files(self):
        """get list of desired HMI files in input folder"""
        self.hmi_files=get_list_files(self.hmi_input_folder,self.hmi_ref_file_series, 'hmi')

    def get_hrt_CRVAL_corrections_skycoord(self):
        """get WCS CRVAL error in HRT maps using HMI as reference, using the continuum intensity images"""
        self.hrt_CRVAL_corrections = {}
        for files in zip(self.hrt_wcs_corr_files,self.hmi_wcs_corr_files):
            errx,erry = get_hrt_wcs_crval_err(files[0],files[1])
            DID = str(files[0].split('.fits')[0].split('_')[-1])
            self.hrt_CRVAL_corrections[DID]=(errx,erry)
    
    def get_hrt_WCS_corrections(self):
        """get WCS CRVAL (and CRPIX) error in HRT maps using HMI as reference, using the continuum intensity images"""
        self.hrt_CRVAL_corrections = {}
        self.hrt_CRVAL_corrections = {}
        for files in zip(self.hrt_wcs_corr_files,self.hmi_wcs_corr_files):
            crval_err,crpix_err = get_hrt_wcs_crval_err(files[0],files[1],save_crpix_err=True)
            DID = str(files[0].split('.fits')[0].split('_')[-1])
            self.hrt_CRVAL_corrections[DID]=crval_err
            self.hrt_CRPIX_corrections[DID]=crpix_err


    # def get_hrt_files_dates(self):
    #     """get datetime list of DATE-EAR in HRT files"""
    #     self.hrt_dates=get_hrt_earth_datetimes(self.hrt_input_folder)

