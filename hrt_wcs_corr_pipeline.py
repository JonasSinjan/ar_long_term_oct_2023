import os
from reproject_funcs import check_if_ic_images_exist, get_hrt_wcs_crval_err
from download_all_hmi_files import get_hrt_earth_datetimes, get_list_files
import json

class CorrectHRTWCSPipe:

    def __init__(self,hrt_input_folder,hmi_input_folder,output_folder,hrt_input_file_series,hmi_target_file_series):
        self.hrt_input_folder=hrt_input_folder
        self.hmi_input_folder=hmi_input_folder
        self.output_folder=output_folder
        self.hrt_input_file_series=hrt_input_file_series #'blos','icnt'
        self.hmi_target_file_series=hmi_target_file_series #'hmi.m_45s, hmi.m_720s, hmi.ic_45s, hmi.ic_720s'

        # load all the input and target files, check if continuum intensity images exist for HRT WCS corrections
        if self.hrt_input_file_series is 'blos':
            try:
                self._check_all_files_for_icnt() #check if continuum intensity images exist
            except:
                print('Cannot locate continuum intensity images for HRT WCS corrections')

        elif self.hrt_input_file_series is 'icnt' and 'ic_' in self.hmi_target_files_series:
            self.get_all_hrt_files()
            self.get_all_hmi_files()
            self.check_if_equal_hrt_hmi_files()
            self.hrt_wcs_corr_files = [self.hrt_input_folder + f for f in self.hrt_files]
            self.hmi_wcs_corr_files = [self.hmi_input_folder + f for f in self.hmi_files]

    def _check_all_files_for_icnt(self):
        """check if the corresponding continuum intensity images for all input HRT files and HMI files exist"""
        self.get_all_hrt_files()
        self.get_all_hmi_files()
        self.check_if_equal_hrt_hmi_files()
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
        self.number_hrt_files = len(self.hrt_files)

    def get_all_hmi_files(self):
        """get list of desired HMI files in input folder"""
        self.hmi_files=get_list_files(self.hmi_input_folder,self.hmi_ref_file_series, 'hmi')
        self.number_hmi_files = len(self.hmi_files)

    def check_if_equal_hrt_hmi_files(self):
        """check if the number of HRT and HMI files are equal"""
        if self.number_hrt_files != self.number_hmi_files:
            raise ValueError(f'Number of HRT and HMI files are not equal\n\
                             HRT files: {self.number_hrt_files}\n\
                             HMI files: {self.number_hmi_files}')

    def calc_and_write_hrt_WCS_corrections(self):
        """get WCS CRVAL (and CRPIX) error in HRT maps using HMI as reference, using the continuum intensity images"""
        self.hrt_CRVAL_corrections = {}
        self.hrt_CRPIX_corrections = {}
        
        for files in zip(self.hrt_wcs_corr_files,self.hmi_wcs_corr_files):
            crval_err,crpix_err = get_hrt_wcs_crval_err(files[0],files[1],save_crpix_err=True)
            DID = str(files[0].split('.fits')[0].split('_')[-1])
            self.hrt_CRVAL_corrections[DID]=crval_err
            self.hrt_CRPIX_corrections[DID]=crpix_err
        
        print(f'Writing HRT WCS corrections to json file in output folder: {self.output_folder}')
        with open(self.output_folder + 'hrt_CRVAL_corrections.json','w') as f:
            json.dump(self.hrt_CRVAL_corrections,f)
        with open(self.output_folder + 'hrt_CRPIX_corrections.json','w') as f:
            json.dump(self.hrt_CRPIX_corrections,f)