import os
from reproject_funcs import get_hrt_remapped_on_hmi
from download_all_hmi_files import get_hrt_earth_datetimes, get_list_files
import json

class ReprojectHRT2HMIPipe:

    def __init__(self,hrt_input_folder,hmi_input_folder,output_folder,hrt_input_file_series,hmi_target_file_series,wcs_corr_folder):
        self.hrt_input_folder=hrt_input_folder
        self.hmi_input_folder=hmi_input_folder
        self.output_folder=output_folder
        self.hrt_input_file_series=hrt_input_file_series #'blos','icnt'
        self.hmi_target_file_series=hmi_target_file_series #'hmi.m_45s, hmi.m_720s, hmi.ic_45s, hmi.ic_720s'

        self.wcs_corr_folder=wcs_corr_folder
        self._check_if_wcs_corrections_exist()

    def _check_if_wcs_corrections_exist(self):
        if os.path.isfile(self.wcs_corr_folder + 'hrt_CRVAL_corrections.json'):
            with open(self.wcs_corr_folder + 'hrt_CRVAL_corrections.json','r') as f:
                self.hrt_CRVAL_corrections = json.load(f)
            self._check_DIDs_in_hrt_CRVAL_corrections()
            self.wcs_corr_exist = True
        else:
            raise OSError('No \'hrt_CRVAL_corrections.json\' in the wcs_corr_folder provided')

    def _check_DIDs_in_hrt_CRVAL_corrections(self):
        """check if all DIDs in HRT files are in the WCS corrections file"""
        DIDs = [str(f.split('.fits')[0].split('_')[-1]) for f in self.hrt_files]
        if set(DIDs) != set(self.hrt_CRVAL_corrections.keys()):
            raise ValueError('DIDs in HRT files are not equal to DIDs in WCS corrections file')

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

    def reproject_all_hrt2hmi(self):
        """reproject all HRT blos (or ic) maps to HMI frame"""
        self._reproject_hrt2hmi()
        self.write_log()


