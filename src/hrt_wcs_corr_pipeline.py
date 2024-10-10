import json
import os
import datetime
from datetime import datetime as dt
from astropy.io import fits

from reproject_funcs import check_if_ic_images_exist, get_hrt_wcs_crval_err
from load_hrt_hmi_files import HRTandHMIfiles

class CorrectHRTWCSPipe:
    """calculate WCS corrections in HRT maps using HMI as reference
    
    Things this Pipeline does:
    -------------------------
    1. Get 'icnt' and 'ic' file pairs
        - if 'blos' in HRT files, try to find 'icnt' files and corresponding HMI 'ic' files
    2. Calculate WCS errors of HRT using HMI
    3. Write WCS errors (CRVAL and CRPIX) to output folder as json files
    
    Critical Assumptions:
    ---------------------
    1. The HRTandHMIfiles instance contains HRT files with both series 'blos' and 'icnt' (No user warning/check before)
    2. The HRTandHMIfiles instance constains HRT files from only one day of observations (No user warning/check before)
    3. The HMI target folders have the structure:
        /path/to/hmi/files/blos_45/
        /path/to/hmi/files/ic_45/
    4. Only correct WCS using the 45s HMI data products
    5. Assumes HMI WCS is perfectly known
    6. HRT CRPIX error is < +/- 500 pixels in X and Y
    7. Linux OS for file paths
    """
    def __init__(self,hrt_hmi_files: HRTandHMIfiles,output_folder: str):
        """init

        Parameters
        ----------
        hrt_hmi_files: HRTandHMIfiles instance
            object that holds .hrt_files and .hmi_files
        output_folder : str
            path to folder where WCS corrections will be written
        """
        self.hrt_hmi_files = hrt_hmi_files
        self.hrt_files = hrt_hmi_files.hrt_files
        self.hmi_files = hrt_hmi_files.hmi_files #only the dataset file name
        self.hrt_fps = hrt_hmi_files.hrt_fps #full path to each file
        self.hmi_fps = hrt_hmi_files.hmi_fps
        self.hrt_date = hrt_hmi_files.hrt_date
        self.output_folder=output_folder

        # load all the input and target files, check if continuum intensity images exist for HRT WCS corrections
        if 'blos' in self.hrt_files[0]:
            try:
                self._check_all_files_for_icnt() #check if continuum intensity images exist
            except:
                raise OSError('Cannot locate continuum intensity images for HRT WCS corrections')

        elif 'icnt' in self.hrt_files[0] and 'ic_' in self.hmi_files[0]:
            self.hrt_wcs_corr_fps = self.hrt_fps
            self.hmi_wcs_corr_fps = self.hmi_fps

    def _check_all_files_for_icnt(self):
        """check if the corresponding continuum intensity images for all input HRT files and HMI files exist"""
        self.hrt_wcs_corr_fps = []
        self.hmi_wcs_corr_fps = []
        for hrtfp,hmifp in zip(self.hrt_fps,self.hmi_fps):
            hrt_icfp,hmi_icfp = check_if_ic_images_exist(hrtfp, hmifp)
            self.hrt_wcs_corr_fps.append(hrt_icfp)
            self.hmi_wcs_corr_fps.append(hmi_icfp)
        
    def file_pair_same_datetime(self,hrt_fp,hmi_fp):
        """check if HRT and HMI files have the same Earth datetime"""
        hrt_earth = fits.getheader(hrt_fp)['DATE_EAR']
        hrt_date = dt.fromisoformat(hrt_earth)
        hmi_date = dt.strptime(fits.open(hmi_fp)[1].header['T_OBS'].strip('_TAI')[:-4],'%Y.%m.%d_%H:%M:%S')

        if hrt_date - hmi_date < datetime.timedelta(seconds=25):
            return True
        else:
            return False
    
    def print_console_wcs_corrections(self,DID,DATE,crval_err,crpix_err):
        """print WCS corrections to console"""
        print('-----------------------------')
        print(f'SO/PHI-HRT DID: {DID}')
        print(f'Date: {DATE}')
        print(f'CRVAL1 error: {crval_err[0]}')
        print(f'CRVAL2 error: {crval_err[1]}')
        print(f'CRPIX1 error: {crpix_err[0]}')
        print(f'CRPIX2 error: {crpix_err[1]}')
        print('-----------------------------')

    def calc_hrt_WCS_corrections(self):
        """get WCS CRVAL (and CRPIX) error in HRT maps using HMI as reference, using the continuum intensity images"""
        self.hrt_CRVAL_corrections = {}
        self.hrt_CRPIX_corrections = {}
        
        for hrt_fp, hmi_fp in zip(self.hrt_wcs_corr_fps,self.hmi_wcs_corr_fps):
            if self.file_pair_same_datetime(hrt_fp,hmi_fp):
                crval_err,crpix_err = get_hrt_wcs_crval_err(hrt_fp,hmi_fp,save_crpix_err=True)
                DID = str(hrt_fp.split('.fits')[0].split('_')[-1])
                DATETIME = dt.strptime(str(hrt_fp.split('_')[-3]), '%Y%m%dT%H%M%S').strftime('%d-%m-%Y %H:%M:%S')
                self.hrt_CRVAL_corrections[DID]=crval_err
                self.hrt_CRPIX_corrections[DID]=crpix_err
                self.print_console_wcs_corrections(DID,DATETIME,crval_err,crpix_err)
            else:
                HRT_input = hrt_fp.split('/')[-1]
                HMI_target = hmi_fp.split('/')[-1]
                print('-----------------------------')
                print('HRT and HMI file pair do not have the same Earth datetime\n')
                print('Skipping WCS corrections for this pair\n')
                print(f'HRT file: \n{HRT_input}')
                print(f'HMI file: \n{HMI_target}')
                print('-----------------------------')
                continue
        
    def write_HRT_WCS_corrections(self):
        """write the HRT WCS corrections to json files in the output folder, in append mode"""
        print(f'Writing HRT WCS corrections to json file in output folder: {self.output_folder}')

        if os.path.isfile(self.output_folder + f'hrt_CRVAL_corrections_{self.hrt_date}.json') or \
            os.path.isfile(self.output_folder + f'hrt_CRPIX_corrections_{self.hrt_date}.json'):

            print(f'File(s) hrt_CRVAL/CRPIX_corrections_{self.hrt_date}.json already exist in output folder.\n\
                        Appending instead, please check for duplicates/multiple dicts in files.')
            with open(self.output_folder + f'hrt_CRVAL_corrections_{self.hrt_date}.json','a') as f:
                json.dump(self.hrt_CRVAL_corrections,f)
            with open(self.output_folder + f'hrt_CRPIX_corrections_{self.hrt_date}.json','a') as f:
                json.dump(self.hrt_CRPIX_corrections,f)
        else:
            with open(self.output_folder + f'hrt_CRVAL_corrections_{self.hrt_date}.json','w') as f:
                json.dump(self.hrt_CRVAL_corrections,f)
            with open(self.output_folder + f'hrt_CRPIX_corrections_{self.hrt_date}.json','w') as f:
                json.dump(self.hrt_CRPIX_corrections,f)

    def run(self):
        self.calc_hrt_WCS_corrections()
        self.write_HRT_WCS_corrections()
        print('HRT WCS corrections calculated and written to json files')
        return None
    
if __name__ == "__main__":
    hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    hmi_input_folder = '/data/slam/sinjan/arlongterm_hmi/ic_45/'
    hrt_input_file_series = 'icnt'
    hmi_target_file_series = 'hmi.ic_45s'
    hrt_dt_start = dt(2023,10,17,0,0,0)
    hrt_dt_end = dt(2023,10,17,3,0,0)
    
    hrt_hmi_files = HRTandHMIfiles(hrt_input_folder, hmi_input_folder,\
                                  hrt_input_file_series, hmi_target_file_series, \
                                  hrt_start_time=hrt_dt_start, hrt_end_time=hrt_dt_end)
    hrt_hmi_files.load()
    
    output_folder = '/data/slam/sinjan/arlongterm_test/'
    
    pipe = CorrectHRTWCSPipe(hrt_hmi_files,output_folder)
    pipe.run()