import os
import json
import _pickle as cPickle
from datetime import datetime as dt
import datetime
from astropy.io import fits
from reproject_funcs import get_hrt_remapped_on_hmi
from load_hrt_hmi_files import HRTandHMIfiles


class ReprojectHRT2HMIPipe:
    """reproject the HRT input files to the target HMI file pair detector
    
    Things this Pipeline does:
    --------------------------
    1. Load WCS corrections
    2. Check WCS corrections fulfil all input HRT files
    3. Correct HRT header and reproject onto HMI
    4. Output HRT and HMI sunpy map objects to pickle
    
    Critical Assumptions:
    ---------------------
    1. WCS correction file contains CRVAL errors for all HRT errors for the input files
    2. Linux OS for file paths
    """
    def __init__(self, hrt_hmi_files: HRTandHMIfiles, output_folder: str, wcs_crval_corr_file: str):
        """init

        Parameters
        ----------
        hrt_hmi_files: HRTandHMIfiles instance
            object that holds .hrt_fps and .hmi_fps
        output_folder : str
            path to folder where remapped HRT and HMI sunpy maps will be written as picles
        wcs_crval_corr_file: str
            path to the CRVAL correction file containing the CRVAL errs for all the HRT input files in the HRTandHMIfiles instance
        """         
        self.hrt_fps = hrt_hmi_files.hrt_fps
        self.hmi_fps = hrt_hmi_files.hmi_fps
        self.hrt_output_series = hrt_hmi_files.hrt_input_file_series
        self.hmi_output_series = hrt_hmi_files.hmi_target_file_series
        self.start_time = hrt_hmi_files.start_time
        self.end_time = hrt_hmi_files.end_time
        self.output_folder = output_folder

        self.wcs_crval_corr_file = wcs_crval_corr_file #CRVAL
        self._check_if_wcs_corrections_exist()

    def _check_if_wcs_corrections_exist(self):
        try:
            os.path.isfile(self.wcs_crval_corr_file)
        except:
            raise OSError('WCS correction file not found.')
                           
    def _get_wcs_corrs(self):
        with open(self.wcs_crval_corr_file,'r') as f:
            self.hrt_CRVAL_corrections = json.load(f)

    def _check_DIDs_in_files_match_those_in_WCS_corrs(self):
        """check if all DIDs in HRT files are in the WCS corrections file"""
        file_DIDs = [str(f.split('.fits')[0].split('_')[-1]) for f in self.hrt_fps]
        if not set(file_DIDs) <= set(self.hrt_CRVAL_corrections.keys()):
            print(file_DIDs)
            print(self.hrt_CRVAL_corrections.keys())
            raise ValueError('DIDs in HRT files are not a (sub)set of the DIDs in WCS corrections file')
                           
    def load_wcs_corrections(self):
        self._get_wcs_corrs()
        self._check_DIDs_in_files_match_those_in_WCS_corrs()
                           
    def print_console(self, DID, DATE, err, hmi_file):
        """print remap status to console"""
        print('-----------------------------')
        print(f'SO/PHI-HRT DID: {DID}')
        print(f'Date: {DATE}')
        print(f'CRVAL Err: {err}')
        print(f'SDO/HMI Target: \n{hmi_file}')
        print('-----------------------------')
                           
    def file_pair_same_datetime(self,hrt_fp,hmi_fp):
        """check if HRT and HMI files have the same Earth datetime"""
        hrt_earth = fits.getheader(hrt_fp)['DATE_EAR']
        hrt_date = dt.fromisoformat(hrt_earth)
        hmi_date = dt.strptime(fits.open(hmi_fp)[1].header['T_OBS'].strip('_TAI')[:-4],'%Y.%m.%d_%H:%M:%S')

        if hrt_date - hmi_date < datetime.timedelta(seconds=25):
            return True
        else:
            return False
                           
    def reproject_hrt2hmi(self):
        self.hrt_remapped_on_hmi_sunpy_maps = []
        self.hmi_target_sunpy_maps = []
                           
        for hrt_fp, hmi_fp in zip(self.hrt_fps,self.hmi_fps):
            if self.file_pair_same_datetime(hrt_fp,hmi_fp):
                file_DID = str(hrt_fp.split('.fits')[0].split('_')[-1])
                err = self.hrt_CRVAL_corrections[file_DID]         
                hrt_remap, hmi_map = get_hrt_remapped_on_hmi(hrt_fp, hmi_fp, err)
                self.hrt_remapped_on_hmi_sunpy_maps.append(hrt_remap)
                self.hmi_target_sunpy_maps.append(hmi_map)
                DID = str(hrt_fp.split('.fits')[0].split('_')[-1])
                DATETIME = dt.strptime(str(hrt_fp.split('_')[-3]), '%Y%m%dT%H%M%S').strftime('%d-%m-%Y %H:%M:%S')
                HMI_target = hmi_fp.split('/')[-1]
                self.print_console(DID,DATETIME, err, HMI_target)
            else:
                HRT_input = hrt_fp.split('/')[-1]
                HMI_target = hmi_fp.split('/')[-1]
                print('-----------------------------')
                print('HRT and HMI file pair do not have the same Earth datetime\n')
                print('Skipping HRT remap for this pair\n')
                print(f'HRT file: \n{HRT_input}')
                print(f'HMI file: \n{HMI_target}')
                print('-----------------------------')
                continue
                           
    def save_hrt_hmi_maps_to_pickles(self):
        starttime = dt.strftime(self.start_time,"%y%M%dT%H%M%S")   
        endtime = dt.strftime(self.end_time,"%y%M%dT%H%M%S")
        hmi_output_series = self.hmi_output_series.split('.')[-1]
        hrt_output_fp = self.output_folder+f"HRTs_{self.hrt_output_series}_remapped_on_HMI_{starttime}_{endtime}.pickle"
        hmi_output_fp = self.output_folder+f"HMIs_{hmi_output_series}_{starttime}_{endtime}.pickle"
        if os.path.isfile(hrt_output_fp) or os.path.isfile(hmi_output_fp):
            print(f'File(s): \n\
                  {hrt_output_fp} \n\
                  {hmi_output_fp} \n\
                  already exist. \n\
                  Adding \'_NEW\' to filenames.')
            hrt_output_fp = hrt_output_fp + '_NEW'
            hmi_output_fp = hmi_output_fp + '_NEW'
                           
        with open(hrt_output_fp, "wb") as input_file:
            cPickle.dump(self.hrt_remapped_on_hmi_sunpy_maps,input_file)

        with open(hmi_output_fp, "wb") as input_file:
            cPickle.dump(self.hmi_target_sunpy_maps,input_file)

    def run(self):
        """reproject all HRT blos (or ic) maps to HMI frame"""
        self.load_wcs_corrections()
        self.reproject_hrt2hmi()
        self.save_hrt_hmi_maps_to_pickles()
                           
if __name__ == "__main__":
    day = 12
    hrt_input_folder = f'/data/solo/phi/data/fmdb/public/l2/2023-10-{day}/'
    hmi_input_folder = '/data/slam/sinjan/arlongterm_hmi/b_720/'
    hrt_input_file_series = 'bmag'
    hmi_target_file_series = 'hmi.b_720s_field' #same as hrt for reprojecting purposes
    hrt_dt_start = dt(2023,10,day,0,0,0)
    hrt_dt_end = dt(2023,10,day+1,0,0,0)
    
    hrt_hmi_files = HRTandHMIfiles(hrt_input_folder, hmi_input_folder,\
                                  hrt_input_file_series, hmi_target_file_series, \
                                  hrt_start_time=hrt_dt_start, hrt_end_time=hrt_dt_end)
    hrt_hmi_files.load()
    
    wcs_corr_file = f'/data/slam/sinjan/arlongterm_hrt_wcs_corr/hrt_CRVAL_corrections_202310{day}.json'
    output_folder = '/data/slam/sinjan/arlongterm_pickles/'
    
    pipe = ReprojectHRT2HMIPipe(hrt_hmi_files, output_folder, wcs_corr_file)
    pipe.run()