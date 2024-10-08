import json
import os
import datetime
from datetime import datetime as dt
from astropy.io import fits

from reproject_funcs import check_if_ic_images_exist, get_hrt_wcs_crval_err
from download_all_hmi_files import get_list_files

class CorrectHRTWCSPipe:
    """Pipeline to calculate WCS corrections in HRT maps using HMI as reference
    
    Critical Assumptions:
    ---------------------
    1. The HRT input folder contains files with both series 'blos' and 'icnt' (No user warning/check before)
    2. The HRT input folder only contains files from one day of observations (No user warning/check before)
    3. The HMI target folders have the structure:
        /path/to/hmi/files/blos_45/
        /path/to/hmi/files/ic_45/
    4. Only correct WCS using the 45s HMI data products
    """
    def __init__(self,hrt_input_folder: str,hmi_input_folder: str,output_folder: str,\
                 hrt_input_file_series: str,hmi_target_file_series: str,\
                 hrt_start_time: dt = None,hrt_end_time: dt = None):
        """init

        Parameters
        ----------
        hrt_input_folder : str
            path to folder containing HRT files
        hmi_input_folder : str
            path to folder containing HMI files
        output_folder : str
            path to folder where WCS corrections will be written
        hrt_input_file_series : str
            series name in HRT files
            'blos'
            'icnt'
        hmi_target_file_series : str
            series name in HMI files
            'hmi.m_45s'
            'hmi.ic_45s'
        hrt_start_time : datetime.datetime, optional
            start time for HRT files to be corrected (default: None)
        hrt_end_time : datetime.datetime, optional
            end time for HRT files to be corrected (default: None)
        """
        self.hrt_input_folder=hrt_input_folder
        self.hmi_input_folder=hmi_input_folder
        self.output_folder=output_folder
        self.hrt_input_file_series=hrt_input_file_series #'blos','icnt'
        self.hmi_target_file_series=hmi_target_file_series #'hmi.m_45s, hmi.m_720s, hmi.ic_45s, hmi.ic_720s'

        if isinstance(hrt_start_time,dt):
            self.start_time = hrt_start_time
        else:
            self.start_time = None
        
        if isinstance(hrt_end_time,dt):
            self.end_time = hrt_end_time
        else:
            self.end_time = None

        # load all the input and target files, check if continuum intensity images exist for HRT WCS corrections
        if self.hrt_input_file_series == 'blos':
            try:
                self._check_all_files_for_icnt() #check if continuum intensity images exist
            except:
                raise OSError('Cannot locate continuum intensity images for HRT WCS corrections')

        elif self.hrt_input_file_series == 'icnt' and 'ic_' in self.hmi_target_file_series:
            self.load_hrt_hmi_files()
            self.hrt_wcs_corr_files = [self.hrt_input_folder + f for f in self.hrt_files]
            self.hmi_wcs_corr_files = [self.hmi_input_folder + f for f in self.hmi_files]

    def _check_all_files_for_icnt(self):
        """check if the corresponding continuum intensity images for all input HRT files and HMI files exist"""
        self.load_hrt_hmi_files()
        self.hrt_wcs_corr_files = []
        self.hmi_wcs_corr_files = []
        for hrtf,hmif in zip(self.hrt_files,self.hmi_files):
            hrt_file = self.hrt_input_folder+hrtf
            hmi_file = self.hmi_input_folder+hmif
            hrt_icfile,hmi_icfile = check_if_ic_images_exist(hrt_file, hmi_file)
            self.hrt_wcs_corr_files.append(hrt_icfile)
            self.hmi_wcs_corr_files.append(hmi_icfile)

    def get_hmi_list_files(self,pdir:str='', series: str='', instrument:str = 'hmi'):
        """get list of hmi files for given series in given directory matching the date with the hrt files
        """
        files = list(set([file for file in os.listdir(pdir) if (series in file and instrument in file and self.hrt_date in file)]))
        files.sort()
        return files

    def get_all_hrt_files(self):
        """get list of desired HRT files in input folder"""
        self.hrt_files = get_list_files(self.hrt_input_folder,self.hrt_input_file_series, 'hrt')
        self.hrt_date=self.hrt_files[0].split('_')[-3].split('T')[0]

    def get_all_hmi_files(self):
        """get list of desired HMI files in input folder"""
        self.hmi_files = self.get_hmi_list_files(self.hmi_input_folder,self.hmi_target_file_series, 'hmi')

    def set_start_end_timechecks(self):
        if self.start_time is None:
            self.start_time = dt.strptime(self.hrt_date,'%Y%m%d') #set to 00:00
        if self.end_time is None:
            self.end_time = self.start_time + datetime.timedelta(days=1)

    def remove_files_outside_start_end_time(self):
        hrttmp = list(self.hrt_files)
        hmitmp = list(self.hmi_files)
        for hrtf in hrttmp:
            hrt_datetime = dt.strptime(str(hrtf.split('_')[-3]), '%Y%m%dT%H%M%S')
            if hrt_datetime <= self.start_time or hrt_datetime >= self.end_time:
                self.hrt_files.remove(hrtf)
                print(f'Removing file: {hrtf} from HRT files list as it is not in the desired time range')

        for hmif in hmitmp: #might have unequal number of files
            hmi_datetime = dt.strptime(str(hmif.split('_TAI')[0]\
                                        .split(self.hmi_target_file_series+'.')[1]), '%Y%m%d_%H%M%S')
            light_travel_time=datetime.timedelta(seconds=400) #6 minutes maximum + 30 seconds safety from hmi 45s  
            if hmi_datetime <= self.start_time + light_travel_time or hmi_datetime >= self.end_time + light_travel_time:
                self.hmi_files.remove(hmif)
                print(f'Removing file: {hmif} from HMI files list as it is not in the desired time range')

        del hrttmp
        del hmitmp

    def check_number_hrt_hmi_files(self):
        """check if the number of HRT and HMI files are equal"""
        self.number_hrt_files = len(self.hrt_files)
        self.number_hmi_files = len(self.hmi_files)
        if self.number_hrt_files == 0:
            raise ValueError('No HRT files found in the desired time range')
        elif self.number_hmi_files == 0:
            raise ValueError('No HMI files found in the desired time range')
        elif self.number_hrt_files != self.number_hmi_files:
            raise ValueError(f'Number of HRT and HMI files are not equal\n\
                             HRT files: {self.number_hrt_files}\n\
                             HMI files: {self.number_hmi_files}')
        
    def load_hrt_hmi_files(self):
        self.get_all_hrt_files()
        self.get_all_hmi_files()
        self.set_start_end_timechecks()
        self.remove_files_outside_start_end_time()
        self.check_number_hrt_hmi_files()
        
    def file_pair_same_datetime(self,hrt_file,hmi_file):
        """check if HRT and HMI files have the same Earth datetime"""
        hrt_earth=fits.getheader(hrt_file)['DATE_EAR']
        hrt_date = dt.fromisoformat(hrt_earth)
        hmi_date = dt.strptime(fits.open(hmi_file)[1].header['T_OBS'].strip('_TAI')[:-4],'%Y.%m.%d_%H:%M:%S')

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
        
        for hrt_file, hmi_file in zip(self.hrt_wcs_corr_files,self.hmi_wcs_corr_files):
            if self.file_pair_same_datetime(hrt_file,hmi_file):
                crval_err,crpix_err = get_hrt_wcs_crval_err(hrt_file,hmi_file,save_crpix_err=True)
                DID = str(hrt_file.split('.fits')[0].split('_')[-1])
                DATETIME = dt.strptime(str(hrt_file.split('_')[-3]), '%Y%m%dT%H%M%S').strftime('%d-%m-%Y %H:%M:%S')
                self.hrt_CRVAL_corrections[DID]=crval_err
                self.hrt_CRPIX_corrections[DID]=crpix_err
                self.print_console_wcs_corrections(DID,DATETIME,crval_err,crpix_err)
            else:
                HRT_input=hrt_file.split('/')[-1]
                HMI_target=hmi_file.split('/')[-1]
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
    output_folder = '/data/slam/sinjan/arlongterm_hrt_wcs_corr/'
    hrt_input_file_series = 'icnt'
    hmi_target_file_series = 'hmi.ic_45s'
    hrt_dt_start = dt(2023,10,17,0,0,0)
    hrt_dt_end = dt(2023,10,17,11,10,0)

    pipe = CorrectHRTWCSPipe(hrt_input_folder,hmi_input_folder,output_folder,\
                             hrt_input_file_series,hmi_target_file_series, \
                             hrt_start_time=hrt_dt_start, hrt_end_time=hrt_dt_end)
    pipe.run()