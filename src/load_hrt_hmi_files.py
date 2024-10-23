import os
import datetime
from datetime import datetime as dt
from download_all_hmi_files import get_list_files

class HRTandHMIfiles:
    """load full filepaths for HRT and HMI files
    
    Things this class does:
    -----------------------
    1. Load all the HRT and HMI files in the given input folders for the given file_series ('blos' or 'continuum intensity')
    2. Remove any files outside the desired time range
    3. Create full file paths
    
    Critical Assumptions:
    ---------------------
    1. Linux OS for file paths
    2. Only using the 45s data products from HMI
    3. HRT input folder only contains HRT files from one date
    
    """
    def __init__(self,hrt_input_folder: str,hmi_input_folder: str,\
                 hrt_input_file_series: str,hmi_target_file_series: str,\
                 hrt_start_time: dt = None,hrt_end_time: dt = None):
        """init

        Parameters
        ----------
        hrt_input_folder : str
            path to folder containing HRT files
        hmi_input_folder : str
            path to folder containing HMI files
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

    def get_hmi_list_files(self,pdir:str='', series: str='', instrument:str = 'hmi'):
        """get list of hmi files for given series in given directory matching the date with the hrt files
        """
        files = list(set([file for file in os.listdir(pdir) if (series in file and instrument in file and self.hrt_date in file)]))
        files.sort()
        return files

    def get_all_hrt_files(self):
        """get list of desired HRT files in input folder"""
        self.hrt_files = get_list_files(self.hrt_input_folder,self.hrt_input_file_series, 'hrt')

    def get_hrt_date(self):
        self.hrt_date=self.hrt_files[0].split('_')[-3].split('T')[0]
        if any([self.hrt_date not in file for file in self.hrt_files]):
            raise AssertionError(f'Not all HRT files contain {self.hrt_date}')

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
            print(f'Number of HRT and HMI files are not equal\n\
                    HRT files: {self.number_hrt_files}\n\
                    HMI files: {self.number_hmi_files}')
            
    def create_full_file_paths(self):
        self.hrt_fps = [os.path.join(self.hrt_input_folder,fn) for fn in self.hrt_files]
        self.hmi_fps = [os.path.join(self.hmi_input_folder,fn) for fn in self.hmi_files]
        
    def load(self):
        self.get_all_hrt_files()
        self.get_hrt_date()
        self.get_all_hmi_files()
        self.set_start_end_timechecks()
        self.remove_files_outside_start_end_time()
        self.check_number_hrt_hmi_files()
        self.create_full_file_paths()