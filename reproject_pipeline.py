import os
import json
import _pickle as cPickle
from reproject_funcs import get_hrt_remapped_on_hmi


class ReprojectHRT2HMIPipe:

    def __init__(self,,hrt_input_folder: str,hmi_input_folder: str,output_folder: str,\
                 hrt_input_file_series: str,hmi_target_file_series: str, wcs_crval_corr_file: str,\
                 hrt_start_time: dt = None,hrt_end_time: dt = None):
                 
        self.hrt_input_folder=hrt_input_folder
        self.hmi_input_folder=hmi_input_folder
        self.output_folder=output_folder
        self.hrt_input_file_series=hrt_input_file_series #'blos','icnt'
        self.hmi_target_file_series=hmi_target_file_series #'hmi.m_45s, hmi.m_720s, hmi.ic_45s, hmi.ic_720s'

        self.wcs_crval_corr_file=wcs_crval_corr_file #CRVAL
        self._check_if_wcs_corrections_exist()
        
        if isinstance(hrt_start_time,dt):
            self.start_time = hrt_start_time
        else:
            self.start_time = None
        
        if isinstance(hrt_end_time,dt):
            self.end_time = hrt_end_time
        else:
            self.end_time = None

    def _check_if_wcs_corrections_exist(self):
        try:
            os.path.isfile(self.wcs_crval_corr_file')
        except:
            raise OSError('WCS correction file not found.')
                           
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
                           
    def _get_wcs_corrs()
        with open(self.wcs_crval_corr_file,'r') as f:
            self.hrt_CRVAL_corrections = json.load(f)

    def _check_DIDs_in_files_match_those_in_WCS_corrs(self):
        """check if all DIDs in HRT files are in the WCS corrections file"""
        file_DIDs = [str(f.split('.fits')[0].split('_')[-1]) for f in self.hrt_files]
        if set(file_DIDs) != set(self.hrt_corrections.keys()):
            raise ValueError('DIDs in HRT files are not equal to DIDs in WCS corrections file')
                           
    def load_wcs_corrections(self):
        self._get_wcs_corrs()
        self._check_DIDs_in_hrt_CRVAL_corrections()
                           
    def print_console(self, DID, DATE, err, hmi_file):
        """print remap status to console"""
        print('-----------------------------')
        print(f'SO/PHI-HRT DID: {DID}')
        print(f'Date: {DATE}')
        print(f'CRVAL Err: {err}')
        print(f'SDO/HMI Target: \n{hmi_file}')
        print('-----------------------------')
                           
    def file_pair_same_datetime(self,hrt_file,hmi_file):
        """check if HRT and HMI files have the same Earth datetime"""
        hrt_earth=fits.getheader(hrt_file)['DATE_EAR']
        hrt_date = dt.fromisoformat(hrt_earth)
        hmi_date = dt.strptime(fits.open(hmi_file)[1].header['T_OBS'].strip('_TAI')[:-4],'%Y.%m.%d_%H:%M:%S')

        if hrt_date - hmi_date < datetime.timedelta(seconds=25):
            return True
        else:
            return False
                           
    def reproject_hrt2hmi(self):
        self.hrt_remapped_on_hmi_sunpy_maps=[]
        self.hmi_target_sunpy_maps=[]
                           
        for hrt_file,hmi_file in zip(self.hrt_files,self.hmi_files):
            if self.file_pair_same_datetime(hrt_file,hmi_file):
                file_DID = str(hrt_file.split('.fits')[0].split('_')[-1])
                err = self.hrt_corrections[file_DID]         
                hrt_remap, hmi_map = get_hrt_remapped_on_hmi(hrt_file, hmi_file, err)
                self.hrt_remapped_on_hmi_sunpy_maps.append(hrt_remap)
                self.hmi_target_sunpy_maps.append(hmi_map)
                DID = str(hrt_file.split('.fits')[0].split('_')[-1])
                DATETIME = dt.strptime(str(hrt_file.split('_')[-3]), '%Y%m%dT%H%M%S').strftime('%d-%m-%Y %H:%M:%S')
                HMI_target = hmi_file.split('/')[-1]
                self.printe_console(DID,DATETIME, err, HMI_target)
            else:
                HRT_input=hrt_file.split('/')[-1]
                HMI_target=hmi_file.split('/')[-1]
                print('-----------------------------')
                print('HRT and HMI file pair do not have the same Earth datetime\n')
                print('Skipping HRT remap for this pair\n')
                print(f'HRT file: \n{HRT_input}')
                print(f'HMI file: \n{HMI_target}')
                print('-----------------------------')
                continue
                           
    def save_hrt_hmi_maps_to_pickles(self):
        DATE=str(self.hrt_files[0].split('_')[-3].split('T')[0])    
        hrt_output_fp = self.output_folder+f"HRTs_remapped_on_HMI_{DATE}.pickle"
        hmi_output_fp = self.output_folder+f"HMIs_target_{DATE}.pickle"
        if os.path.isfile(hrt_output_fp) or os.path.isfile(hmi_output_fp):
            print(f'File(s): \n\
                  HRTs_remapped_on_HMI_{DATE}.pickle \n\
                  HMIs_target_{DATE}.pickle \n\
                  already exist in output folder. \n\
                  Adding \'_NEW\' to filenames.')
            hrt_output_fp = hrt_output_fp + '_NEW'
            hmi_output_fp = hmi_output_fp + '_NEW'
                           
        with open(hrt_output_fp, "wb") as input_file:
            cPickle.dump(self.hrt_remapped_on_hmi_sunpy_maps,input_file)

        with open(hmi_output_fp, "wb") as input_file:
            cPickle.dump(self.hmi_target_sunpy_maps,input_file)

    def run(self):
        """reproject all HRT blos (or ic) maps to HMI frame"""
        self.load_hrt_hmi_files()
        self.load_wcs_corrections()
        self.reproject_hrt2hmi()
        self.save_hrt_hmi_maps_to_pickles()