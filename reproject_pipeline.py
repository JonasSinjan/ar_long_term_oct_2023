import os
import json
import _pickle as cPickle
from reproject_funcs import get_hrt_remapped_on_hmi


class ReprojectHRT2HMIPipe:

    def __init__(self, hrt_hmi_files: HRTandHMIfiles, output_folder: str, wcs_crval_corr_file: str):
                 
        self.hrt_fps = hrt_hmi_files.hrt_fps
        self.hmi_fps = hrt_hmi_files.hmi_fps
        self.output_folder = output_folder

        self.wcs_crval_corr_file = wcs_crval_corr_file #CRVAL
        self._check_if_wcs_corrections_exist()

    def _check_if_wcs_corrections_exist(self):
        try:
            os.path.isfile(self.wcs_crval_corr_file')
        except:
            raise OSError('WCS correction file not found.')
                           
    def _get_wcs_corrs()
        with open(self.wcs_crval_corr_file,'r') as f:
            self.hrt_CRVAL_corrections = json.load(f)

    def _check_DIDs_in_files_match_those_in_WCS_corrs(self):
        """check if all DIDs in HRT files are in the WCS corrections file"""
        file_DIDs = [str(f.split('.fits')[0].split('_')[-1]) for f in self.hrt_fps]
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
            if self.file_pair_same_datetime(hrt_file,hmi_file):
                file_DID = str(hrt_fp.split('.fits')[0].split('_')[-1])
                err = self.hrt_corrections[file_DID]         
                hrt_remap, hmi_map = get_hrt_remapped_on_hmi(hrt_fp, hmi_fp, err)
                self.hrt_remapped_on_hmi_sunpy_maps.append(hrt_remap)
                self.hmi_target_sunpy_maps.append(hmi_map)
                DID = str(hrt_fp.split('.fits')[0].split('_')[-1])
                DATETIME = dt.strptime(str(hrt_fp.split('_')[-3]), '%Y%m%dT%H%M%S').strftime('%d-%m-%Y %H:%M:%S')
                HMI_target = hmi_fp.split('/')[-1]
                self.printe_console(DID,DATETIME, err, HMI_target)
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
        DATE=str(self.hrt_fps[0].split('_')[-3].split('T')[0])    
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
        self.load_wcs_corrections()
        self.reproject_hrt2hmi()
        self.save_hrt_hmi_maps_to_pickles()
                           
if __name__ == "__main__":
    hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-12/'
    hmi_input_folder = '/data/slam/sinjan/arlongterm_hmi/blos_45/'
    hrt_input_file_series = 'blos'
    hmi_target_file_series = 'hmi.m_45s'
    hrt_dt_start = dt(2023,10,12,0,0,0)
    hrt_dt_end = dt(2023,10,13,0,0,0)
    
    hrt_hmi_files = HRTandHMIfiles(hrt_input_folder, hmi_input_folder,\
                                  hrt_input_file_series, hmi_target_file_series, \
                                  hrt_start_time=hrt_dt_start, hrt_end_time=hrt_dt_end)
    hrt_hmi_files.load()
    
    wcs_corr_folder = '/data/slam/sinjan/arlongterm_hrt_wcs_corr/'
    output_folder = '/data/slam/sinjan/arlongterm_out_pickles/'
    
    pipe = ReprojectHRT2HMIPipe(hrt_hmi_files, output_folder, wcs_corr_folder)
    pipe.run()