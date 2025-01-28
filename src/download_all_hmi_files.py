import os
from datetime import datetime as dt
from datetime import timedelta
import drms
import numpy as np
from astropy.io import fits

def download_hmi_file(date: dt, series: str = 'hmi.m_45s', email: str = 'yourname@mail.com', out_dir: str='/path/to/save/files/'):
    """get one HMI blos/ic map for a given datetime (in UTC)

    WARNING: JSOC will blacklist your email if you submit too many requests (1000+). For large downloads submit one request. You can also use the JSOC website to download the data manually.

    Parameters
    ----------
    date : datetime.datetime
        date for which the closest HMI blos map is to be downloaded (UTC)
    series : str
        DRMS series name
        Options: 
            'hmi.m_45s' (default)
            'hmi.m_720s'
            'hmi.ic_45s'
            'hmi.ic_720s'
    email : str
        email address for DRMS (default: 'yourname@gmail.com')
    out_dir : str

    Returns
    -------
    None   
    """
    dtai = timedelta(seconds=37) # difference between TAI and UTC
    if series[-3:] == '45s':
        halfcad=23
        dcad = timedelta(seconds=35) # half HMI cadence (23) + margin
    elif series[-4:] == '720s':
        halfcad=360
        dcad = timedelta(seconds=380) # half HMI cadence (360) + margin
    else:
        raise ValueError('Cadence of series not recognized: only 45s and 720s are supported')

    client = drms.Client(email=email)
    kwlist = ['T_REC','T_OBS','DATE-OBS','CADENCE']

    keys = client.query(str(series)+'['+(date+dtai-dcad).strftime('%Y.%m.%d_%H:%M:%S')+'-'+
                      (date+dtai+dcad).strftime('%Y.%m.%d_%H:%M:%S')+']',seg=None,key=kwlist,n=2,)

    ind = np.argmin([np.abs((dt.strptime(t,'%Y.%m.%d_%H:%M:%S_TAI') - dtai - date).total_seconds()) for t in keys['T_OBS']])
    name_h = str(series)+'['+keys['T_REC'][ind]+']'

    if np.abs((dt.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai - date).total_seconds()) > halfcad:
        print('WARNING: Closer file exists but has not been found')
        print(name_h)
        print('T_OBS:',dt.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai)
        print('Input DATETIME: ',date)
        print('')
    
    m = client.export(name_h,protocol='fits')
    m.download(out_dir)
    return None


def get_list_files(pdir:str='', series: str = 'blos', instrument:str = 'hrt'):
    """get list of hrt files for given series in given directory

    Parameters
    ----------
    pdir : str
        path to directory containing HRT blos data
    series : str
        series name in file
        'blos' (default)
        'icnt'

    """
    files = list(set([file for file in os.listdir(pdir) if (series in file and instrument in file)]))
    files.sort()
    return files


def get_hrt_earth_datetimes(hrt_dir:str='', series: str = 'blos', start_time: dt = None, end_time: dt = None):
    """get all dates for all HRT blos files in given directory

    Parameters
    ----------
    hrt_dir : str
        path to directory containing HRT blos data
    series : str
        series name in file
        'blos' (default)
        'icnt'
    end_time : datetime.datetime
        end time for which to get dates (default: None)

    Returns
    -------
    dates : list
        list of datetime.datetime objects
    """
    dates = []
    input_files=get_list_files(hrt_dir, series, 'hrt')

    for file in input_files:
            ht = fits.getheader(hrt_dir+file)
            date = dt.fromisoformat(ht['DATE_EAR'])
            date_hrt = dt.fromisoformat(ht['DATE-OBS']) #find time range on HRT time
            if start_time <= date_hrt <= end_time:
                dates.append(date) 
            else:
                continue
    return dates


def download_all_hmi(hrt_dir:str='', series: str = 'hmi.m_45s', email: str = '', out_dir: str='', \
                     hrt_start_datetime: dt = None, hrt_end_datetime: dt = None):
    """download HMI blos maps for a list of dates
    
    Parameters
    ----------
    dates : list
        list of datetime.datetime objects
    series : str
        DRMS series name
        Options: 
            'hmi.m_45s' (default)
            'hmi.m_720s'
    email : str
        email address for DRMS (default: '')

    Returns
    -------
    None
    """
    dates = get_hrt_earth_datetimes(hrt_dir, start_time=hrt_start_datetime, end_time=hrt_end_datetime)
    for date in dates:
        download_hmi_file(date, series, email, out_dir)
    if series == 'B_720s':
        clean_up_folder_b_720s(out_dir)
    return None

def clean_up_folder_b_720s(out_dir):
    """clean up folder containing HMI 720s files, leaves 'field.fits, 'inclination.fits', 'azimuth.fits'

    Parameters
    ----------
    out_dir : str
        path to directory containing HMI 720s files

    Returns
    -------
    None
    """
    files = os.listdir(out_dir)
    unwanted_files = ['confid_map','info_map','conf_disambig','azimuth_alpha_err','inclination_alpha_err',\
                      'field_alpha_err','inclin_azimuth_err','field_az_err','field_inclination_err','alpha_err',\
                      'vlos_err','field_err','azimuth_err','inclination_err','conv_flag','chisq','alpha_mag',\
                      'src_grad','src_continuum','damping','eta_0','dop_width','vlos_mag','azimuth','disambig']
    for file in files:
        if any(file_name in file for file_name in unwanted_files):
            os.remove(out_dir+file)

    extensions_to_check = ('field.fits','inclination.fits','azimuth.fits')
    assert all(file.endswith(extensions_to_check) for file in os.listdir(out_dir))
    return None

if __name__ == '__main__':

    series = 'hmi.ic_720s'
    email = 'jonassinjan8@gmail.com'
    out_dir = '/scratch/slam/sinjan/arlongterm_hmi/ic_720/'

    for i in [12]:
        hrt_dir = f'/data/solo/phi/data/fmdb/public/l2/2023-10-{i}/'
        end = i+1
        endhour = 0
        endminute = 0
        if i == 17:
            endhour = 11
            endminute = 2
            end = i
        hrt_start_datetime = dt(2023,10,i,0,0,0)
        hrt_end_datetime = dt(2023,10,end,endhour,endminute,0)

        download_all_hmi(hrt_dir, series, email, out_dir, hrt_start_datetime=hrt_start_datetime,hrt_end_datetime=hrt_end_datetime)
    
        #clean_up_folder_b_720s(out_dir)