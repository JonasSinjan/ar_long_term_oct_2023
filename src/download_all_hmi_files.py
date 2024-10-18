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
    if series == 'hmi.m_45s':
        halfcad=23
        dcad = timedelta(seconds=35) # half HMI cadence (23) + margin
    elif series == 'hmi.m_720s':
        halfcad=360
        dcad = timedelta(seconds=360)
    elif series == 'hmi.ic_45s':
        halfcad=23
        dcad = timedelta(seconds=35)
    elif series == 'hmi.ic_720s':
        halfcad=360
        dcad = timedelta(seconds=360)
    else:
        raise ValueError('Invalid hmi series name, choose from \'hmi.m_45s\', \'hmi.m_720s\', \'hmi.ic_45s\', \'hmi.ic_720s\'')

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
    return None


if __name__ == '__main__':
    hrt_dir = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    series = 'hmi.m_45s'
    email = 'jonassinjan8@gmail.com'
    out_dir = '/data/slam/sinjan/arlongterm_hmi/blos_45/'
    hrt_start_datetime = dt(2023,10,17,0,0,0)
    hrt_end_datetime = dt(2023,10,17,11,0,0)

    download_all_hmi(hrt_dir, series, email, out_dir, hrt_start_datetime=hrt_start_datetime,hrt_end_datetime=hrt_end_datetime)