from datetime import datetime as dt
import drms
import numpy as np
from astropy.io import fits

def get_hmi_blos(date: dt, series: str = 'hmi.m_45s', email: str = 'yourname@mail.com', out_dir: str='/path/to/save/files/'):
    """get one HMI blos map for a given datetime (in UTC)

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
    email : str
        email address for DRMS (default: 'yourname@gmail.com')
    out_dir : str

    Returns
    -------
    None   
    """
    dtai = dt.timedelta(seconds=37) # difference between TAI and UTC
    if series == 'hmi.m_45s':
        halfcad=23
        dcad = dt.timedelta(seconds=35) # half HMI cadence (23) + margin
    elif series == 'hmi.m_720s':
        halfcad=360
        dcad = dt.timedelta(seconds=360)

    client = drms.Client(email=email, verbose=True)
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


def get_hrt_dates(hrt_dir:str=''):
    """get all dates for all HRT blos files in given directory

    Parameters
    ----------
    hrt_dir : str
        path to directory containing HRT blos data

    Returns
    -------
    dates : list
        list of datetime.datetime objects
    """
    import os
    from datetime import datetime as dt
    dates = []
    base_files = os.listdir(hrt_dir)
    blos_files = set([file for file in base_files if 'blos' in file].sort())

    for file in blos_files:
            ht = fits.getheader(hrt_dir+file)
            dates.append(dt.fromisoformat(ht['DATE_EAR'])) #take into account the light time travel difference
    return dates


def download_all_hmi_blos(hrt_dir:str='', series: str = 'hmi.m_45s', email: str = '', out_dir: str=''):
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
    dates = get_hrt_dates(hrt_dir)
    for date in dates:
        get_hmi_blos(date, series, email, out_dir)
    return None


