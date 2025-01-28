# load, rotate and save HMI ic files to a pickle

import os
import numpy as np
import sunpy.map
import _pickle as cPickle

hmi_dir = '/scratch/slam/sinjan/arlongterm_hmi/ic_45/'

hmi_files = []

dates = [12,13,14,15,16,17]

for i in dates:
    hmi_files += [j for j in os.listdir(hmi_dir) if f'202310{i}' in j]
    
hmi_files.sort()

hmi_ic_arr=np.zeros((4102,4102,len(hmi_files)))

# with open('/data/slam/sinjan/arlongterm_pickles/hrt_arr_first_ar.pickle','rb') as f:
#     hrt_arr = cPickle.load(f)

def get_hrt_hmi_arr_from_pickles(folder, hrt_series, hrt_psf, hrt_suffix = '', year = '2023'):
    start = 0
    dates = [12,13,14,15,16,17]
    num_files = 129

    hrt_arr = np.zeros((4102,4102,num_files))
    
    for i in dates:
        if i == 17:
            endhour = '11'
            endmin = '02'
            endday = 17
        else:
            endhour = '00'
            endmin = '00'
            endday = i+1
        with open(folder+f"HRTs_{hrt_series}_remapped_on_HMI_{year}10{i}T000000_{year}10{endday}T{endhour}{endmin}00{hrt_psf}.pickle{hrt_suffix}", "rb") as input_file:
            hrt_tmps = cPickle.load(input_file)

            
        for i,smap in enumerate(hrt_tmps):
            hrt_arr[:,:,start+i] = smap.data
            
        start += len(hrt_tmps)

    return hrt_arr

folder = '/data/slam/sinjan/arlongterm_pickles_hann_SL/'
hrt_series = 'icnt'
hrt_psf = '_hmipsf_True'

hrt_arr = get_hrt_hmi_arr_from_pickles(folder, hrt_series, hrt_psf)
    
for i,hfile in enumerate(hmi_files):
    smap = sunpy.map.Map(hmi_dir+hfile).rotate()
    norm = smap.data[1500:2500,1500:2500].mean()
    nans = np.isnan(hrt_arr[:,:,i])
    tmp = smap.data
    tmp[nans] = np.nan
    hmi_ic_arr[...,i] = tmp/norm
    
with open('/data/slam/sinjan/arlongterm_pickles_hann_SL/hmi_ic_cutout_20231012_20231017.pickle','wb') as f:
    cPickle.dump(hmi_ic_arr,f)