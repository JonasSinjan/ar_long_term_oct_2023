import numpy as np
import pickle as cPickle

def get_hrt_hmi_arr_from_pickles_45s(folder, hrt_series, hmi_series, hrt_psf, hrt_suffix = '', hmi_suffix = '', year = '2023'):
    start = 0
    dates = [12,13,14,15,16,17]
    num_files = 129

    hrt_arr = np.zeros((4102,4102,num_files))
    hmi_arr = np.zeros((4102,4102,num_files))

    hrt_meta_list = []
    
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
        
        with open(folder+f"HMIs_{hmi_series}_{year}10{i}T000000_{year}10{endday}T{endhour}{endmin}00.pickle{hmi_suffix}", "rb") as input_file:
            hmi_tmps = cPickle.load(input_file)
            
        for i,smap in enumerate(hrt_tmps):
            hrt_arr[:,:,start+i] = smap.data
            hrt_meta_list.append(smap.meta)
    
        for i,smap in enumerate(hmi_tmps): #could zip together the two for loops, but need iterable
            nans=np.isnan(hrt_arr[:,:,start+i])
            tmp = smap.data
            tmp[nans] = np.nan
            hmi_arr[:,:,start+i] = tmp
            
        start += len(hrt_tmps)

    return hrt_arr, hmi_arr, hrt_meta_list


def get_hrt_hmi_arr_from_pickles_720s(folder, hrt_series, hmi_series, hrt_psf, hrt_suffix = '', hmi_suffix = '', year = '2023'):
    start = 0
    dates = [12,13,14,15,16,17]
    num_files = 129

    hrt_arr = np.zeros((4098,4098,129))
    hmi_arr = np.zeros((4098,4098,129))

    hrt_meta_list = []
    
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
        
        with open(folder+f"HMIs_{hmi_series}_{year}10{i}T000000_{year}10{endday}T{endhour}{endmin}00.pickle{hmi_suffix}", "rb") as input_file:
            hmi_tmps = cPickle.load(input_file)
            
        for i,smap in enumerate(hrt_tmps):
            hrt_arr[:,:,start+i] = smap.data
            hrt_meta_list.append(smap.meta)
    
        for i,smap in enumerate(hmi_tmps): #could zip together the two for loops, but need iterable
            nans=np.isnan(hrt_arr[:,:,start+i])
            tmp = smap.data
            tmp[nans] = np.nan
            hmi_arr[:,:,start+i] = tmp
            
        start += len(hrt_tmps)

    return hrt_arr, hmi_arr, hrt_meta_list