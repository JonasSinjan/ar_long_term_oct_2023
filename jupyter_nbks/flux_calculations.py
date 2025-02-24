import numpy as np

def get_fluxes(hrt_mag, hmi_mag, hrt_ic, hmi_ic, hrt_mu_arr, hmi_mu_arr, noise_level=50, filter_bleft_corner=True, ic_threshold=0.55):

    #print all shapes
    print('HRT mag shape:',hrt_mag.shape)
    print('HMI mag shape:',hmi_mag.shape)   
    print('HRT mu shape:',hrt_mu_arr.shape)
    print('HMI mu shape:',hmi_mu_arr.shape)
    print('HRT ic shape:',hrt_ic.shape)
    print('HMI ic shape:',hmi_ic.shape)

    time_steps = min(hrt_mag.shape[-1], hmi_mag.shape[-1], \
                     hrt_mu_arr.shape[-1], hmi_mu_arr.shape[-1], \
                     hrt_ic.shape[-1], hmi_ic.shape[-1])
    
    print('Time steps:',time_steps)

    hrt_mag = hrt_mag[:,:,:time_steps]
    hmi_mag = hmi_mag[:,:,:time_steps]
    hrt_ic = hrt_ic[:,:,:time_steps]
    hmi_ic = hmi_ic[:,:,:time_steps]
    hrt_mu_arr = hrt_mu_arr[:,:,:time_steps]
    hmi_mu_arr = hmi_mu_arr[:,:,:time_steps]

    #for some reason all 45s products when rotated are 4102,4102 arrays
    #while all 720s when rotated are 4098,4098 arrays

    if hrt_mag.shape[0] == 4102:
        hrt_mag = hrt_mag[2:-2,2:-2,:]
    if hmi_mag.shape[0] == 4102:
        hmi_mag = hmi_mag[2:-2,2:-2,:]
    if hrt_ic.shape[0] == 4102:
        hrt_ic = hrt_ic[2:-2,2:-2,:]
    if hmi_ic.shape[0] == 4102:
        hmi_ic = hmi_ic[2:-2,2:-2,:]
    if hrt_mu_arr.shape[0] == 4102:
        hrt_mu_arr = hrt_mu_arr[2:-2,2:-2,:]
    if hmi_mu_arr.shape[0] == 4102:
        hmi_mu_arr = hmi_mu_arr[2:-2,2:-2,:]

    size = hrt_mag.shape[0]

    hmi_umbra_mask = hmi_ic <= ic_threshold
    hrt_umbra_mask = hrt_ic <= ic_threshold

    #think this is wrong
    if filter_bleft_corner:
        not_nans = ~np.isnan(hrt_ic)
        # Find Y indices where values are valid
        y_indices, x_indices, z_indices = np.where(not_nans)

        # Initialize min_y with NaN (default for empty slices)
        min_y = np.full(time_steps, np.nan)  # Shape (128,), one min Y per slice
        min_x = np.full(time_steps, np.nan)  # Shape (128,), one min Y per slice

        # Vectorized method to compute min Y for each slice
        for z in np.unique(z_indices):  # Iterate only over slices with valid data
            min_y[z] = np.min(y_indices[z_indices == z])  # Find min Y for slice z
            min_x[z] = np.min(x_indices[z_indices == z])

        y_coords = np.arange(size)[:, None, None]  # Shape (4098, 1, 1)
        x_coords = np.arange(size)[None, :, None]  # Shape (1, 4098, 1)

        # Expand min_y and min_x to match dimensions
        min_y_expanded = min_y[None, None, :] + 10  # Shape (1, 1, 128)
        min_x_expanded = min_x[None, None, :] + 10  # Shape (1, 1, 128)

        # Create the mask: True where (y, x) is greater than (min_y + 20, min_x + 20)
        mask = (y_coords >= min_y_expanded) & (x_coords >= min_x_expanded)
        hrt_umbra_mask &= mask
    
    hrt_noise_mask = np.abs(hrt_mag) > noise_level
    hmi_noise_mask = np.abs(hmi_mag) > noise_level
    
    valid_hrt_mask = hrt_umbra_mask & hrt_noise_mask
    valid_hmi_mask = hmi_umbra_mask & hmi_noise_mask
    
    hrt_unsigned_flux = np.nansum(np.abs(hrt_mag) * valid_hrt_mask, axis=(0, 1))
    hmi_unsigned_flux = np.nansum(np.abs(hmi_mag) * valid_hmi_mask, axis=(0, 1))
    
    hrt_mu = np.where(valid_hrt_mask, hrt_mu_arr, np.nan)
    hmi_mu = np.where(valid_hmi_mask, hmi_mu_arr, np.nan)
    
    hrt_mus = np.nanmean(hrt_mu, axis=(0, 1))
    hmi_mus = np.nanmean(hmi_mu, axis=(0, 1))
    
    hrt_unsigned_flux_mu_corr = np.nansum(np.abs(hrt_mag / hrt_mu) * valid_hrt_mask, axis=(0, 1))
    hmi_unsigned_flux_mu_corr = np.nansum(np.abs(hmi_mag / hmi_mu) * valid_hmi_mask, axis=(0, 1))
    
    return (hrt_unsigned_flux, hmi_unsigned_flux), (hrt_mus, hmi_mus), (hrt_unsigned_flux_mu_corr, hmi_unsigned_flux_mu_corr)