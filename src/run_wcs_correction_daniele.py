import sys
import os
from wcs_correction import WCS_correction
from astropy.io import fits
import json

sys.path.append(os.path.abspath('./'))

for date in ['13', '14', '15', '16', '17']:

    #load the paths to the hrt fits files
    fdir = f'/scratch/solo/phi/AR_Long_Term_2023_SL/l2/2023-10-{date}/'
    hmi_fdir = '/scratch/slam/sinjan/arlongterm_hmi/blos_45/'

    phi_files = os.listdir(fdir)
    blos_f = [fdir + f for f in phi_files if 'blos' in f]
    blos_f.sort()

    selected_wcs_keywords = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 
                        'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CROTA']

    all_results = {}

    for file in blos_f:

        h = fits.getheader(file)
        start_row = h['PXBEG2']-1
        start_col = h['PXBEG1']-1

        out = WCS_correction(file,"your@email.com",
                                    dir_out='./',
                                    undistortion=False,
                                    logpol=False,
                                    allDID=False,
                                    verbose=False,
                                    deriv=False,
                                    values_only=True,
                                    subregion=(slice(150+start_row,1050+start_row),slice(150+start_col,1050+start_col)),
                                    crota_manual_correction=0.15,
                                    hmi_file=hmi_fdir)
        

        crpix1, crpix2, crval1, crval2 = out[1:5]
        h['CRPIX1'] = crpix1
        h['CRPIX2'] = crpix2
        h['CRVAL1'] = crval1
        h['CRVAL2'] = crval2            
        h['PC1_1'] = out[-2].fits_header['PC1_1']
        h['PC1_2'] = out[-2].fits_header['PC1_2']
        h['PC2_1'] = out[-2].fits_header['PC2_1']
        h['PC2_2'] = out[-2].fits_header['PC2_2']
        h['CROTA'] = out[-2].fits_header['CROTA']

        all_results[h['PHIDATID']] = {key: h[key] for key in selected_wcs_keywords if key in h}

    json_file = f"../hrt_wcs/phi_hrt_wcs_headers_arlongterm_202310{date}.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=4)