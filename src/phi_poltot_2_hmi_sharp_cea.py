"""
Poltot on CEA grid

1. Load Stokes maps
2. Load corresponding HMI sharp map (can be just Br)
3. Compute poltot
4. Apply HMI PSF?
5. Remap poltot onto HMI cea grid
6. Save CEA poltot map as file
"""

from sunpy.map.header_helper import make_fitswcs_header
import sunpy.map
from sunpy.coordinates import propagate_with_solar_surface
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
import sunpy.coordinates
from sunpy.coordinates import frames
import numpy as np
import os
import json
import datetime
from scipy.fftpack import fft2, ifft2, fftshift

from utils import shift_array_multi, fits_get_sampling
from apply_hmi_psf import make_psf_hmi_th
from phi_2_hmi_sharp_cea import reproject_phi_2_hmi_cea, make_cea_hdr_for_phi, update_hdr, calculate_hmi_br_filetimes, get_hmi_cea_br_map

def load_phi_stokes_files(date):
    """
    Load phi stokes files for the given date.
    """
    fdir = f'/scratch/solo/phi/AR_Long_Term_2023_SL/l2/2023-10-{date}/'
    phi_files = os.listdir(fdir)

    phi_stokes_files = [fdir + f for f in phi_files if 'stokes' in f]

    phi_stokes_files.sort()
    return phi_stokes_files


def calculate_poltot(stokes_fp):
    stokes = fits.getdata(stokes_fp)
    _,_,_,cpos = fits_get_sampling(stokes_fp)
    if cpos == 0:
        poltot = (stokes[cpos+1:,1,:,:]**2 + stokes[cpos+1,2,:,:]**2 + stokes[cpos+1,3,:,:]**2)/stokes[cpos+1,0,:,:]**2
    elif cpos == 5:
        poltot = (stokes[:cpos,1,:,:]**2 + stokes[:cpos,2,:,:]**2 + stokes[:cpos,3,:,:]**2)/stokes[:cpos,0,:,:]**2
    poltot = np.mean(poltot, axis=0)


def apply_hmi_psf(poltot, stokes_fp):
    h = fits.getheader(stokes_fp)
    phi_dsun = h['DSUN_OBS']
    hmi_psf = make_psf_hmi_th(poltot.shape[1],phi_dsun)
    hmi_psf /= hmi_psf.max()

    poltot_psf = fftshift(ifft2(fft2(poltot)/poltot.size * fft2(hmi_psf/hmi_psf.sum())).real * poltot.size) 
    return poltot_psf

def make_poltot_map_updated_hdr(poltot, stokes_fp):
    date = stokes_fp.split('T')[0][-2:]
    new_hdr = update_hdr(stokes_fp, date)
    return sunpy.map.Map(poltot, new_hdr)


def get_poltot_cea_map(poltot_map, hmi_br_cea_map):
    cea_hdr_phi = make_cea_hdr_for_phi(hmi_br_cea_map, poltot_map)
    return reproject_phi_2_hmi_cea(poltot_map, cea_hdr_phi)


def apply_shifts(poltot_cea_map, stokes_fp):
    br_file = stokes_fp.replace('stokes', 'cea_Br')
    hdr = fits.getheader(br_file)
    shift = hdr['YXSHIFT']
    if shift is not None:
        print(f'Applying shift: {shift}')
        poltot_cea_shift = shift_array_multi(poltot_cea_map.data, shift, fill_value=np.nan)
        poltot_cea_map = sunpy.map.Map((poltot_cea_shift, poltot_cea_map.fits_header))
    return poltot_cea_map


def save_poltot_cea_maps_to_files(out_dir, stokes_fp, poltot_cea_map, intermediate=False, extension=None):
     # Save the maps to files
    stokes_filename = stokes_fp.split('/')[-1]
    phi_cea_poltot_filename = stokes_filename.replace('stokes','cea_Poltot').replace('.gz','')
    
    if extension is not None:
        phi_cea_poltot_filename = phi_cea_poltot_filename.replace('.fits', f'_{extension}.fits')

    if intermediate:
        phi_cea_poltot_filename = phi_cea_poltot_filename.replace('.fits','_intermediate.fits')
           
    poltot_cea_map.save(out_dir + phi_cea_poltot_filename, overwrite=True)


def main(out_dir, date, num_files=None):

    phi_stokes_files = load_phi_stokes_files(date)
    hmi_br_filetimes, hmi_br_files = calculate_hmi_br_filetimes()

    for stokes_fp in zip(phi_stokes_files[:num_files]):
        print(f'Processing {stokes_fp}')
        poltot = calculate_poltot(stokes_fp)
        hmi_br_cea_map = get_hmi_cea_br_map(hmi_br_filetimes, hmi_br_files, stokes_fp)
        #poltot = apply_hmi_psf(poltot, stokes_fp)
        poltot_map = make_poltot_map_updated_hdr(poltot, stokes_fp)
        poltot_cea_map = get_poltot_cea_map(poltot_map, hmi_br_cea_map)
        poltot_cea_map = apply_shifts(poltot_cea_map, stokes_fp)
        print('Saving poltot map to files')
        save_poltot_cea_maps_to_files(out_dir, poltot_cea_map, intermediate=False, extension=None)


if __name__ == "__main__":
    out_dir = '/scratch/slam/sinjan/arlongterm/phi_cea_maps/'
    main(out_dir, '12', 24)