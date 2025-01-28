import numpy as np 
from astropy.io import fits
import sunpy
import sunpy.map
from datetime import datetime as dt
import astropy.units as u

#from stereo_help import image_register

from astropy.wcs import WCS
from reproject import reproject_adaptive
import os
#from scipy.ndimage import map_coordinates

def hmi2phi(hmi_map: sunpy.map.Map, phi_map: sunpy.map.Map) -> sunpy.map.Map:
    """remap hmi blos onto phi blos map coords, using reproject_adaptive with custom WCS out header

    Parameters
    ----------
    hmi_map : sunpy.map.Map
        HMI blos map to be remapped
    phi_map : sunpy.map.Map
        PHI blos map with base (target) coords

    Returns
    -------
    hmi_og_size : sunpy.map.Map
        HMI blos map remapped to PHI coords with same size as PHI map (submap)
    """

    out_header = sunpy.map.make_fitswcs_header(
         hmi_map.data.shape, phi_map.reference_coordinate.replicate(rsun=hmi_map.reference_coordinate.rsun),
         scale=u.Quantity(phi_map.scale),
         instrument="HMI",
         observatory="SDO",
         wavelength=hmi_map.wavelength
         )
    out_header['dsun_obs'] = phi_map.coordinate_frame.observer.radius.to(u.m).value
    out_header['hglt_obs'] = phi_map.coordinate_frame.observer.lat.value
    out_header['hgln_obs'] = phi_map.coordinate_frame.observer.lon.value
    
    out_header['crpix1'] = phi_map.fits_header['CRPIX1']
    out_header['crpix2'] = phi_map.fits_header['CRPIX2']
    out_header['crval1'] = phi_map.fits_header['CRVAL1']
    out_header['crval2'] = phi_map.fits_header['CRVAL2']    
    out_header['crota2'] = phi_map.fits_header['CROTA']
    out_header['PC1_1'] = phi_map.fits_header['PC1_1']
    out_header['PC1_2'] = phi_map.fits_header['PC1_2']
    out_header['PC2_1'] = phi_map.fits_header['PC2_1']
    out_header['PC2_2'] = phi_map.fits_header['PC2_2']
    out_header['cdelt1'] = phi_map.fits_header['cdelt1']
    out_header['cdelt2'] = phi_map.fits_header['cdelt2']
    out_WCS=WCS(out_header)
    
    hmi_repro, _ = reproject_adaptive(hmi_map, out_WCS, hmi_map.data.shape)
    hmi_remap = sunpy.map.Map((hmi_repro, out_WCS))
    
    #take the submap of the hmi using hrt coords
    top_right = hmi_remap.world_to_pixel(phi_map.top_right_coord)
    bottom_left = hmi_remap.world_to_pixel(phi_map.bottom_left_coord)

    tr = np.array([top_right.x.value,top_right.y.value])
    bl = np.array([bottom_left.x.value,bottom_left.y.value])
    hmi_og_size =  hmi_remap.submap(bl*u.pix,top_right=tr*u.pix)
    
    return hmi_og_size


def check_if_ic_images_exist(hrt_file: str, hmi_file: str) -> tuple:
    """check if corresponding HRT and HMI continuum intensity images exist, and return the paths
    
    Parameters
    ----------
    hrt_file : str
        HRT blos map path
    hmi_file : str
        HMI blos map path
    
    Returns
    -------
    tuple
        (hrt_file, hmi_file) if both continuum intensity images exist, else raises OSError
    """
    hrt_icnt = hrt_file.replace('blos','icnt')
    hmi_icnt = hmi_file.replace('magnetogram','continuum')
    hmi_icnt = hmi_icnt.replace('.m_','.ic_')
    hmi_icnt = hmi_icnt.replace('/blos_','/ic_')
    try:
        os.path.isfile(hrt_icnt)
        os.path.isfile(hmi_icnt)
    except:
        raise OSError(f'Continuum intensity images not found at attempted locations:\n{hrt_icnt}\n{hmi_icnt}')
    return hrt_icnt, hmi_icnt


def get_hrt_wcs_crval_err(hrt_file: str,hmi_file: str, save_crpix_err:bool = False) -> tuple:    
    """get wcs error in HRT maps using HMI as reference, using the continuum intensity images

    Parameters
    ----------
    hrt_file : str
        HRT blos map path
    hmi_file : str
        HMI blos map path
    save_crpix_err : bool, optional
        whether to return CRPIX1 and CRPIX2 errors in addition to CRVAL1 and CRVAL2 errors (default: False)

    Returns
    -------
    (errx,erry) : tuple of astropy.coordinates.Skycoord
        CRVAL1 error (x direction), CRVAL2 error (y direction) in Helioprojective (I think)
    (s[1],s[0]) : tuple of floats
        CRPIX1 error (x direction), CRPIX2 error (y direction) in HRT pixel units
    """
    print(hrt_file)
    print(hmi_file)
    while 'blos' in hrt_file and 'magnetogram' in hmi_file:
        print('Input files are magnetograms')
        print('Continuum images result in much better HRT WCS CRVAL corrections')
        print('Checking if corresponding continuum intensity images exist')
        try:
            hrt_file, hmi_file = check_if_ic_images_exist(hrt_file,hmi_file)
        except:
            return None
        
    h=fits.getheader(hrt_file)
    hrt_map = sunpy.map.Map(fits.getdata(hrt_file),h)
    hmi_map = sunpy.map.Map(hmi_file)

    hmi_remap = hmi2phi(hmi_map, hrt_map)

    _,s = image_register(hmi_remap.data, hrt_map.data, False, False)
    print(s)

    x_HRT=h['CRPIX1']
    y_HRT=h['CRPIX2']

    x_HMI=h['CRPIX1']-s[1]
    y_HMI=h['CRPIX2']-s[0]

    feature_coordshrt = hrt_map.pixel_to_world(x_HRT * u.pixel, y_HRT * u.pixel)
    feature_coordshmi = hmi_remap.pixel_to_world(x_HMI * u.pixel, y_HMI * u.pixel)

    errx=feature_coordshrt.Tx.value-feature_coordshmi.Tx.value
    erry=feature_coordshrt.Ty.value-feature_coordshmi.Ty.value

    if save_crpix_err:
        return (float(errx),float(erry)), (int(s[1]),int(s[0])) #json does not serialize numpy data types
    else:
        return (float(errx),float(erry))


def get_hrt_remapped_on_hmi(hrt_file: str, hmi_file: str, err: tuple, reproject_args: dict = {'kernel': 'Gaussian', 'kernel_width': 10000,'sample_region_width': 1}) -> sunpy.map.Map:
    """remap HRT blos map to HMI blos map coords with HMI pixel size, masking out the field stop region/apodization areas if PHI map is larger than (1800,1800)

    Parameters
    ----------
    hrt_file : str
        HRT blos map path
    hmi_file : str
        HMI blos map path
    err : tuple
        CRPIX1 and CRPIX2 error in Skycoords
    
    Returns
    -------
    hrt_remap : sunpy.map.Map
        HRT blos map remapped to HMI coords
    hmi_map : sunpy.map.Map
        HMI blos map
    """
    print(dt.now())

    errx=err[0]
    erry=err[1]
    h = fits.getheader(hrt_file)
    h['CRVAL1']=h['CRVAL1']-errx
    h['CRVAL2']=h['CRVAL2']-erry
    tmp = fits.getdata(hrt_file)
    arr=np.zeros(tmp.shape)
    if tmp.shape[0]<=1800 and tmp.shape[1]<=1800:
        arr=tmp
    elif tmp.shape[0]>1800 and tmp.shape[1]>1800:
        arr[150:-150,150:-150]=tmp[150:-150,150:-150]
    hrt_map = sunpy.map.Map(arr,h)

    hmi_map=sunpy.map.Map(hmi_file).rotate()

    out_header = sunpy.map.make_fitswcs_header(
         hmi_map.data.shape, hmi_map.reference_coordinate.replicate(rsun=hrt_map.reference_coordinate.rsun),
         scale=u.Quantity(hmi_map.scale),
         instrument="SO/PHI-HRT",
         observatory="SolO",
         wavelength=hrt_map.wavelength
         )
    out_header['dsun_obs'] = hmi_map.coordinate_frame.observer.radius.to(u.m).value
    out_header['hglt_obs'] = hmi_map.coordinate_frame.observer.lat.value
    out_header['hgln_obs'] = hmi_map.coordinate_frame.observer.lon.value
    
    out_header['crpix1'] = hmi_map.fits_header['CRPIX1']
    out_header['crpix2'] = hmi_map.fits_header['CRPIX2']
    out_header['crval1'] = hmi_map.fits_header['CRVAL1']
    out_header['crval2'] = hmi_map.fits_header['CRVAL2']    
    out_header['PC1_1'] = hmi_map.fits_header['PC1_1']
    out_header['PC1_2'] = hmi_map.fits_header['PC1_2']
    out_header['PC2_1'] = hmi_map.fits_header['PC2_1']
    out_header['PC2_2'] = hmi_map.fits_header['PC2_2']
    out_header['cdelt1'] = hmi_map.fits_header['cdelt1']
    out_header['cdelt2'] = hmi_map.fits_header['cdelt2']
    out_WCS=WCS(out_header)

    hrt_repro, _ = reproject_adaptive(hrt_map, out_WCS, hmi_map.data.shape, conserve_flux=False,kernel=reproject_args['kernel']\
                                ,kernel_width=reproject_args['kernel_width'],sample_region_width=reproject_args['sample_region_width'])

    if tmp.shape[0]<=1800 and tmp.shape[1]<=1800:
        hrt_remap = sunpy.map.Map((hrt_repro, hmi_map.meta))
    elif tmp.shape[0]>1800 and tmp.shape[1]>1800:
        arr_mask=np.zeros(tmp.shape)
        arr_mask[150:-150,150:-150]=1
        mask_map = sunpy.map.Map(arr_mask,h)
        mask_hrt,_= reproject_adaptive(mask_map, out_WCS, hmi_map.data.shape,kernel='Gaussian',sample_region_width=1)
        mask_hrt[mask_hrt<1]=np.nan
        hrt_remap = sunpy.map.Map((hrt_repro*mask_hrt, hmi_map.meta))
    print(dt.now())
    
    return hrt_remap, hmi_map

def get_hmi_hrt_aligned(hrt_file,hmi_file,err:tuple = None):
    """correct the WCS of HRT map using HMI map as reference and then remap HRT map to HMI map coords

    Parameters
    ----------
    hrt_file : str
        HRT blos map path
    hmi_file : str
        HMI blos map path
    err: tuple
        err in HRT CRVAL1 and CRVAL2 WCS keywords in astropy.coorindates.Skycoord, optional
        if None, tries to compute it first by remapping HMI onto HRT frame and cross-correlation

    Returns
    -------
    list
        [hrt_remap,hmi_map] where hrt_remap is the HRT blos map remapped to HMI coords and hmi_map is the HMI blos map
    """
    if err is None:
        err=get_hrt_wcs_err(hrt_file,hmi_file)
    hrt_remap, hmi_map = get_hrt_remapped_on_hmi(hrt_file, hmi_file, err)
    return [hrt_remap,hmi_map]


def mu_angle_arr(hdr,coord=None):
    """
    input
    hdr: header or filename
    coord: pixel for which the mu angle is found (if None: center of the FoV)
    
    output
    mu = cosine of the heliocentric angle
    """
    if type(hdr) is str:
        hdr = fits.getheader(hdr)
    
    center=center_coord(hdr)
    Rpix=(hdr['RSUN_ARC']/hdr['CDELT1'])
    xlen=int(hdr['NAXIS1'])
    ylen=int(hdr['NAXIS2'])
    x = range(xlen)
    y = range(ylen)
    xv, yv = np.meshgrid(x, y)
    coord = np.ones((len(xv.flatten()),2))
    coord[:,0] = xv.flatten()
    coord[:,1] = yv.flatten()

    if coord is None:
        coord = np.asarray([(hdr['PXEND1']-hdr['PXBEG1'])/2,
                            (hdr['PXEND2']-hdr['PXBEG2'])/2])
    else:
        coord = np.asarray(coord,dtype=float)
    
    coord_tmp = coord - center[:2]
    mu = np.sqrt(Rpix**2 - (coord_tmp[:,0]**2 + coord_tmp[:,1]**2)) / Rpix
    return mu


def center_coord(hdr):
    """calculate the center of the solar disk in the rotated reference system
    Parameters
    ----------
    hdr : header
        header of the fits file
    Returns
    -------
    center: [x,y,1] coordinates of the solar disk center (units: pixel)
    """
    pxbeg1 = hdr['PXBEG1']
    pxend1 = hdr['PXEND1']
    pxbeg2 = hdr['PXBEG2']
    pxend2 = hdr['PXEND2']
    coord=np.asarray([hdr['CRPIX1'],
            hdr['CRPIX2'],
           1])

    angle = hdr['CROTA'] # positive angle = clock-wise rotation of the reference system axes 
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    rc = [(pxend1-pxbeg1)/2,(pxend2-pxbeg2)/2] # CRPIX from 1 to 2048, so 1024.5 is the center

    tr = np.asarray([[1,0,rc[0]],[0,1,rc[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-rc[0]],[0,1,-rc[1]],[0,0,1]])
    M = tr @ rot @ invtr

    coord = (M @ coord)[:2]

    # center of the sun in the rotated reference system
    center=np.asarray([coord[0]-hdr['CRVAL1']/hdr['CDELT1']-1,
                       coord[1]-hdr['CRVAL2']/hdr['CDELT2']-1,
                       1])
    # rotation of the sun center back to the original reference system
    angle = -hdr['CROTA'] # positive angle = clock-wise rotation of the reference system axes 
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    
    tr = np.asarray([[1,0,rc[0]],[0,1,rc[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-rc[0]],[0,1,-rc[1]],[0,0,1]])
    M = tr @ rot @ invtr

    center = (M @ center)
    
    return center


def mu_angle_arr_hmi(hdr,coord):
    """
    input
    hdr: header or filename
    coord: pixel for which the mu angle is found
    
    output
    mu = cosine of the heliocentric angle
    """
    if type(hdr) is str:
        hdr = fits.getheader(hdr)
    
    center=center_coord_hmi(hdr)
    Rpix=(hdr['RSUN_OBS']/hdr['CDELT1'])
    
    coord = np.asarray(coord,dtype=float)
    
    coord_tmp = coord - center[:2]
    mu = np.sqrt(Rpix**2 - (coord_tmp[:,0]**2 + coord_tmp[:,1]**2)) / Rpix
    return mu


def center_coord_hmi(hdr):
    """calculate the center of the solar disk in the rotated reference system
    Parameters
    ----------
    hdr : header
        header of the fits file
    Returns
    -------
    center: [x,y,1] coordinates of the solar disk center (units: pixel)
    """
    pxbeg1 = 0
    pxend1 = hdr['NAXIS1']-1
    pxbeg2 = 0
    pxend2 = hdr['NAXIS1']-1
    coord=np.asarray([hdr['CRPIX1'],
            hdr['CRPIX2'],
           1])

    hdr['CROTA2'] = 0
    angle = hdr['CROTA2'] # positive angle = clock-wise rotation of the reference system axes 
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    rc = [(pxend1-pxbeg1)/2,(pxend2-pxbeg2)/2] # CRPIX from 1 to 2048, so 1024.5 is the center

    tr = np.asarray([[1,0,rc[0]],[0,1,rc[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-rc[0]],[0,1,-rc[1]],[0,0,1]])
    M = tr @ rot @ invtr

    coord = (M @ coord)[:2]

    # center of the sun in the rotated reference system
    center=np.asarray([coord[0]-hdr['CRVAL1']/hdr['CDELT1']-1,
                       coord[1]-hdr['CRVAL2']/hdr['CDELT2']-1,
                       1])
    # rotation of the sun center back to the original reference system
    angle = -hdr['CROTA2'] # positive angle = clock-wise rotation of the reference system axes 
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    
    tr = np.asarray([[1,0,rc[0]],[0,1,rc[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-rc[0]],[0,1,-rc[1]],[0,0,1]])
    M = tr @ rot @ invtr

    center = (M @ center)
    
    return center


def get_hrt_MU_remapped_on_hmi(hrt_file, hmi_file, err):
    """remap HRT MU to HMI blos map coords with HMI pixel size, masking out the field stop region/apodization areas if PHI map is larger than (1800,1800)

    Parameters
    ----------
    hrt_file : str
        HRT blos map path
    hmi_file : str
        HMI blos map path
    err : tuple
        CRPIX1 and CRPIX2 error in Skycoords
    
    Returns
    -------
    hrt_remap : sunpy.map.Map
        HRT blos map remapped to HMI coords
    hmi_map : sunpy.map.Map
        HMI blos map
    """
    print(dt.now())

    errx=err[0]
    erry=err[1]
    h = fits.getheader(hrt_file)
    h['CRVAL1']=h['CRVAL1']-errx
    h['CRVAL2']=h['CRVAL2']-erry
    tmp=mu_angle_arr(h).reshape(fits.getdata(hrt_file).shape)
    arr=np.zeros(tmp.shape)
    if tmp.shape[0]<=1800 and tmp.shape[1]<=1800:
        arr=tmp
    elif tmp.shape[0]>1800 and tmp.shape[1]>1800:
        arr[150:-150,150:-150]=tmp[150:-150,150:-150]
    hrt_mu_map = sunpy.map.Map(arr,h)
    hmi_map=sunpy.map.Map(hmi_file).rotate()

    out_header = sunpy.map.make_fitswcs_header(
         hmi_map.data.shape, hmi_map.reference_coordinate.replicate(rsun=hrt_mu_map.reference_coordinate.rsun),
         scale=u.Quantity(hmi_map.scale),
         instrument="SO/PHI-HRT",
         observatory="SolO",
         wavelength=hrt_mu_map.wavelength
         )
    out_header['dsun_obs'] = hmi_map.coordinate_frame.observer.radius.to(u.m).value
    out_header['hglt_obs'] = hmi_map.coordinate_frame.observer.lat.value
    out_header['hgln_obs'] = hmi_map.coordinate_frame.observer.lon.value
    
    out_header['crpix1'] = hmi_map.fits_header['CRPIX1']
    out_header['crpix2'] = hmi_map.fits_header['CRPIX2']
    out_header['crval1'] = hmi_map.fits_header['CRVAL1']
    out_header['crval2'] = hmi_map.fits_header['CRVAL2']    
    out_header['PC1_1'] = hmi_map.fits_header['PC1_1']
    out_header['PC1_2'] = hmi_map.fits_header['PC1_2']
    out_header['PC2_1'] = hmi_map.fits_header['PC2_1']
    out_header['PC2_2'] = hmi_map.fits_header['PC2_2']
    out_header['cdelt1'] = hmi_map.fits_header['cdelt1']
    out_header['cdelt2'] = hmi_map.fits_header['cdelt2']
    out_WCS=WCS(out_header)

    hrt_repro, footprint = reproject_adaptive(hrt_mu_map, out_WCS, hmi_map.data.shape)
    
    #print(tmp.shape)
    
    if tmp.shape[0]<=1800 and tmp.shape[1]<=1800:
        hrt_mu_map = sunpy.map.Map((hrt_repro, hmi_map.meta))
    elif tmp.shape[0]>1800 and tmp.shape[1]>1800:
        arr_mask=np.zeros(tmp.shape)
        arr_mask[150:-150,150:-150]=1
        mask_map = sunpy.map.Map(arr_mask,h)
        mask_hrt,_= reproject_adaptive(mask_map, out_WCS, hmi_map.data.shape)
        mask_hrt[mask_hrt<1]=np.nan
        hrt_mu_map = sunpy.map.Map((hrt_repro*mask_hrt,hmi_map.meta))

    hmi_shape = hmi_map.data.shape
    x = range(hmi_shape[0])
    y = range(hmi_shape[1])
    xv, yv = np.meshgrid(x, y)
    hmi_coords = np.ones((len(xv.flatten()),2))
    hmi_coords[:,0] = xv.flatten()
    hmi_coords[:,1] = yv.flatten()
    hmi_mu = mu_angle_arr_hmi(hmi_map.fits_header, hmi_coords)
    hmi_mu = hmi_mu.reshape(hmi_shape)
    if tmp.shape[0]>1800 and tmp.shape[1]>1800:
        hmi_mu[np.where(np.isnan(mask_hrt))] = np.nan
    hmi_mu_map = sunpy.map.Map((hmi_mu,hmi_map.meta))
    print(dt.now())
    
    return hrt_mu_map, hmi_mu_map

