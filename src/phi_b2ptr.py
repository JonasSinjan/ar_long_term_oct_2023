import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from sunpy.coordinates import frames

def get_stonyhurst_lonlat(wcs, shape):
    """
    Vectorized: Calculate Stonyhurst heliographic longitude and latitude for all pixels.
    
    Args:
        wcs : astropy.wcs.WCS object
        shape : (ny, nx) - shape of image
        
    Returns:
        lon : 2D array (degrees)
        lat : 2D array (degrees)
    """
    ny, nx = shape
    
    # Generate pixel coordinate grids
    xpix, ypix = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Astropy/SunPy expect (pixel centers) => +0.5 offset
    coords = wcs.pixel_to_world(xpix + 0.5, ypix + 0.5)
    
    # Transform from Helioprojective to Heliographic Stonyhurst
    hgs_coords = coords.transform_to(frames.HeliographicStonyhurst())
    
    lon = hgs_coords.lon.to(u.deg).value
    lat = hgs_coords.lat.to(u.deg).value
    
    return lon, lat

def phi_b2ptr(header, bvec, xudong=0):
    """
    Equivalent Python translation of IDL's phi_b2ptr.pro.
    Args:
        header : object containing necessary fields like naxis1, naxis2, crlt_obs, crota, hgln_obs, rsun_ref
        bvec : 3D numpy array with shape (3, nx, ny)
        xudong : int, if 1 calls hmi_b2ptr_for_phi (not implemented here)
    Returns:
        bptr : 3D numpy array of shape (nx, ny, 3) [Bp, Bt, Br (G)
;			Bp is positive when pointing west; Bt is positive when pointing south]
        lonlat : 3D numpy array of shape (nx, ny, 2)
        aa : transformation matrix as a 3x3 array
    """
    if xudong == 1:
        print('Using hmi_b2ptr (not implemented in this script)')
        # resolve and call hmi_b2ptr_for_phi equivalent
        raise NotImplementedError("hmi_b2ptr_for_phi function is not implemented.")
    else:
        print('Using phi_b2ptr')

        # Check dimensions
        if bvec.shape[0] != 3 or bvec.shape[1] != header['NAXIS1'] or bvec.shape[2] != header['NAXIS2']:
            print('Dimension of bvec incorrect')
            return None, None, None
        
        nx, ny = bvec.shape[1], bvec.shape[2]

        # Convert bvec to B_xi, B_eta, B_zeta
        field = bvec[0, :, :]
        gamma = np.deg2rad(bvec[1, :, :])
        psi = np.deg2rad(bvec[2, :, :] + 90)

        b_xi = field * np.sin(gamma) * np.cos(psi)
        b_eta = field * np.sin(gamma) * np.sin(psi)
        b_zeta = field * np.cos(gamma)

        # Set WCS solar radius reference (environment variable handling skipped)
        
        # WCS conversion
        wcs = WCS(header) 

        # Get Stonyhurst lon/lat
        phi, lambd = get_stonyhurst_lonlat(wcs, (nx,ny)) # You must define this - 'Heliographic'

        lonlat = np.zeros((nx, ny, 2), dtype=np.float32)
        lonlat[:, :, 0] = phi
        lonlat[:, :, 1] = lambd

        # Get angles
        b = np.deg2rad(header['CRLT_OBS'])  # disk center latitude
        crota = header['CROTA']       
        p = -np.deg2rad(crota)           # p-angle

        phi0 = header['HGLN_OBS']            # image center longitude

        print('   B0,P,L0 angles [deg]=', np.rad2deg(b), np.rad2deg(p), phi0)

        phi = np.deg2rad(phi - phi0)
        lambd = np.deg2rad(lambd)

        sinb = np.sin(b)
        cosb = np.cos(b)
        sinp = np.sin(p)
        cosp = np.cos(p)
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        sinlam = np.sin(lambd)
        coslam = np.cos(lambd)

        # Compute transformation matrix elements
        a31 = coslam * (sinb * sinp * cosphi + cosp * sinphi) - sinlam * cosb * sinp
        a32 = -coslam * (sinb * cosp * cosphi - sinp * sinphi) + sinlam * cosb * cosp
        a33 = coslam * cosb * cosphi + sinlam * sinb

        a21 = -sinlam * (sinb * sinp * cosphi + cosp * sinphi) - coslam * cosb * sinp
        a22 = sinlam * (sinb * cosp * cosphi - sinp * sinphi) + coslam * cosb * cosp
        a23 = -sinlam * cosb * cosphi + coslam * sinb

        a11 = -sinb * sinp * sinphi + cosp * cosphi
        a12 = sinb * cosp * sinphi + sinp * cosphi
        a13 = -cosb * sinphi

        aa = np.zeros((3, 3, nx, ny))
        aa[0, 0, :, :] = a11
        aa[0, 1, :, :] = a12
        aa[0, 2, :, :] = a13
        aa[1, 0, :, :] = a21
        aa[1, 1, :, :] = a22
        aa[1, 2, :, :] = a23
        aa[2, 0, :, :] = a31
        aa[2, 1, :, :] = a32
        aa[2, 2, :, :] = a33

        # Apply the transformation
        bptr = np.zeros((nx, ny, 3), dtype=float)

        bptr[:, :, 0] = a11 * b_xi + a12 * b_eta + a13 * b_zeta
        bptr[:, :, 1] = a21 * b_xi + a22 * b_eta + a23 * b_zeta
        bptr[:, :, 2] = a31 * b_xi + a32 * b_eta + a33 * b_zeta

        return bptr, lonlat, aa

