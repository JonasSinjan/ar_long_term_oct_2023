from astropy.io import fits

def phi_disambig(bazi,bamb,method=2):
    """
    input
    bazi: magnetic field azimut. Type: str or array
    bamb: disambiguation fits. Type: str or array
    method: method selected for the disambiguation (0, 1 or 2). Type: int (2 as Default)
    
    output
    disbazi: disambiguated azimut. Type: array
    """
    # from astropy.io import fits
    if type(bazi) is str:
        bazi = fits.getdata(bazi)
    if type(bamb) is str:
        bamb = fits.getdata(bamb)
    
    disambig = bamb[0]/2**method
    disbazi = bazi.copy()
    disbazi[disambig%2 != 0] += 180
    
    return disbazi