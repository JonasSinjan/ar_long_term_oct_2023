import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy import wcs
import astropy.units as u
import astropy.convolution as convolution
from astropy.coordinates import SkyCoord
import numpy as np
import sunpy.data.sample
import sunpy.map
from sunpy.coordinates import HeliographicCarrington, HeliographicStonyhurst, Heliocentric, Helioprojective, frames
from astropy.io import fits
import matplotlib.colors
from scipy import signal
from scipy import misc
import math


mag_str = "blos"


path_data_phi = "/home/amoreno/Escritorio/comparisonnewdata/v3/solo_L2_phi-fdt-" + mag_str +"_20220308T070009_V202307311648_0243080501.fits"
hmi_mag_path = "/home/amoreno/Dropbox/Doctorado/AlignmentHMI-FDT/Data/HMI_ME_720s/hmi.ME_720s_fd10.20220308_070000_TAI.field.fits"
hmi_inc_path = "/home/amoreno/Dropbox/Doctorado/AlignmentHMI-FDT/Data/HMI_ME_720s/hmi.ME_720s_fd10.20220308_070000_TAI.inclination.fits"
            

hmi_mag_data, hmi_header = fits.getdata(hmi_mag_path, header = True)
hmi_inc_data, hmi_header = fits.getdata(hmi_inc_path, header = True)
hmi_data = hmi_mag_data * np.cos(np.deg2rad(hmi_inc_data))
data_phi, header_new_phi = fits.getdata(path_data_phi, header = True)


area = 1


if area == 1:

    ymin, ymax = 890, 1041
    xmin, xmax = 685, 836

else:
                
    ymin = np.nan



#####Defining maps#########

phi_map = sunpy.map.Map(data_phi, header_new_phi)
hmi_map = sunpy.map.Map(hmi_data, hmi_header)



##################3Selecting pixel in PHI####################
crpix_1 = int(phi_map.meta["CRPIX1"])
crpix_2 = int(phi_map.meta["CRPIX2"])

matrix_phi = np.zeros(((ymax - ymin) * (xmax-xmin)))#Create the matrix to fill in with b values of both magnetograms
matrix_hmi = np.zeros(((ymax - ymin) * (xmax-xmin)))
matrix_std = np.zeros(((ymax - ymin) * (xmax-xmin)))

aux_mat = []
count = 0


for pix_phi_y in range(ymin,ymax): 

        for pix_phi_x in range(xmin, xmax):


                #####################Pixel to x,y intermediate world coordinates ################
                #Comment: The pixel criteria is selected as in fortran (1 to N, instead of 0 to N-1). We need to correct this problem.

                theta_x_phi = header_new_phi['CDELT1'] * (header_new_phi['PC1_1'] * (pix_phi_x - header_new_phi['CRPIX1'] + 1) + header_new_phi['PC1_2'] * (pix_phi_y - header_new_phi['CRPIX2'] + 1))
                theta_y_phi = header_new_phi['CDELT2'] * (header_new_phi['PC2_1'] * (pix_phi_x - header_new_phi['CRPIX1'] + 1) + header_new_phi['PC2_2'] * (pix_phi_y - header_new_phi['CRPIX2'] + 1))

                ######################CRVAL CORRECTIONS#################################
                theta_x_phi = theta_x_phi + header_new_phi['CRVAL1']
                theta_y_phi = theta_y_phi + header_new_phi['CRVAL2']

                #In our case the axis units are arcsec and the reference point is the center of the sun -> Plane projection coordinates

                hpc_coord = phi_map.pixel_to_world(pix_phi_x * u.pix, pix_phi_y * u.pix) #standard astropy

                print('Pix to helioprojective: ', theta_x_phi * u.arcsec, theta_y_phi * u.arcsec)
                print(hpc_coord)

                #Now we define the pixel size to know each corner of the pixel

                theta_x_phi = np.array([theta_x_phi - header_new_phi['CDELT1']/2, theta_x_phi, theta_x_phi + header_new_phi['CDELT1']/2])
                theta_y_phi = np.array([theta_y_phi - header_new_phi['CDELT2']/2, theta_y_phi, theta_y_phi + header_new_phi['CDELT2']/2])

                #Tests: If we want to convert we need to transform the theta x/y to deg
                theta_x_phi_deg = (theta_x_phi * u.arcsec).to("degree")
                theta_y_phi_deg = (theta_y_phi * u.arcsec).to("degree") 

                #Comparison between theta calculated and frame

                # print('Angles in deg: theta_x ',theta_x_phi_deg, 'theta_y ',theta_y_phi_deg)
                a = SkyCoord(theta_x_phi * u.arcsec, theta_y_phi * u.arcsec, frame = phi_map.coordinate_frame) #skycoord object in helioprojective
                # print(a)


                #now we can calculate d, distance between the "feature" (in our case the pixel) and the observer(PHI)

                r_sun = header_new_phi['RSUN_REF'] #meters
                r_sun_arcsec = header_new_phi['RSUN_ARC']#arcsec
                phi_center_sun_distance = header_new_phi['DSUN_OBS']#meters
                phi_distance = phi_center_sun_distance - r_sun #meters

                #We will calculate the distance using  law of cosines; as in the original heliocentric frame conversion

                cos_alpha = np.cos(np.deg2rad(theta_x_phi_deg)) * np.cos(np.deg2rad(theta_y_phi_deg))
                c = phi_center_sun_distance * phi_center_sun_distance - r_sun * r_sun
                b = -2 * phi_center_sun_distance * cos_alpha
                d_phi = ((-1 * b) - np.sqrt(b * b - 4 * c)) / 2 

                #################Now we calculate helio-catesian x, y, z ################

                x_phi = d_phi * np.cos(np.deg2rad(theta_y_phi_deg.value)) * np.sin(np.deg2rad(theta_x_phi_deg.value)) #meters
                y_phi = d_phi * np.sin(np.deg2rad(theta_y_phi_deg.value)) #meters
                z_phi = phi_center_sun_distance - d_phi * np.cos(np.deg2rad(theta_y_phi_deg.value)) * np.cos(np.deg2rad(theta_x_phi_deg.value)) #meters

                #####To compare between helioprojective and heliocentric coordinates conversion#####

                print('x: ', x_phi, 'y: ', y_phi, 'z: ', z_phi)
                a = a.transform_to(frames.Heliocentric) #standard astropy
                print(a)


                ########Last step-> Convert x,y,z to Stonyhurst coord.##############
                #We'll need stonyhurst latitude/long of the observer
                #HGLT_OBS=            1.5219614 / [deg] Stonyhurst latitude B0 angle                     
                #HGLN_OBS=            15.666861 / [deg] Stonyhurst longitude  

                r_phi = np.sqrt(x_phi * x_phi + y_phi * y_phi + z_phi * z_phi)
                theta_phi = np.rad2deg(np.arcsin((y_phi * np.cos(np.deg2rad(header_new_phi['HGLT_OBS'])) + z_phi * np.sin(np.deg2rad(header_new_phi['HGLT_OBS'])))/r_phi))
                phi_phi = header_new_phi['HGLN_OBS'] * u.deg + np.rad2deg(np.arctan2(x_phi, z_phi * np.cos(np.deg2rad(header_new_phi['HGLT_OBS'])) - y_phi * np.sin(np.deg2rad(header_new_phi['HGLT_OBS']))))

                # #####To compare between helioprojective and stonyhurst angles#####

                # print('R: ', r_phi, ' Phi (lon): ', phi_phi, ' Theta (lat): ', theta_phi)

                a = a.transform_to(frames.HeliographicStonyhurst)  #standard astropy
                print('Stonyhurst lon: ', a.lon.deg, 'Stonyhurst lat: ', a.lat.deg, ' Stonyhurst rad: ', a.radius.m)
                print(a)

                ####################################Heliographic Stonyhurst PHI to HMI################################
                #Now we are in a common coord. system, so phi_PHI = phi_HMI and theta_PHI == theta_HMI. We need to correct 
                #the delay between FITS file because they have 20 secs between them.

                hgs_coord = SkyCoord(phi_phi, theta_phi, frame='heliographic_stonyhurst', obstime= header_new_phi['DATE-OBS'])
                new_frame = HeliographicStonyhurst(obstime= hmi_header['DATE-OBS'])
                new_hgs_coord = hgs_coord.transform_to(new_frame)
                # print(new_hgs_coord.lon.deg, new_hgs_coord.lat.deg)

                r_hmi = hmi_header['RSUN_REF']
                # r_hmi = r_phi
                # r_hmi = sunpy.sun.constants.radius.to_value(u.m)
                # r_hmi = a.radius

                theta_hmi = new_hgs_coord.lat.deg * u.deg
                phi_hmi = new_hgs_coord.lon.deg * u.deg

                #we'll need also the stonyhurst lat/lon of the observer
                hmi_header['HGLT_OBS'] = hmi_header['CRLT_OBS']#the latitude is the same as in carrington b0
                #To calculate the longitude we will use the L0 angle of the Earth at the observation time and the carrington longitude in the header
                # hmi_header['HGLN_OBS'] = hmi_header['CRLN_OBS'] - sunpy.coordinates.sun.L0(time = hmi_header['DATE-OBS']).deg l0
                hmi_header['HGLN_OBS'] = hmi_map.heliographic_longitude.value


                ###########################Heliographic Stonyhurst to Heliocentric-Cartesian########################
                #Now we need to go back to a pixel point, so we will do the same procces in the other way.

                x_hmi = r_hmi * np.cos(np.deg2rad(theta_hmi.value)) * np.sin(np.deg2rad(phi_hmi.value - hmi_header['HGLN_OBS']))

                y_hmi = (r_hmi * (np.sin(np.deg2rad(theta_hmi.value)) * np.cos(np.deg2rad(hmi_header['HGLT_OBS'])) 
                        - np.cos(np.deg2rad(theta_hmi.value)) * np.cos(np.deg2rad(phi_hmi.value - hmi_header['HGLN_OBS'])) 
                        * np.sin(np.deg2rad(hmi_header['HGLT_OBS']))))

                z_hmi = (r_hmi * (np.sin(np.deg2rad(theta_hmi)) * np.sin(np.deg2rad(hmi_header['HGLT_OBS'])) 
                        + np.cos(np.deg2rad(theta_hmi)) * np.cos(np.deg2rad(phi_hmi.value - hmi_header['HGLN_OBS'])) 
                        * np.cos(np.deg2rad(hmi_header['HGLT_OBS']))))

                print('HMI X:', x_hmi, 'Y: ', y_hmi, 'Z: ', z_hmi)

                #Definition of the new system using the heliographic-SH coordinates with the angles after rotation and the obs time as in HMI
                b = SkyCoord(phi_hmi, theta_hmi, r_hmi * u.m, obstime = hmi_header['date-obs'], frame = frames.HeliographicStonyhurst)
                b = b.transform_to(hmi_map.coordinate_frame)
                b = b.transform_to(frames.Heliocentric)  #standard astropy
                print(b)
                #######################Heliocentric-Cartesian to Helioprojective-Cartesian####################

                d_hmi = np.sqrt(x_hmi * x_hmi + y_hmi * y_hmi + (hmi_header['DSUN_OBS'] - z_hmi) *(hmi_header['DSUN_OBS'] - z_hmi))
                theta_x_hmi_deg = np.rad2deg(np.arctan2(x_hmi, hmi_header['DSUN_OBS'] - z_hmi))
                theta_y_hmi_deg = np.rad2deg(np.arcsin(y_hmi / d_hmi))

                #Transform the angles to arcsec

                theta_x_hmi = (theta_x_hmi_deg).to("arcsec")
                theta_y_hmi = (theta_y_hmi_deg).to("arcsec")

                theta_x_hmi = theta_x_hmi + hmi_header['CRVAL1']#This step is necessary to set the image to the center of the sun. In HMI case it is 0,0
                theta_y_hmi = theta_y_hmi + hmi_header['CRVAL2']

                #Comparison
                print('HMI: theta_x = ', theta_x_hmi, ', theta_y = ', theta_y_hmi)
                b = b.transform_to(frames.Helioprojective)  #standard astropy
                print(b)
               

                ##########################From Helioprojective-Cartesian to Pixel####################
                #We need to calculate the PCij matrix using the rotation matrix and the CROTA angle
                hmi_header['PC1_1'] = np.cos(np.deg2rad(hmi_header['CROTA2']))
                hmi_header['PC1_2'] = -np.sin(np.deg2rad(hmi_header['CROTA2']))
                hmi_header['PC2_1'] = np.sin(np.deg2rad(hmi_header['CROTA2']))
                hmi_header['PC2_2'] = np.cos(np.deg2rad(hmi_header['CROTA2']))

                pix_hmi_x = hmi_header['PC1_1'] * theta_x_hmi.value / hmi_header['CDELT1'] + hmi_header['PC2_1'] * theta_y_hmi.value / hmi_header['CDELT2'] + hmi_header['CRPIX1'] - 1
                pix_hmi_y = hmi_header['PC1_2'] * theta_x_hmi.value / hmi_header['CDELT1'] + hmi_header['PC2_2'] * theta_y_hmi.value / hmi_header['CDELT2'] + hmi_header['CRPIX2'] - 1

                print('HMI pix_x: ', pix_hmi_x, ' pix_y: ', pix_hmi_y)
                #Comparison
                b = hmi_map.world_to_pixel(b)  #standard astropy
                print(b.x.value, b.y.value)

                
                ###########Select the pixels inside the "contour" of PHI pixel size projected into HMI
                
                suma_hmi = 0
                data_number = 0

                #Check if there is nan in the array              
                array_sum = np.sum(pix_hmi_y)
                array_sum_2 = np.sum(pix_hmi_x)
                array_has_nan = np.isnan(array_sum)
                array_2_has_nan = np.isnan(array_sum_2)

              
                if array_has_nan == False or array_2_has_nan == False: #avoid nans
        
                    for k in range(round(pix_hmi_y[2]), round(pix_hmi_y[0]) + 1):
                        
                        for m in range(round(pix_hmi_x[2]), round(pix_hmi_x[0]) + 1):
                            
                            
                            suma_hmi = suma_hmi + hmi_map.data[k, m]
                            data_number = data_number + 1
                            aux_mat.append(hmi_map.data[k, m])


                    aux_mat = np.array(aux_mat)
                    aux_mat = aux_mat[np.logical_not(np.isnan(aux_mat))]
                    mean_hmi = np.mean(aux_mat)
                    std_hmi = np.std(aux_mat)
                    aux_mat = []

                    ###############Filling matrixes with B values for each PHI pixel and mean for HMI phi size pixels
                    matrix_phi[count] = phi_map.data[pix_phi_y, pix_phi_x]
                    matrix_hmi[count] = mean_hmi
                    matrix_std[count] = std_hmi

                else:

                    matrix_phi[count] = np.nan
                    matrix_hmi[count] = np.nan
                    matrix_std[count] = np.nan

                mean_hmi = 0
                count += 1

                if (count % 1000) == 0:

                    print(count)
                if count == 25200:
                    print(count)

print(np.shape(matrix_phi))
print(count)


#Finally I save them as a txt file
np.savetxt('/home/amoreno/Dropbox/Doctorado/Bibliografía/Charlas/PosterBelfast/matrix_phi_march_45s_fulldisk.txt', matrix_phi)
np.savetxt('/home/amoreno/Dropbox/Doctorado/Bibliografía/Charlas/PosterBelfast/matrix_hmi_march_45s_fulldisk.txt', matrix_hmi)
np.savetxt('/home/amoreno/Dropbox/Doctorado/Bibliografía/Charlas/PosterBelfast/matrix_std_march_45s_fulldisk.txt', matrix_std)


#############Example using meshgrid

# Map shape
ny_phi, nx_phi = phi_map.data.shape
ny_hmi, nx_hmi = hmi_map.data.shape

# Create a pixel grid
x_pixels_hmi, y_pixels_hmi = np.arange(nx_hmi), np.arange(ny_hmi)
x_hmi, y_hmi = np.meshgrid(x_pixels_hmi, y_pixels_hmi)

x_pixels_phi, y_pixels_phi = np.arange(nx_phi), np.arange(ny_phi)
x_phi, y_phi = np.meshgrid(x_pixels_phi, y_pixels_phi)


# Central coordinates of each pixel in Stonyhurst (lat/lon grid)
central_coords_hmi = hmi_map.pixel_to_world(x_hmi * u.pixel, y_hmi * u.phi).transform_to(HeliographicStonyhurst(obstime=hmi_map.date))
central_coords_phi = phi_map.pixel_to_world(x_phi * u.phi, y_phi * u.pixel).transform_to(HeliographicStonyhurst(obstime=phi_map.date))
# Bottom-left coordinates of each pixel in Stonyhurst (lat/lon grid)
bottom_left_coords_hmi = hmi_map.pixel_to_world((x_hmi - 0.5) * u.pixel, (y_hmi - 0.5) * u.pixel).transform_to(HeliographicStonyhurst(obstime=hmi_map.date))
bottom_left_coords_phi = phi_map.pixel_to_world((x_phi - 0.5) * u.pixel, (y_phi - 0.5) * u.pixel).transform_to(HeliographicStonyhurst(obstime=phi_map.date))
# Top-right coordinates of each pixel in Stonyhurst (lat/lon grid)
top_right_coords_hmi = hmi_map.pixel_to_world((x_hmi + 0.5) * u.pixel, (y_hmi + 0.5) * u.pixel).transform_to(HeliographicStonyhurst(obstime=hmi_map.date))
top_right_coords_phi = phi_map.pixel_to_world((x_phi + 0.5) * u.pixel, (y_phi + 0.5) * u.pixel).transform_to(HeliographicStonyhurst(obstime=phi_map.date))
