Python scripts and jupyter notebooks to analyse the SO/PHI-HRT data from the Active Region Long Term campaign in October 2023

## Campaign Details

### Oct 12 - Oct 20 : the campaign dates. 

Co-coordinators: Jonas Sinjan and Laura Hayes.

### Oct 12 - Oct 17 11:00 

- Tracked one new active region that emerged a few days prior on the backside of the Sun.
- SO/PHI-HRT started tracking just past quadrature, with the AR at disc centre for SO/PHI-HRT, and very close to the limb in SDO/HMI.
- At Oct 17 11:00 this active region was a disc centre for SDO/HMI.

### Oct 17 11:00 - 21:00 

- Campaign interuppted for a polar campaign co-ordinated with Hinode.

### Oct 17 21:00 - Oct 20 00:00 

- Tracked a different active region that again emerged a few days prior. 
- Unfortuneately the pores that emerged did not develop into strong sunspot(s). 

## Codes/Pipelines developed for analysis

### `src/hrt_wcs_corr_pipeline.py` -> Pipeline to correct the WCS keywords in HRT

- reproject HMI Ic onto HRT detector frame
- cross-correlate with HRT icnt and find X-Y translation
- output the CRPIX and CRVAL errors to json files

### `src/reproject_pipeline.py` -> Pipeline to reproject HRT data onto HMI detector frame

- reproject HRT blos or icnt onto HMI detector frame
- uses WCS errors calculated prior
- outputs remapped HRT and target HMI maps as sunpy map objects to pickles

### `src/reproject_mu_pipeline.py` -> Pipeline to reproject HRT $\mu$ map onto HMI detector frame

- identical to `reproject_pipeline.py` but instead computes the $\mu$ value in each HRT pixel
- reprojects these $\mu$ values onto HMI detector frame
- computes the $\mu$ values in the HMI detector frame
- outputs these $\mu$ maps as sunpy map obejects to pickles

### `src/download_all_hmi_files.py` -> Download HMI files that correspond to HRT files from JSOC using drms

- Find light-travel corrected time on EARTH for each input HRT file
- Downloads the desired HMI file such that each HRT file has corresponding HMI file

### Misc files

#### `src/stereo_help` -> helper functions for cross-correlation

#### `src/reproject_funcs.py` -> helper functions for reprojection and $\mu$ computation