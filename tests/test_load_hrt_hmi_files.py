import pytest
import os
from datetime import datetime as dt

from src.load_hrt_hmi_files import HRTandHMIfiles

@pytest.fixture
def loader() -> None:
    return HRTandHMIfiles('','','','','','')

def test_get_all_hrt_files_icnt(loader) -> None:
    """test that only hrt files of given file series and date are loaded and instrument and no duplicates"""
    loader.hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    loader.hrt_input_file_series = 'icnt'
    loader.get_all_hrt_files()

    assert all(['hrt' in file for file in loader.hrt_files])
    assert all(['fdt' not in file for file in loader.hrt_files])
    assert all(['icnt' in file for file in loader.hrt_files])
    assert all(['20231017' in file for file in loader.hrt_files])
    assert len(loader.hrt_files) == len(set(loader.hrt_files))

def test_get_all_hrt_files_blos(loader) -> None:
    """test that only hmi files of given file series and date are loaded and instrument and no duplicates"""
    #test blos as well as icnt above and different date
    loader.hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-12/'
    loader.hrt_input_file_series = 'blos'
    loader.get_all_hrt_files()

    assert all(['hrt' in file for file in loader.hrt_files])
    assert all(['fdt' not in file for file in loader.hrt_files])
    assert all(['blos' in file for file in loader.hrt_files])
    assert all(['20231012' in file for file in loader.hrt_files])
    assert len(loader.hrt_files) == len(set(loader.hrt_files))

def test_get_hrt_date(loader)-> None:
    """test the date is retrieved from HRT dates, and throws AssertionError if not all files have this date"""
    loader.hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    loader.hrt_input_file_series = 'blos'
    loader.get_all_hrt_files()
    loader.get_hrt_date()

    assert loader.hrt_date == '20231017'

    loader.hrt_input_folder = '/data/solo/phi/data/fmdb/l1/groundflat/'
    loader.hrt_input_file_series = 'flat'
    loader.get_all_hrt_files()

    with pytest.raises(AssertionError):
        loader.get_hrt_date()

def test_get_all_hmi_files_m45(loader) -> None:
    """test that only hmi files of given file series and date are loaded and instrument and no duplicates"""
    loader.hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    loader.hmi_input_folder = '/data/slam/sinjan/arlongterm_hmi/blos_45'
    loader.hmi_target_file_series = 'm_45s'

    loader.get_all_hrt_files()
    loader.get_hrt_date()
    loader.get_all_hmi_files()

    assert all(['hmi' in file for file in loader.hmi_files])
    assert all(['m_720s' not in file for file in loader.hmi_files])
    assert all(['m_45s' in file for file in loader.hmi_files])
    assert all(['20231017' in file for file in loader.hmi_files])
    assert len(loader.hmi_files) == len(set(loader.hmi_files))

def test_get_all_hmi_files_ic45(loader) -> None:
    """test that only hmi files of given file series and date are loaded and instrument and no duplicates"""
    #test ic_45s as well as m_45s above and different date
    loader.hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-12/'
    loader.hmi_input_folder = '/data/slam/sinjan/arlongterm_hmi/ic_45'
    loader.hmi_target_file_series = 'ic_45s'

    loader.get_all_hrt_files()
    loader.get_hrt_date()
    loader.get_all_hmi_files()
    assert all(['hmi' in file for file in loader.hmi_files])
    assert all(['ic_720s' not in file for file in loader.hmi_files])
    assert all(['ic_45s' in file for file in loader.hmi_files])
    assert all(['20231012' in file for file in loader.hmi_files])
    assert len(loader.hmi_files) == len(set(loader.hmi_files))

def test_start_end_time_init_non_dt() -> None:
    """test that if a none datetime object is passed for start time, None is set"""
    loader = HRTandHMIfiles('','','','','10:00','')

    assert loader.start_time is None
    assert loader.end_time is None

def test_start_end_time_set_if_none(loader)-> None:
    """test if non datetime object passed, after loading HRT files, the start and 
    end times are set to 00:00 and 24:00 of the date given by the HRT files"""
    loader.hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    loader.hrt_input_file_series = 'icnt'
    loader.hrt_dt_start = '10:00'
    loader.hrt_dt_end = '23:00'
 
    loader.get_all_hrt_files()
    loader.get_hrt_date()
    loader.set_start_end_timechecks()
    assert loader.start_time == dt(2023,10,17,0,0,0)
    assert loader.end_time == dt(2023,10,18,0,0,0)

def test_remove_files_outside_start_end_time(loader,hrt_20231017,hmi_files,correct_hrt,correct_hmi) -> None:
    """test to make sure that no files with datetimes lie outside the start and end time, not taking into account light travel time"""
    loader.hrt_files = hrt_20231017
    loader.hmi_files = hmi_files
    loader.start_time = dt(2023,10,17,2,0,0)
    loader.end_time = dt(2023,10,17,14,29,0,0)
    loader.hmi_target_file_series='m_45s'

    loader.remove_files_outside_start_end_time()
    assert set(loader.hrt_files) == set(correct_hrt)
    assert set(loader.hmi_files) == set(correct_hmi)

def test_number_check_no_files(loader) -> None:
    """test if the check has expected behaviour with valueErorrs if no files found and alerting the user if files are not equal"""
    loader.hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-18/'
    loader.hmi_input_folder = '/data/slam/sinjan/arlongterm_hmi/ic_45'
    loader.hrt_input_file_series = 'icnt'
    loader.hrt_dt_start = '10:00'
    loader.hrt_dt_end = '23:00'
  
    loader.get_all_hrt_files()
    loader.get_hrt_date()
    loader.get_all_hmi_files()
    with pytest.raises(ValueError):
        loader.check_number_hrt_hmi_files()

    #check raise error no hrt files
    loader.hrt_files=[]
    with pytest.raises(ValueError):
        loader.check_number_hrt_hmi_files()

def test_number_check_files_notequal(loader,capfd) -> None:
    """test if the check has expected behaviour with valueErorrs if no files found and alerting the user if files are not equal"""
   
    loader.hrt_files=['file1','file2']
    loader.hmi_files=['file1','file2','file3']
    
    loader.check_number_hrt_hmi_files()
    out, err = capfd.readouterr()
    assert 'Number of HRT and HMI files are not equal' in out
    
def test_create_full_fp(loader) -> None:
    loader.hrt_files = ['file1','file2.fits','file3.fits.gz']
    loader.hmi_files = ['file1','file2.fits','file3.fits.gz']
    loader.hrt_input_folder = '/path/to/hrt/files'
    loader.hmi_input_folder = '/path/to/hmi/files'

    loader.create_full_file_paths()
    assert loader.hrt_fps == ['/path/to/hrt/files/file1','/path/to/hrt/files/file2.fits','/path/to/hrt/files/file3.fits.gz']
    assert loader.hmi_fps == ['/path/to/hmi/files/file1','/path/to/hmi/files/file2.fits','/path/to/hmi/files/file3.fits.gz']