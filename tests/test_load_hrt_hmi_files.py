import pytest
import os
from datetime import datetime as dt

from src.load_hrt_hmi_files import HRTandHMIfiles

def test_get_all_hrt_files_icnt() -> None:
    """test that only hrt files of given file series and date are loaded and instrument and no duplicates"""
    hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    hmi_input_folder = ''
    hrt_input_file_series = 'icnt'
    hmi_target_file_series = ''
    hrt_dt_start = ''
    hrt_dt_end = ''
    tmp = HRTandHMIfiles(hrt_input_folder, hmi_input_folder, hrt_input_file_series, hmi_target_file_series, \
                         hrt_dt_start, hrt_dt_end)
    tmp.get_all_hrt_files()
    assert 'hrt' in all(tmp.hrt_files)
    assert 'fdt' not in all(tmp.hrt_files)
    assert 'icnt' in all(tmp.hrt_files)
    assert '20231017' in all(tmp.hrt_files)
    assert len(tmp.hrt_files) == len(set(tmp.hrt_files))
    del tmp

def test_get_all_hrt_files_blos() -> None:
    """test that only hmi files of given file series and date are loaded and instrument and no duplicates"""
    #test blos as well as icnt above and different date
    hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-12/'
    hmi_input_folder = ''
    hrt_input_file_series = 'blos'
    hmi_target_file_series = ''
    hrt_dt_start = ''
    hrt_dt_end = ''
    tmp = HRTandHMIfiles(hrt_input_folder, hmi_input_folder, hrt_input_file_series, hmi_target_file_series, \
                         hrt_dt_start, hrt_dt_end)
    tmp.get_all_hrt_files()
    assert 'hrt' in all(tmp.hrt_files)
    assert 'fdt' not in all(tmp.hrt_files)
    assert 'blos' in all(tmp.hrt_files)
    assert '20231012' in all(tmp.hrt_files)
    assert len(tmp.hrt_files) == len(set(tmp.hrt_files))

def test_get_hrt_date()-> None:
    """test the date is retrieved from HRT dates, and throws AssertionError if not all files have this date"""
    hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    hmi_input_folder = ''
    hrt_input_file_series = 'blos'
    hmi_target_file_series = ''
    hrt_dt_start = ''
    hrt_dt_end = ''
    tmp = HRTandHMIfiles(hrt_input_folder, hmi_input_folder, hrt_input_file_series, hmi_target_file_series, \
                         hrt_dt_start, hrt_dt_end)
    tmp.get_all_hrt_files()
    tmp.get_hrt_date()
    assert tmp.hrt_date == '20231017'
    del tmp

    hrt_input_folder = '/data/solo/phi/data/fmdb/l1/groundflat/'
    hmi_input_folder = ''
    hrt_input_file_series = 'FLAT'
    hmi_target_file_series = ''
    hrt_dt_start = ''
    hrt_dt_end = ''
    tmp = HRTandHMIfiles(hrt_input_folder, hmi_input_folder, hrt_input_file_series, hmi_target_file_series, \
                         hrt_dt_start, hrt_dt_end)
    tmp.get_all_hrt_files()
    with pytest.raises(AssertionError):
        tmp.get_hrt_date()

def test_get_all_hmi_files_m45() -> None:
    """test that only hmi files of given file series and date are loaded and instrument and no duplicates"""
    hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    hmi_input_folder = '/data/slam/sinjan/arlongterm_hmi/blos_45'
    hrt_input_file_series = ''
    hmi_target_file_series = 'm_45s'
    hrt_dt_start = ''
    hrt_dt_end = ''
    tmp = HRTandHMIfiles(hrt_input_folder, hmi_input_folder, hrt_input_file_series, hmi_target_file_series, \
                         hrt_dt_start, hrt_dt_end)
    tmp.get_all_hrt_files()
    tmp.get_hrt_date()
    tmp.get_all_hmi_files()
    assert 'hmi' in all(tmp.hmi_files)
    assert 'm_720s' not in all(tmp.hmi_files)
    assert 'm_45s' in all(tmp.hmi_files)
    assert '20231017' in all(tmp.hmi_files)
    assert len(tmp.hmi_files) == len(set(tmp.hmi_files))

def test_get_all_hmi_files_ic45() -> None:
    """test that only hmi files of given file series and date are loaded and instrument and no duplicates"""
    #test ic_45s as well as m_45s above and different date
    hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-12/'
    hmi_input_folder = '/data/slam/sinjan/arlongterm_hmi/ic_45'
    hrt_input_file_series = ''
    hmi_target_file_series = 'ic_45s'
    hrt_dt_start = ''
    hrt_dt_end = ''
    tmp = HRTandHMIfiles(hrt_input_folder, hmi_input_folder, hrt_input_file_series, hmi_target_file_series, \
                         hrt_dt_start, hrt_dt_end)
    tmp.get_all_hrt_files()
    tmp.get_hrt_date()
    tmp.get_all_hmi_files()
    assert 'hmi' in all(tmp.hmi_files)
    assert 'ic_720s' not in all(tmp.hmi_files)
    assert 'ic_45s' in all(tmp.hmi_files)
    assert '20231012' in all(tmp.hmi_files)
    assert len(tmp.hmi_files) == len(set(tmp.hmi_files))

def test_start_end_time_init_non_dt() -> None:
    """test that if a none datetime object is passed for start time, None is set"""
    tmp = HRTandHMIfiles('','','','','10:00','')
    assert tmp.start_time is None
    assert tmp.end_time is None

def test_start_end_time_set_if_none()-> None:
    """test if non datetime object passed, after loading HRT files, the start and 
    end times are set to 00:00 and 24:00 of the date given by the HRT files"""
    hrt_input_folder = '/data/solo/phi/data/fmdb/public/l2/2023-10-17/'
    hmi_input_folder = ''
    hrt_input_file_series = 'icnt'
    hmi_target_file_series = ''
    hrt_dt_start = '10:00'
    hrt_dt_end = '23:00'
    tmp = HRTandHMIfiles(hrt_input_folder, hmi_input_folder, hrt_input_file_series, hmi_target_file_series, \
                         hrt_dt_start, hrt_dt_end)
    tmp.get_all_hrt_files()
    tmp.set_start_end_timechecks()
    assert tmp.start_time is dt(2023,10,17,0,0,0)
    assert tmp.end_time is dt(2023,10,18,0,0,0)

def test_remove_files_outside_start_end_time() -> None:
    """test to make sure that no files with datetimes lie outside the start and end time, not taking into account light travel time"""
    pass

def test_number_check_files():
    """test if the check has expected behaviour with valueErorrs if no files found and alerting the user if files are not equal"""

