import echopype as ep
import numpy as np
import argparse

import xarray
import matplotlib.pyplot as plt

from AcousticExport.utils import find_files
from multiprocessing import Queue, Process
import pyproj
from scipy.ndimage import convolve
from scipy.signal import find_peaks
from echopype.core import SONAR_MODELS
from echopype.calibrate.calibrate_ek import CalibrateEK80
from xml.etree import ElementTree
import pickle
import math


def fetch_latitude_longitude(ed: xarray.Dataset, nmea_msg: str = "RMC"):
    idx = ed.platform.sentence_type == nmea_msg
    latitude = ed.platform.latitude[idx]
    longitude = ed.platform.longitude[idx]
    times = ed.beam.ping_time
    latitude = latitude.interp(location_time=times, kwargs={"fill_value": "extrapolate"})
    longitude = longitude.interp(location_time=times, kwargs={"fill_value": "extrapolate"})
    return {"latitude": latitude, "longitude": longitude}


def fetch_cruise_distance(ed: xarray.Dataset, nmea_msg: str = "RMC"):
    """
    Calculate relative cruise distance based on latitude / longitude data.
    :param ed: EchoData object
    :param nmea_msg: Message type to filter on.
    :return: cruise distance DataArray object.
    """
    idx = ed.platform.sentence_type == nmea_msg
    latitude = ed.platform.latitude[idx]
    longitude = ed.platform.longitude[idx]
    times = ed.beam.ping_time
    latitude = latitude.interp(location_time=times, kwargs={"fill_value": "extrapolate"})
    longitude = longitude.interp(location_time=times, kwargs={"fill_value": "extrapolate"})
    G = pyproj.Geod(ellps='WGS84')
    lat_init = latitude[:-1]
    long_init = longitude[:-1]
    lat_end = latitude[1:]
    long_end = longitude[1:]
    _, _, distances = G.inv(long_init, lat_init, long_end, lat_end)
    cruise_distance = np.concatenate((np.zeros(1), distances)).cumsum()
    D = xarray.DataArray(data=cruise_distance, coords=[times],
                         name="cruise_distance",
                         attrs={"units": "m",
                                "long_name": "Relative Cruise Distance Interpolated to Ping Times"})
    return D


def detect_bottom(calibrated: xarray.Dataset, min_depth: float = 0.0, threshold: float = -30.0):
    """
    Simple bottom detection that detects the first threshold dB value after a minimum depth.
    :param calibrated: DataSet object containing calibrated TS or Sv data.
    :param min_depth: Minimum depth to start bottom detection from (m).
    :param threshold: Threshold to detect bottom at (dB).
    :return: Binary array where the bottom is detected (True)
    """
    cbs = get_backscatter_pointer(calibrated)
    R = calibrated.range
    depth_mask = R >= min_depth
    intensity_mask = cbs >= threshold
    idx = (depth_mask * intensity_mask).argmax("range_bin")
    return cbs.range_bin > idx


def plot_slice(v: xarray.DataArray, frequency: int = 0):
    """
    Plots the DataArray of a specific frequency index.
    :param v: DataArray object to plot
    :param frequency: Frequency index to display.
    :return: None
    """
    v.isel(frequency=frequency).transpose().plot(yincrease=False)


def get_backscatter_pointer(calibrated: xarray.Dataset):
    """
    Detect and return either the calibrated Sv or calibrated TS of the Dataset object. Sv prioritized.
    :param calibrated: Dataset object containing Sv or Sp data.
    :return:
    """
    if hasattr(calibrated, "Sv"):
        return calibrated.Sv
    elif hasattr(calibrated, "Sp"):
        return calibrated.Sp
    raise AttributeError("Neither Sv nor Sp found, calibrate first with echopype.")


def best_bottom_candidate(calibrated: xarray.Dataset,
                          frequency: float = None,
                          frequency_idx: int = None,
                          min_depth: float = 0.0,
                          max_depth: float = 1000.0,
                          minimum_sv: float = -70.0,
                          discrimination_level: float = -50.0,
                          use_backstep: bool = True,
                          backstep_range: float = -0.5,
                          peak_threshold: float = -50.0,
                          maximum_dropouts: int = 2,
                          window_radius: int = 8,
                          minimum_peak_asymmetry=-1.0
                          ):
    """

    :param calibrated:
    :param min_depth:
    :param max_depth:
    :param minimum_sv:
    :param discrimination_level:
    :param use_backstep:
    :param backstep_range:
    :param peak_threshold:
    :param maximum_dropouts:
    :param window_radius:
    :param minimum_peak_asymmetry:
    :return:
    """
    # Do algorithm on Sv if possible, with the lowest frequency if not specified otherwise.
    cbs = get_backscatter_pointer(calibrated)
    if frequency is None and frequency_idx is None:
        i = cbs.frequency.argmin("frequency")
        cbs = cbs.isel(frequency=i)
    elif frequency is None:
        cbs = cbs.isel(frequency=frequency_idx)
    else:
        cbs = cbs.sel(frequency=frequency)
    # Step one: The algorithm finds the first ping in the context that has more than one candidate peak.
    # This is done by evaluating all the samples in the ping.
    # The algorithm retains up to six of the highest peaks as candidates.
    peaks = cbs >= peak_threshold


def peak_wrapper(*args):
    peaks, _ = find_peaks(*args)
    idx = np.zeros_like(args[0], dtype=bool)
    idx[peaks] = True
    return idx


def extract_bottom_line(calibrated: xarray.Dataset,
                        min_depth: float = 0.0,
                        threshold: float = None,
                        kernel_size: int = 5,
                        frequency: float = None,
                        frequency_idx: int = None):
    """
    Slightly more advanced bottom detection that slides and erosion filter over the thresholded bottom first and then
    finds the first occurance for each ping.
    :param calibrated: Dataset object containing either Sv or Sp attribute
    :param min_depth: Minimum depth to consider bottom (m)
    :param threshold: Minimum dB to detect bottom (dB)
    :param kernel_size: Size of the kernel to convolve over data, only square supported for now.
    :param frequency: Frequency in Hz to select for bottom detection
    :param frequency_idx: Frequency index to select for bottom detection
    :return: Index array of bottom line.
    """
    cbs = get_backscatter_pointer(calibrated)
    if frequency is None and frequency_idx is None:
        i = cbs.frequency.argmin("frequency")
        cbs = cbs.isel(frequency=i)
    elif frequency is None:
        cbs = cbs.isel(frequency=frequency_idx)
    else:
        cbs = cbs.sel(frequency=frequency)
    # peaks = xarray.apply_ufunc(peak_wrapper, cbs, threshold, input_core_dims=[["range"], []], output_core_dims=[["range"]], vectorize=True)
    intensity_mask = cbs >= threshold if threshold is not None else cbs >= cbs.max()
    R = calibrated.range
    depth_mask = R >= min_depth
    idx = (depth_mask * intensity_mask)
    idx.data = convolve(idx.data, np.ones((kernel_size, kernel_size)), mode="nearest")
    bottom_index = idx.argmax("range")
    bottom_index = bottom_index.where(idx.any("range"), len(idx.range) - 1)
    return bottom_index


def mask_bottom(calibrated: xarray.Dataset, bottom_mask: xarray.DataArray):
    """
    Simply sets values of dataset outside of valid region to be NaN
    :param calibrated: Dataset containing Sv or Sp data.
    :param bottom_mask: DataArray containing True/False for Bottom/NotBottom
    :return: Dataset containing masked data.
    """
    cbs = get_backscatter_pointer(calibrated)
    calibrated["calibrated"] = cbs.where(~bottom_mask)
    return calibrated


def get_bottom_clip(calibrated: xarray.Dataset, bottom_line: xarray.DataArray):
    """
    Clips the data to the valid region above the bottom.
    :param calibrated: Dataset object containing Sv or Sp data.
    :param bottom_line: DataArray object containing index information of bottom.
    :return: DataArray object with the range information to bottom.
    """
    return calibrated.range.isel(range=bottom_line.max())
    return calibrated.range.isel(range=bottom_line).max()
    return calibrated.range.sel(frequency=bottom_line.frequency).isel(range_bin=bottom_line).max()


def clip_to_ranging(calibrated: xarray.Dataset, rmax):
    """
    Removes the data outside of range > rmax
    :param calibrated: Dataset object containing Sv or Sp data.
    :param rmax: maximum range to allow.
    :return: Range clipped Dataset object
    """
    return calibrated.sel(range=slice(0, rmax))


def angular_position(split_r: xarray.DataArray, split_i: xarray.DataArray, angle_sensitivity_alongship: xarray.DataArray, angle_sensitivity_athwartship: xarray.DataArray):
    # split_r = ed.beam.backscatter_r.sel(frequency=ed.beam.frequency[ed.beam.beam_type==17])  # get the splitbeam transducers only
    # split_i = ed.beam.backscatter_i.sel(frequency=ed.beam.frequency[ed.beam.beam_type==17])
    # angle_sensitivity_alongship = ed.beam.angle_sensitivity_alongship.sel(frequency=ed.beam.frequency[ed.beam.beam_type == 17])
    # angle_sensitivity_athwartship = ed.beam.angle_sensitivity_athwartship.sel(frequency=ed.beam.frequency[ed.beam.beam_type == 17])
    # split_r = ed.beam.backscatter_r  # real component of backscatter complex data (64 bit float)
    # split_i = ed.beam.backscatter_i  # imaginary component of backscatter complex data (64 bit float)
    # angle_sensitivity_alongship = ed.beam.angle_sensitivity_alongship
    # angle_sensitivity_athwartship = ed.beam.angle_sensitivity_athwartship

    # angle_offset_alongship = ed.beam.angle_offset_alongship
    # angle_offset_athwartship = ed.beam.angle_offset_athwartship
    # According to data sheet for ES38-18/200-18C combination transducer
    xstr = split_r.isel(quadrant=0)  # Starboard is first sector
    ystr = split_i.isel(quadrant=0)
    xprt = split_r.isel(quadrant=1)  # Port is second sector
    yprt = split_i.isel(quadrant=1)
    xfwd = split_r.isel(quadrant=2)  # Forward is third sector
    yfwd = split_i.isel(quadrant=2)

    mechanical_angle_alongship = xarray.ufuncs.arcsin(1 / (math.sqrt(3) * angle_sensitivity_alongship) *
                                                      (
                                                              xarray.ufuncs.arctan2((xstr * yfwd - xfwd * ystr),
                                                                                    (xfwd * xstr + yfwd * ystr))
                                                              +
                                                              xarray.ufuncs.arctan2((xprt * yfwd - xfwd * yprt),
                                                                                    (xfwd * xprt + yfwd * yprt))
                                                      )
                                                      )
    mechanical_angle_athwartship = xarray.ufuncs.arcsin(1 / angle_sensitivity_athwartship *
                                                        (
                                                                xarray.ufuncs.arctan2((xprt * yfwd - xfwd * yprt),
                                                                                      (xfwd * xprt + yfwd * yprt))
                                                                -
                                                                xarray.ufuncs.arctan2((xstr * yfwd - xfwd * ystr),
                                                                                      (xfwd * xstr + yfwd * ystr))
                                                        )
                                                        )
    # NO SPECIAL SCALING FOR COMPLEX DATA ?
    # angle_alongship_mech = xarray.ufuncs.arcsin(
    #     (2 / (3 ** 0.5) * 180 / 128 * angle_alongship) / angle_sensitivity_alongship) * 180 / math.pi
    # angle_athwartship_mech = xarray.ufuncs.arcsin(
    #     (2 * 180 / 128 * angle_athwartship) / angle_sensitivity_athwartship) * 180 / math.pi
    # Convert to degrees
    # ed.beam = ed.beam.assign({"angle_alongship": mechanical_angle_alongship * 180 / math.pi,
    #                           "angle_athwartship": mechanical_angle_athwartship * 180 / math.pi})
    return mechanical_angle_alongship * 180 / math.pi, mechanical_angle_athwartship * 180 / math.pi


def interpolate_range(calibrated: xarray.Dataset):
    # if "ping_number" not in calibrated.dims:
    #     calibrated = calibrated.swap_dims({"ping_time": "ping_number"})
    cbs = get_backscatter_pointer(calibrated)
    valid_mask = xarray.ufuncs.isnan(cbs).max("frequency").isel(ping_time=0)
    valid_min = calibrated.range[:, :, np.array(~valid_mask)].min()
    valid_max = calibrated.range[:, :, np.array(~valid_mask)].max()
    spacing = calibrated.range[:, :, np.array(~valid_mask)].isel(ping_time=0).diff("range_bin").min()
    query_range = np.arange(valid_min, valid_max, spacing)
    data_dict = {}
    if "Sp" in calibrated.keys():
        S_ranged = []
        for F, R, S in zip(calibrated.frequency, calibrated.range, calibrated.Sp):
            S = S.assign_coords(range=("range_bin", np.array(R.isel(ping_time=0))))
            S = S.swap_dims({"range_bin": "range"})
            S = S.dropna("range")
            S_ranged.append(
                S.interp(range=query_range, kwargs={"fill_value": "extrapolate"}))
        S_ranged = xarray.concat(S_ranged, "frequency")
        data_dict.update({name: value for name, value in calibrated.items() if name not in ["Sv", "Sp"]})
        data_dict.update({S_ranged.name: S_ranged})
    if "Sv" in calibrated.keys():
        S_ranged = []
        for F, R, S in zip(calibrated.frequency, calibrated.range, calibrated.Sv):
            S = S.assign_coords(range=("range_bin", np.array(R.isel(ping_time=0))))
            S = S.swap_dims({"range_bin": "range"})
            S = S.dropna("range")
            S_ranged.append(
                S.interp(range=query_range, kwargs={"fill_value": "extrapolate"}))
        S_ranged = xarray.concat(S_ranged, "frequency")
        data_dict.update({name: value for name, value in calibrated.items() if name not in ["Sv", "Sp"]})
        data_dict.update({S_ranged.name: S_ranged})
    return xarray.Dataset(data_dict)


def gridded_interpolation(calibrated: xarray.Dataset, spacing: list = None):
    """
    Interpolates a dataset to desired spacing in cruise distance and range.
    :param calibrated: Dataset object containing calibrated Sv or Sp data.
    :param spacing: List ordered with spacing for depth (first element) and cruise distance (second element).
    :return:  Dataset object with interpolated even spaced data.
    """
    cbs = get_backscatter_pointer(calibrated)
    valid_mask = xarray.ufuncs.isnan(cbs).max("frequency").isel(cruise_distance=0)
    valid_min = calibrated.range[:, :, np.array(~valid_mask)].min()
    valid_max = calibrated.range[:, :, np.array(~valid_mask)].max()
    if spacing is None:
        spacing = [calibrated.range[:, :, np.array(~valid_mask)][0, 0, :].diff("range_bin").mean(),
                   calibrated.cruise_distance.diff("cruise_distance").mean()]
    query_depth = np.arange(valid_min, valid_max, spacing[0])
    query_cruise = np.arange(calibrated.coords["cruise_distance"].min(), calibrated.coords["cruise_distance"].max(),
                             spacing[1])
    S_gridded = []
    for F, R, S in zip(calibrated.frequency, calibrated.range, cbs):
        S = S.assign_coords(range=("range_bin", np.array(R.isel(cruise_distance=0))))
        S = S.swap_dims({"range_bin": "range"})
        S = S.dropna("range")
        S_gridded.append(
            S.interp(cruise_distance=query_cruise, range=query_depth, kwargs={"fill_value": "extrapolate"}))
    S_gridded = xarray.concat(S_gridded, "frequency")
    return xarray.Dataset({S_gridded.name: S_gridded,
                           "temperature": calibrated.temperature,
                           "salinity": calibrated.salinity,
                           "pressure": calibrated.pressure,
                           "sound_speed": calibrated.sound_speed,
                           "sound_absorption": calibrated.sound_absorption,
                           "sa_correction": calibrated.sa_correction,
                           "gain_correction": calibrated.gain_correction,
                           "equivalent_beam_angle": calibrated.equivalent_beam_angle})


def extract_calibration_xml(xml_files: list = None):
    if xml_files is None:
        return
    paths = find_files(xml_files, ".xml", resolve=True)
    cal_params = {}
    for path in paths:
        tree = ElementTree.parse(path)
        for calibration_result in tree.findall('.//CalibrationResults'):
            frequency = int(calibration_result.find('Frequency').text)
            sacorrection = float(calibration_result.find('SaCorrection').text)
            gain = float(calibration_result.find('Gain').text)
            cal_params[frequency] = {"sa": sacorrection, "gain": gain}
    return None


def extract_environment_xml(xml_files: list = None):
    if xml_files is None:
        return
    paths = find_files(xml_files, ".xml", resolve=True)
    env_params = {}
    for path in paths:
        tree = ElementTree.parse(path)
        for calibration_result in tree.findall('.//EnvironmentData'):
            # env_params['sound_speed'] = float(calibration_result.find('SoundVelocity').text)
            # env_params['sound_absorption'] = float(calibration_result.find('AbsorptionCoefficient').text)
            env_params['temperature'] = float(calibration_result.find('Temperature').text)
            env_params['salinity'] = float(calibration_result.find('Salinity').text)
            env_params['acidity'] = float(calibration_result.find('Acidity').text)
    return env_params


def get_ping_number(ping_times: xarray.DataArray, index_file: str = None):
    if index_file is None:
        return xarray.DataArray(np.arange(len(ping_times)), coords=ping_times, name="ping_number")
    with open(index_file, "rb") as f:
        mapping = pickle.load(f)
    mapping = mapping.set_index("ping_time")
    ping_number = xarray.DataArray(mapping.loc[ping_times].ping_number)
    return ping_number


def process_ed(file: str,
               sonar_model: str,
               waveform_mode: str,
               encode_mode: str,
               backscatter: str = None,
               min_depth: float = 0.0,
               bottom_threshold: float = None,
               kernel_size: int = 5,
               frequency_idx: int = None,
               frequency: float = None,
               calibration_files: list = None,
               index_file: str = None,
               ):
    """
    Process a given .raw file with bottom thresholding and gridded interpolation.
    :param file: path to a .raw file
    :param sonar_model: type of sonar model, see echopype.core.SONAR_MODELS for details
    :param backscatter: type of calibrated backscatter to obtain, can be Sp or Sv. If none then both are extracted.
    :param waveform_mode: type of waveform used in data , can be narrowband CW or wideband BB
    :param encode_mode: type of encoding to present, can be power or complex for CW or complex for BB
    :param spacing: List containing spacing requirements for gridded data, first element is depth spacing (m) and second element is cruise distance spacing (m)
    :param min_depth: For bottom detection, minimum depth to consider as part of the bottom.
    :param bottom_threshold: For bottom detection, minimum dB strength to consider as part of the bottom.
    :param kernel_size: For bottom detection, size of the square kernel used to refine bottom.
    :param frequency_idx: For bottom detection, index of the frequency to apply bottom detection to.
    :param frequency: For bottom detection, value of the frequency (Hz) to apply bottom detection to.
    :param db_threshold: Thresholds the final data to some dB range.
    :param calibration_files: List of paths to calibration files for each frequency.
    :param index_file: Path to a .pkl file containing mappings from ping time to ping number.
    :param exclude_pings: List of ping numbers/ranges to mask.
    :return: Dataset object that has been processed.
    """
    # DO ALL THE THINGS NECESSARY WITH ED THEN DELETE ED
    ed = ep.open_raw(file, sonar_model=sonar_model)
    # Calculate angular position
    mechanical_angle_alongship, mechanical_angle_athwartship = angular_position(ed.beam.backscatter_r, ed.beam.backscatter_i, ed.beam.angle_sensitivity_alongship, ed.beam.angle_sensitivity_athwartship)
    ed.beam = ed.beam.assign({"angle_alongship": mechanical_angle_alongship * 180 / math.pi,
                              "angle_athwartship": mechanical_angle_athwartship * 180 / math.pi})
    # Assign cruise distance and ping number as alternative coordinates
    ed.beam = ed.beam.assign_coords({"ping_number": get_ping_number(ed.beam.ping_time, index_file), })
    # "cruise_distance": fetch_cruise_distance(ed)})
    # Extract calibration parameters
    cal_params = extract_calibration_xml(calibration_files)
    # Extract environment parameters
    env_params = extract_environment_xml(calibration_files)
    if backscatter is None:
        Sp = ep.calibrate.compute_Sp(ed, waveform_mode=waveform_mode, encode_mode=encode_mode, env_params=env_params,
                                     cal_params=cal_params)
        Sv = ep.calibrate.compute_Sv(ed, waveform_mode=waveform_mode, encode_mode=encode_mode, env_params=env_params,
                                     cal_params=cal_params)
        calibrated = xarray.combine_by_coords((Sp, Sv))
        del Sp, Sv
    else:
        # Get the calibrated calibrated/Sp
        calibrated = ep.calibrate.api._compute_cal(backscatter, ed, waveform_mode=waveform_mode,
                                                   encode_mode=encode_mode,
                                                   env_params=env_params, cal_params=cal_params)
    calibrated = calibrated.assign(fetch_latitude_longitude(ed))
    # TODO beam width alongship & athwartship
    calibrated = calibrated.assign({"angular_position_alongship": ed.beam.angle_alongship,
                                    "angular_position_athwartship": ed.beam.angle_athwartship,
                                    "beamwidth_twoway_alongship": ed.beam.beamwidth_twoway_alongship,
                                    "beamwidth_twoway_athwartship": ed.beam.beamwidth_twoway_athwartship})
    del ed
    # Right now, don't do any cruise distance-based interpolation.
    # calibrated = calibrated.swap_dims({"ping_time": "cruise_distance"})
    # Interpolate the calibrated onto a spatial grid with required resolution
    calibrated = interpolate_range(calibrated)
    calibrated = calibrated.drop_dims(["range_bin"])
    calibrated = calibrated.drop_vars(["location_time"])
    # calibrated = gridded_interpolation(calibrated, spacing=spacing)

    if bottom_threshold is not None:
        # Detect Bottom
        bottom_line = extract_bottom_line(calibrated, min_depth=min_depth,
                                          threshold=bottom_threshold,
                                          kernel_size=kernel_size,
                                          frequency_idx=frequency_idx,
                                          frequency=frequency)
        calibrated = calibrated.assign({"bottom_detection": bottom_line})
        # Find range to bottom
        rmax = get_bottom_clip(calibrated, bottom_line)
        # Clip the grid to the bottom range
        calibrated = clip_to_ranging(calibrated, rmax)

    # # Right now, don't apply thesholding
    # # Apply Lower and Upper Thresholding
    # if db_threshold is not None:
    #     db_threshold.sort()
    #     if hasattr(calibrated, "Sv"):
    #         calibrated["Sv"] = calibrated.Sv.clip(*db_threshold)
    #     elif hasattr(calibrated, "Sp"):
    #         calibrated["Sp"] = calibrated.Sp.clip(*db_threshold)
    #     else:
    #         raise AttributeError("No Sv or Sp found, calibrated dataset first!")
    return calibrated


def mp_process_ed(jobqueue: Queue, resultqueue: Queue):
    while True:
        try:
            task = jobqueue.get()
            if task is None:
                return
            resultqueue.put((process_ed(str(task[0]), **task[1]), task))
        except MemoryError as e:
            print("Memory error, could not process file: {} to desired resolution.".format(task[0]))


def mp_save_grid(resultqueue: Queue):
    while True:
        task = resultqueue.get()
        if task is None:
            return
        save_grid(task)


def save_grid(data: tuple):
    calibrated = data[0]
    path = data[1][0]
    output = path.parent.joinpath("echopype")
    output.mkdir(exist_ok=True, parents=True)
    print("Saving file: {}".format(output.joinpath(path.with_suffix(".nc").name)))
    calibrated.to_netcdf(str(output.joinpath(path.with_suffix(".nc").name)), engine="h5netcdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_files", nargs="+", type=str,
                        help="Space delimited list of .raw files or directories containing .raw files.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Flag to search directories recursively.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers to process echograms with.")
    parser.add_argument("--sonar_model", type=str, default="EK80", choices=list(SONAR_MODELS.keys()),
                        help="Choose sonar model (check echopype documentation).")
    parser.add_argument("--waveform_mode", type=str, default="BB", choices=["BB", "CW"],
                        help="Choose either wide band BB or narrow band CW.")
    parser.add_argument("--encode_mode", type=str, default="complex", choices=["power", "complex"],
                        help="Choose either power or complex encoding.")
    parser.add_argument("--backscatter", type=str, default=None, choices=["Sp", "Sv"],
                        help="Choose either backscattering strength (Sp) or volume backscattering strength (Sv), default is to extract both.")
    parser.add_argument("--bottom_threshold", type=float, default=None,
                        help="Minimum bottom Sv for thresholding (dB).")
    parser.add_argument("--min_depth", type=float, default=5, help="Minimum bottom depth for thresholding (m).")
    # parser.add_argument("--max_depth", type=float, default=1000, help="Maximum bottom depth for thresholding (m).")
    # parser.add_argument("--use_backstep", action="store_true", help="Use backstepping for bottom detection.")
    # parser.add_argument("--discrimination_level", type=float, default=-50, help="The value at which the bottom depth is found on the candidate peak."
    #                                                                             " The bottom depth is the intersection of the Discrimination level with the rise side of the peak.")
    # parser.add_argument("--backstep_range", type=float, default=-0.5, help="The offset above the Discrimination level.")
    # parser.add_argument("--peak_threshold", type=float, default=-50, help="Samples with values greater than the Peak threshold are considered to be part of a candidate peak.")
    # parser.add_argument("--max_dropouts", type=int, default=2, help="The maximum of number of contiguous samples in a"
    #                                                                 " candidate peak allowed to fall below the Peak threshold. "
    #                                                                 "More than one group of dropouts can occur in a peak.")
    # parser.add_argument("--window_radius", type=int, default=8, help="The distance on either side of the range for a candidate peak that is searched in the adjacent ping.")
    # parser.add_argument("--minimum_peak_asymmetry", type=float, default=-1.0, help="The ratio of a measure of the gradients of the rise and decay sides of the candidate peak."
    #                                                                                "Candidate peaks that have a ratio less than the Minimum peak asymmetry are rejected.")
    parser.add_argument("--frequency", type=float, default=None,
                        help="Frequency to threshold bottom on (default is lowest frequency)")
    # parser.add_argument("--db_threshold", type=float, nargs=2, default=None,
    #                     help="Max and min for dB thresholding. Default is none applied.")
    parser.add_argument("--calibration_files", nargs="+", type=str, default=None,
                        help="Paths to calibration .xml files. Default is none, use calibration values extracted from .raw")
    parser.add_argument("--index_file", type=str, default=None,
                        help="Path to a pickled index file containing mappings from ping_time to ping_number.")
    # parser.add_argument("--exclude_pings", nargs="+", type=str, default=None,
    #                     help="Specify ping numbers to exclude (space delimited), can be specified as single numbers or a range (e.g. 1-42)")
    args = parser.parse_args()
    kws = vars(args)
    paths = find_files(kws.pop("raw_files"), ".raw", kws.pop("recursive"))
    jobqueue = Queue()
    resultqueue = Queue(maxsize=1)
    # Need to index the files via ping time first (extract ping number)
    # input: set of raw files -> pointer to raw file + range of ping numbers
    process_workers = [Process(target=mp_process_ed, args=(jobqueue, resultqueue)) for _ in range(kws.pop("workers"))]
    save_workers = [Process(target=mp_save_grid, args=(resultqueue,)) for _ in range(1)]
    [p.start() for p in process_workers]  # spin up the echoype processing workers
    [p.start() for p in save_workers]  # spin up the netcdf4 saving workings
    [jobqueue.put((p, kws)) for p in paths]  # pass raw files and preprocessing parameters into processing workers queue
    [jobqueue.put(None) for _ in process_workers]  # terminate command for process workers
    [p.join() for p in process_workers]  # wait for the process workers to stop
    [resultqueue.put(None) for _ in save_workers]  # terminate command for save workers
    [s.join() for s in save_workers]  # wait for the save workers to stop


if __name__ == "__main__":
    main()
