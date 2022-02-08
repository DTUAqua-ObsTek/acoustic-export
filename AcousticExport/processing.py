import echopype as ep
import numpy as np
import argparse

import xarray

from .utils import find_files
from multiprocessing import Queue, Process
import pyproj
from scipy.ndimage import convolve
from echopype.core import SONAR_MODELS


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
    intensity_mask = cbs >= threshold if threshold is not None else cbs >= cbs.max()
    R = calibrated.range
    depth_mask = R >= min_depth
    idx = (depth_mask * intensity_mask)
    idx.data = convolve(idx, np.ones((1, kernel_size, kernel_size)), mode="nearest")
    if frequency is None and frequency_idx is None:
        i = idx.frequency.argmin("frequency")
        return idx.isel(frequency=i).argmax("range_bin")
    if frequency is None:
        return idx.isel(frequency=frequency_idx).argmax("range_bin")
    return idx.sel(frequency=frequency).argmax("range_bin")


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
    return calibrated.range.sel(frequency=bottom_line.frequency).isel(range_bin=bottom_line).max()


def clip_to_ranging(calibrated: xarray.Dataset, rmax):
    """
    Removes the data outside of range > rmax
    :param calibrated: Dataset object containing Sv or Sp data.
    :param rmax: maximum range to allow.
    :return: Range clipped Dataset object
    """
    idx = (calibrated.range > rmax).argmax("range").max()
    return calibrated.isel(range=np.arange(idx))


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
        spacing = [calibrated.range[:, :, np.array(~valid_mask)][0,0,:].diff("range_bin").mean(), calibrated.cruise_distance.diff("cruise_distance").mean()]
    query_depth = np.arange(valid_min, valid_max, spacing[0])
    query_cruise = np.arange(calibrated.coords["cruise_distance"].min(), calibrated.coords["cruise_distance"].max(), spacing[1])
    S_gridded = []
    for F, R, S in zip(calibrated.frequency, calibrated.range, cbs):
        S = S.assign_coords(range=("range_bin", np.array(R.isel(cruise_distance=0))))
        S = S.swap_dims({"range_bin": "range"})
        S = S.dropna("range")
        S_gridded.append(S.interp(cruise_distance=query_cruise, range=query_depth, kwargs={"fill_value": "extrapolate"}))
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


def process_ed(file: str,
               sonar_model: str,
               backscatter: str,
               waveform_mode: str,
               encode_mode: str,
               spacing: list = None,
               min_depth: float = 0.0,
               bottom_threshold: float = None,
               kernel_size: int = 5,
               frequency_idx: int = None,
               frequency: float = None,
               db_threshold: list = None
               ):
    """
    Process a given .raw file with bottom thresholding and gridded interpolation.
    :param file: path to a .raw file
    :param sonar_model: type of sonar model, see echopype.core.SONAR_MODELS for details
    :param backscatter: type of calibrated backscatter to obtain, can be Sp or Sv
    :param waveform_mode: type of waveform used in data , can be narrowband CW or wideband BB
    :param encode_mode: type of encoding to present, can be power or complex for CW or complex for BB
    :param spacing: List containing spacing requirements for gridded data, first element is depth spacing (m) and second element is cruise distance spacing (m)
    :param min_depth: For bottom detection, minimum depth to consider as part of the bottom.
    :param bottom_threshold: For bottom detection, minimum dB strength to consider as part of the bottom.
    :param kernel_size: For bottom detection, size of the square kernel used to refine bottom.
    :param frequency_idx: For bottom detection, index of the frequency to apply bottom detection to.
    :param frequency: For bottom detection, value of the frequency (Hz) to apply bottom detection to.
    :param db_threshold: Thresholds the final data to some dB range.
    :return: Dataset object that has been processed.
    """
    # DO ALL THE THINGS NECESSARY WITH ED THEN DELETE ED
    ed = ep.open_raw(file, sonar_model=sonar_model)
    # Get the calibrated calibrated/Sp
    calibrated = ep.calibrate.api._compute_cal(backscatter, ed, waveform_mode=waveform_mode, encode_mode=encode_mode)
    # Calculate cruise distance and set as dimension instead of ping_time
    calibrated.coords["cruise_distance"] = fetch_cruise_distance(ed)
    del(ed)
    if bottom_threshold is not None:
        # Detect Bottom
        bottom_line = extract_bottom_line(calibrated, min_depth=min_depth,
                                          threshold=bottom_threshold,
                                          kernel_size=kernel_size,
                                          frequency_idx=frequency_idx,
                                          frequency=frequency)
        # Find range to bottom
        rmax = get_bottom_clip(calibrated, bottom_line)
    calibrated = calibrated.swap_dims({"ping_time": "cruise_distance"})
    # Interpolate the calibrated onto a spatial grid with required resolution
    calibrated = gridded_interpolation(calibrated, spacing=spacing)
    if bottom_threshold is not None:
        # Clip the grid to the bottom range
        calibrated = clip_to_ranging(calibrated, rmax)
    # Apply Lower and Upper Thresholding
    if db_threshold is not None:
        db_threshold.sort()
        if hasattr(calibrated, "Sv"):
            calibrated["Sv"] = calibrated.Sv.clip(*db_threshold)
        elif hasattr(calibrated, "Sp"):
            calibrated["Sp"] = calibrated.Sp.clip(*db_threshold)
        else:
            raise AttributeError("No Sv or Sp found, calibrated dataset first!")
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
    calibrated.to_netcdf(str(output.joinpath(path.with_suffix(".nc").name)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_files", nargs="+", type=str,
                        help="Space delimited list of .raw files or directories containing .raw files.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Flag to search directories recursively.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers to process echograms with.")
    parser.add_argument("-s", "--spacing", nargs=2, type=float, default=None, help="Spacing for spatial gridded interpolation. First argument is vertical (depth/range spacing), second argument is horizontal (cruise distance spacing) Default is interpolation to first frequency's average range and cruise distance spacing.")
    parser.add_argument("--sonar_model", type=str, default="EK80", choices=list(SONAR_MODELS.keys()), help="Choose sonar model (check echopype documentation).")
    parser.add_argument("--waveform_mode", type=str, default="BB", choices=["BB", "CW"], help="Choose either wide band BB or narrow band CW.")
    parser.add_argument("--encode_mode", type=str, default="complex", choices=["power", "complex"], help="Choose either power or complex encoding.")
    parser.add_argument("--backscatter", type=str, default="Sp", choices=["Sp", "Sv"], help="Choose either backscattering strength (Sp) or volumen backscattering strength (Sv)")
    parser.add_argument("--bottom_threshold", type=float, default=None, help="Minimum bottom strength for thresholding (dB).")
    parser.add_argument("--min_depth", type=float, default=0.0, help="Minimum bottom depth for thresholding (m).")
    parser.add_argument("--frequency", type=float, default=None, help="Frequency to threshold bottom on (default is lowest frequency)")
    parser.add_argument("--db_threshold", type=float, nargs=2, default=None, help="Max and min for dB thresholding. Default is none applied.")
    #parser.add_argument("--calibration_files", nargs="+", type=str, default=None, help="Paths to calibration .xml files. Default is none, use calibration values extracted from .raw")
    #parser.add_argument("--exclude_pings", nargs="+", type=str, default=None, help="Specify ping numbers to exclude (space delimited), can be specified as single numbers or a range (e.g. 1-42)")
    args = parser.parse_args()
    kws = vars(args)
    paths = find_files(kws.pop("raw_files"), ".raw", kws.pop("recursive"))
    jobqueue = Queue()
    resultqueue = Queue(maxsize=1)
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
