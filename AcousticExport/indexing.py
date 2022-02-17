import xarray

from AcousticExport.utils import find_files
from argparse import ArgumentParser
import echopype as ep
from multiprocessing import Process, Queue
from echopype.core import SONAR_MODELS
from pathlib import Path
import numpy as np
import pandas as pd


def index_ed(file: str, sonar_model: str) -> xarray.DataArray:
    """

    :param file: path to a .raw file
    :param sonar_model: type of sonar model, see echopype.core.SONAR_MODELS for details
    :return:
    """
    ed = ep.open_raw(file, sonar_model=sonar_model)
    return ed.beam.ping_time


def mp_index_ed(jobqueue: Queue, resultqueue: Queue):
    while True:
        try:
            task = jobqueue.get()
            if task is None:
                return
            resultqueue.put(index_ed(str(task[0]), **task[1]))
        except MemoryError as e:
            print("Memory error, could not process file: {} to desired resolution.".format(task[0]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("raw_files", nargs="+", type=str,
                        help="Space delimited list of .raw files or directories containing .raw files.")
    parser.add_argument("--sonar_model", type=str, default="EK80", choices=list(SONAR_MODELS.keys()), help="Choose sonar model (check echopype documentation).")
    parser.add_argument("-r", "--recursive", action="store_true", help="Flag to search directories recursively.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers to process echograms with.")
    parser.add_argument("-o", "--output", type=str, default=".", help="Path to output index file.")
    args = parser.parse_args()
    kws = vars(args)
    paths = find_files(kws.pop("raw_files"), ".raw", kws.pop("recursive"))
    output = Path(kws.pop("output")).resolve()
    if output.is_dir():
        output = output.joinpath("index.pkl")
    elif not output.suffix.endswith("pkl"):
        output = output.with_suffix(".pkl")
    jobqueue = Queue()
    resultqueue = Queue()
    index_workers = [Process(target=mp_index_ed, args=(jobqueue, resultqueue)) for _ in range(kws.pop("workers"))]
    [p.start() for p in index_workers]  # spin up the echoype indexing workers
    [jobqueue.put((p, kws)) for p in paths]  # pass raw files and preprocessing parameters into indexing workers queue
    [jobqueue.put(None) for _ in index_workers]  # terminate command for index_workers workers
    ping_times = np.concatenate([resultqueue.get().to_index() for _ in paths])
    index = ping_times.argsort()
    LUT = pd.DataFrame({"ping_time": ping_times, "ping_number": index})
    LUT.to_pickle(output)
