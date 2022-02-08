from pathlib import Path
from typing import Union
from os import PathLike
from collections.abc import Iterable


def find_files(search_list: Union[str, Path, PathLike, Iterable], ext: str, resolve: bool = False, recursive: bool = False):
    """
    find_files creates a list of pathlib.Path objects given a list containing strings or Path objects.
    It appends files that end with the desired extension and searches directories (optional recursively).
    :param search_list: list containing strings or Path objects that point to specific files or directories
    :param ext: string of the extension suffix of the file type.
    :param resolve: flag to fully resolve paths to absolute form
    :param recursive: flags to search recursively within given directories
    :return:
    """
    if not ext.startswith("."):
        ext = "."+ext
    if isinstance(search_list, Iterable):
        p = [Path(i) for i in search_list]
    else:
        p = [Path(search_list)]
    paths = []
    for j in p:
        path = Path(j).resolve() if resolve else Path(j)
        if not path.exists():
            continue
        if path.is_dir():
            if recursive:
                paths.extend(path.rglob("*"+ext))
            else:
                paths.extend(path.glob("*"+ext))
        elif str(path).endswith(ext):
            paths.append(path)
    return paths