import itertools
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterable
import os, sys

def imerge(a: Iterable, b: Iterable) -> Iterable:
    """
    Inteeleaved merge

    Args:
        a (Iterable): The first iterable.
        b (Iterable): The second iterable.

    Yields:
        Elements from both iterables in an interleaved manner.
    """
    for i, j in zip(a, b):
        yield i
        yield j

@contextmanager
def cd(newdir: Path | str):
    """
    Context manager approach to move into and then out of a given directory. Creates the directory if it does not exist.

    Args:
        newdir (Path or str): The directory to move into.

    Yields:
        None: The context manager does not yield any value.
    """
    prevdir = Path.cwd()
    os.mkdir(newdir ) if not Path(newdir).exists() else None
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)

def normalise_string_responses(responses: list[str]) -> list[str]:
    """
    Normalises a list of string responses by stripping whitespace and converting to lowercase.

    Args:
        responses (list of str): List of string responses to normalise.

    Returns:
        list of str: Normalised responses (lowe case, no dots)

    """
    return [response.strip().lower().replace(".", "") for response in responses]