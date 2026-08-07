import itertools
from contextlib import contextmanager
from pathlib import Path
import os, sys

def imerge(a, b):
    """
    Merges two iterables a and b in an interleaved fashion.
    """
    for i, j in zip(a, b):
        yield i
        yield j

@contextmanager
def cd(newdir):
    """
    Context manager approach to move into and then out of a given directory. Creates the directory if it does not exist.
    """
    prevdir = Path.cwd()
    os.mkdir(newdir ) if not Path(newdir).exists() else None
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prevdir)

def normalise_string_responses(responses):
    """
    Normalises a list of string responses by stripping whitespace and converting to lowercase.
    """
    return [response.strip().lower().replace(".", "") for response in responses]