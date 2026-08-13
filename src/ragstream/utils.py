from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterable
from typing import Dict
import os
import urllib.request
import urllib.error
import logging
import json
from time import sleep

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

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

def check_services(
        ports: Dict[str, int] = {
            "LLM": 8080,
            "GROBID": 8070
        },
        timeout: int = 5) -> bool:
    """
    Check to see whether a local LLM and the GROBID server are running. It assumes that llama.cpp is being used to host the models.

    Args:
        ports (Dict): dictionary of which ports belong to which tool. Default is
            `{"LLM": 8080, "GROBID":8070}`
        timeout (int): number of second before HTTP request timesout. Defaults to 5.

    Returns:
        bool: True is both the LLM and GROBID respond, false otherwise.
    """
    grobid_addr = f"http://localhost:{ports['GROBID']}/api/isalive"
    llm_addr = f"http://localhost:{ports['LLM']}/health"

    grobid_isalive = False
    llm_isalive = False

    try:
        response = urllib.request.urlopen(grobid_addr, timeout=timeout)
        grobid_isalive = response.getcode() == 200
        if not grobid_isalive:
            logging.warning(f"GROBID server responded with {response.getcode()}.")
    except (urllib.error.URLError, ConnectionResetError, TimeoutError):
        logging.warning("GROBID server cannot be reached")
    try:
        response = urllib.request.urlopen(llm_addr, timeout=timeout)
        if response.getcode() == 200:
            data = json.loads(response.read().decode('utf-8'))
            llm_isalive = data.get("status") == "ok"
    except (urllib.error.URLError, ConnectionResetError, TimeoutError):
        logging.warning("LLM (llama.cpp) cannot be reached.")
    except urllib.error.HTTPError as e:
        # HTTP 503 confirms the process is alive, but the model is still loading.
        if e.code == 503:
            logging.info("LLM is live, but model is likely loading (HTTP 503)")
            sleep(timeout)
            return(check_services(ports=ports))
    return llm_isalive and grobid_isalive