import base64
import functools
import os
import time
import urllib.request
from urllib.parse import urlparse

import numpy as np
import logging


LOGGER = logging.getLogger('api')


def timeit(f):
    @functools.wraps(f)
    def run(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        LOGGER.info("Time exec '{}': {:.3}s".format(f.__name__, end_time - start_time))
        return result

    return run


def compute_sim(embedding1, embedding2, new_range=False):
    emb1 = embedding1.flatten()
    emb2 = embedding2.flatten()
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if new_range:
        sim = (sim + 1) / 2
    return float(sim)


class NoFaceException(Exception):
    def __init__(self, message):
        self.message = message


dfloat32 = np.dtype('>f4')


def encode_array(arr):
    base64_str = base64.b64encode(np.array(arr).astype(dfloat32)).decode("utf-8")
    return base64_str


def download_model(url: str, output_dir: str = None):
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser('~'), '.cv_end_to_end')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        print("File is exists, load file from {}".format(output_path))
        return output_path

    print("Get {} from {}".format(filename, url))
    print("Downloading .....")
    with urllib.request.urlopen(url) as f:
        with open(output_path, 'wb') as output_file:
            output_file.write(f.read())
    return output_path


def normalC(conf):
    if conf < 0:
        conf = 0
    conf += 0.3
    if conf > 1.0:
        conf = 1.0
    return conf
