import json
import os
import tempfile
import numpy as np
from starlette.datastructures import UploadFile


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def convert_input(form):
    """Returns a new input and a list of files to be cleaned up"""
    new_input = {}
    files = []
    for k, v in form.multi_items():
        if type(v) == UploadFile:
            new_input[k] = _write_file(v, files)
        else:
            new_input[k] = v

    return files, new_input


def _write_file(v, files):
    suffix = os.path.splitext(v.filename)[1]
    named_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix)
    files.append(named_file)
    named_file.write(v.file.read())
    named_file.close()
    return named_file.name