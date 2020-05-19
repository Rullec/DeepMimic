from zipfile import BadZipFile

import numpy as np
import os


def load_buffer_npz(file_path):
    dirs = None
    if os.path.exists(file_path):
        dirs = os.listdir(file_path)

    buffer = []
    for _id, p in enumerate(dirs):
        b = None
        try:
            b = np.load(os.path.join(file_path, p))
        except BadZipFile:
            b = None
        if b is not None:
            buffer.append(b)
    return buffer


def save_buffer_npz(buffer, file_path, compressed=True):
    if compressed:
        np.savez_compressed(file_path, **buffer)
    else:
        np.savez(file_path, **buffer)

