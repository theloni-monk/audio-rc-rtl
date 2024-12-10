import os
from pathlib import Path
from typing import List

import numpy as np


from fixedpoint import FixedPoint


def np2fxp(arr: np.ndarray, fmt="q4.8") -> List[List[FixedPoint]] | List[FixedPoint]:
    if fmt != "q4.8":
        raise NotImplementedError
    out = []
    if arr.ndim == 1:
        if isinstance(arr[0], (np.float16, np.float32, np.float64)):
            out = [FixedPoint(float(el),  signed=True, m=4, n=8) for el in arr]
        elif isinstance(arr[0], (np.int8, np.int16, np.int32, np.int64)):
            out = [FixedPoint(int(el), signed=True,  m=4, n=8) for el in arr]
    elif arr.ndim == 2:
        if isinstance(arr[0, 0], (np.float16, np.float32, np.float64)):
            out = [[FixedPoint(float(el), signed=True,  m=4, n=8) for el in row] for row in arr]
        elif isinstance(arr[0, 0], (np.int8, np.int16, np.int32, np.int64)):
            out = [[FixedPoint(int(el),  signed=True, m=4, n=8) for el in row] for row in arr]
    else:
        raise NotImplementedError

    return list(reversed(out))


def write_bram_file(path, chunk_size, fxplist, dims):
    # print(fxplist)
    f = open(path, 'wt')
    if dims == 2:  # matrix
        for x in range(len(fxplist)):  # reversed col major
            for y in range(len(fxplist[0]) // chunk_size):
                for i in range(chunk_size):
                    f.write(f"{fxplist[x][y*chunk_size+i]:03x}")
                f.write("\n")
    elif dims == 1:  # vector
        for x in range(len(fxplist) // chunk_size):
            for i in range(chunk_size):
                f.write(f"{fxplist[x*chunk_size+i]:03x}")
            f.write("\n")
    else:
        raise NotImplementedError
    f.close()


def gen_bram_file(path, chunk_size, npdata):
    # print("writing out", npdata)
    fxpdata = np2fxp(npdata)
    print("WRITING BRAM WITH SHAPE", npdata.shape, npdata.ndim, "TO", path, "CHUNKS", chunk_size)
    write_bram_file(path, chunk_size, fxpdata, npdata.ndim)
