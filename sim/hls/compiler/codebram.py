import os
from typing import List

import numpy as np


from fixedpoint import FixedPoint

def np2fxp(arr: np.ndarray, fmt = 'q2.6') -> List[List[FixedPoint]] | List[FixedPoint]:
    if fmt != 'q2.6':
        raise NotImplementedError
    out = []
    if len(arr.shape) == 1:
        if isinstance(arr[0], (np.float16, np.float32, np.float64)):
            out = [FixedPoint(float(el), m=2, n=6) for el in arr]
        elif isinstance(arr[0], (np.int8, np.int16, np.int32, np.int64)):
            out = [FixedPoint(int(el), m=2, n=6) for el in arr]
    elif len(arr.shape) == 2:
        if isinstance(arr[0], (np.float16, np.float32, np.float64)):
            out =[[FixedPoint(float(el), m=2, n=6) for el in row] for row in arr]
        elif isinstance(arr[0], (np.int8, np.int16, np.int32, np.int64)):
            out = [[FixedPoint(int(el), m=2, n=6) for el in row] for row in arr]
    else:
        raise NotImplementedError

    return out

def write_bram_file(path, chunk_size, fxplist):
    os.makedirs("/".join(path.split("/")[:-1]),exist_ok=True)
    with open(path, 'w') as f:
        if isinstance(fxplist[0], list): # matrix
            for x in range(len(fxplist)): # row major
                for y in range(len(fxplist[0]) // chunk_size):
                    for i in range(chunk_size):
                        f.write(f'{fxplist[x, y*chunk_size+i]:02x}')
                    f.write('\n')
        else: # vector
            for x in range(len(fxplist) // chunk_size):
                for i in range(chunk_size):
                    f.write(f'{fxplist[x*chunk_size+i]:02x}')
                f.write('\n')

def gen_bram_file(path, chunk_size, npdata):
    fxpdata = np2fxp(npdata)
    write_bram_file(path, chunk_size, fxpdata)