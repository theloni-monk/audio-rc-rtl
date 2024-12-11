from abc import ABC, abstractmethod
from typing import List

import cocotb
import cocotb.binary
import cocotb.types
import cocotb.types
import numpy as np
from cocotb.triggers import (ClockCycles, FallingEdge, ReadOnly, RisingEdge)
from cocotb_bus.drivers import BusDriver
from cocotb_bus.monitors import BusMonitor
from fixedpoint import FixedPoint

import onnxruntime as ort

from .ctb_util import *
from .msgtypes import BspkVectorMsg

def np2fxp(arr: np.ndarray, fmt = 'q4.8') -> List[FixedPoint]:
    if fmt != 'q4.8':
        raise NotImplementedError
    out = []
    if arr.ndim == 1:
        if isinstance(arr[0], (np.float16, np.float32, np.float64)):
            out = [FixedPoint(float(el), signed=True, m=4, n=8) for el in arr]
        elif isinstance(arr[0], (np.int8, np.int16, np.int32, np.int64)):
            out = [FixedPoint(int(el), signed=True, m=4, n=8) for el in arr]
    else:
        raise NotImplementedError

    return out

def fxp2np(arr: List[FixedPoint]) -> np.array:
    return np.array([float(x) for x in arr], dtype=np.float32)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class BspkDriver(BusDriver):
    def __init__(self, dut, name, clk, debug=False):
        self.debug = debug
        self._signals = ['in_data_ready', 'in_data_top']
        BusDriver.__init__(self, dut, name, clk)
        self.clock = clk
        self.dut = dut
        self._init_sigs()
        self.n_sent = 0
        self.sent_buf = []

    def _init_sigs(self):
        self.dut.in_data_ready.value = 0
        self.dut.in_data_top.value = 0

    async def _driver_send(self, msg:np.array, sync=True):
        fxparr = np2fxp(msg)
        await RisingEdge(self.clock)

        insig = cocotb.binary.LogicArray("".join([f"{fxp:012b}" for fxp in fxparr]))
        if self.debug:
            print(self.name, "sending bits", insig)
        self.dut.in_data_top.value = insig
        self.dut.in_data_ready.value = 1
        await ClockCycles(self.clock, 1)
        self.dut.in_data_ready.value = 0
        if self.debug:
            print(self.name, "sent data", msg)
        await ClockCycles(self.clock, 1)
        self.n_sent += 1
        self.sent_buf.append(fxparr)

class BspkFIFOMonitor(BusMonitor):
    def __init__(self, fifo, name, clk, mon_in = True, debug=False, **kwargs):
        self.debug = debug
        self._signals = ['wr_en', 'wr_data', 'rd_en', 'rd_data', 'ptr_rst']
        self.mon_in = mon_in
        BusMonitor.__init__(self, fifo, name, clk, **kwargs)
        self.clock = clk
        self.dut = fifo
        self.fxp_recvd_buf = []
        self.np_recvd_buf = []


    async def _monitor_recv(self):
        falling_edge = FallingEdge(self.clock)
        read_only = ReadOnly()
        while True:
            await falling_edge #sometimes see in AXI shit
            await read_only  #readonly (the postline)
            en = self.dut.wr_en if self.mon_in else self.dut.rd_en
            data = self.dut.wr_data if self.mon_in else self.dut.rd_data
            if en.value:
                larr = cocotb.types.LogicArray(data.value)
                # print(list(chunks(larr.binstr, 12)))
                fxpbuff = [FixedPoint(f"0b{lstr}", signed=True, m=4, n=8) for lstr in chunks(larr.binstr, 12)]
                self.fxp_recvd_buf.append(fxpbuff)
                npbuf = fxp2np(fxpbuff)
                self.np_recvd_buf.append(npbuf)
                if self.debug:
                    print(self.name,"RECV:", list(chunks(larr.binstr, 12)), "=>", npbuf)
                self._recv(BspkVectorMsg(npbuf, f"{self.name}_{len(self.np_recvd_buf)}_msg"))

def onnx_infer(x, sess):
    x = np.expand_dims(x, axis=0)
    # print("x", x, x.shape)
    return sess.run(["Y"], {"X": x})


async def tie(s, cond, clk):
    while True:
        if cond.value:
            s.value = 1
        else:
            s.value = 0
        await ClockCycles(clk, 1)

class BspkModuleTester(ABC):
    def __init__(self, dut, modelpath, in_dim, board, integrate_chunksout, debug):
        self.dut = dut
        self.in_dim = in_dim
        self.debug = debug
        self.integrate_chunksout = integrate_chunksout
        self.expectedbuf = []
        self.onnx_sess = ort.InferenceSession(modelpath)
        self.model = lambda x: self.expectedbuf.append((np.stack(onnx_infer(x.data, self.onnx_sess)),))

        self.ind = BspkDriver(dut, 'top_in_drv', dut.clk_in, debug=debug)
        self.inm = BspkFIFOMonitor(dut.v_fifo_in_top, 'top_in_mon', dut.clk_in, debug=debug, callback=self.model)
        self.outm = BspkFIFOMonitor(dut.v_fifo_out_top, 'top_out_mon', dut.clk_in, debug=debug)

        self.scoreboard = board
        self.scoreboard.add_interface(self.outm, self.expectedbuf)

    def _init_clks(self):
        cocotb.start_soon(generate_clock(self.dut.clk_in))

    async def startup(self):
        self._init_clks()
        await reset(self.dut.rst_in, self.dut.clk_in)
        await ClockCycles(self.dut.clk_in, 500)
        self.inm.dut.rd_en.value = 0
        self.outm.dut.rd_en.value = 0
        if self.integrate_chunksout:
            cocotb.start_soon(tie( self.outm.dut.rd_en, self.dut.out_data_valid, self.dut.clk_in))

    def stop(self):
        self.inm.stop()
        self.outm.stop()
        self.ind.stop()

    @abstractmethod
    def _gen_indata(self):
        pass

    async def run_io_test(self, n_io_pairs):
        while not self.dut.module_ready.value:
                await ClockCycles(1)
        for _ in range(n_io_pairs - 1):
            msg = self._gen_indata()
            self.ind.append(msg)
            await RisingEdge(self.dut.module_ready)
        msg = np.zeros(self.in_dim, dtype=np.float32)
        self.ind.append(msg)
        await RisingEdge(self.dut.module_ready)
