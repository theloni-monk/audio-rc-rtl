from dataclasses import dataclass
from functools import partial
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import cocotb
import cocotb.binary
import cocotb.types
import cocotb.types
import numpy as np
from cocotb.triggers import (ClockCycles, FallingEdge, First, Join, ReadOnly,
                             RisingEdge, Timer, with_timeout)
from cocotb_bus.drivers import BusDriver
from cocotb_bus.monitors import BusMonitor
from fixedpoint import FixedPoint

import onnx
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator

from .ctb_util import *
from .msgtypes import BspkVectorMsg

def np2fxp(arr: np.ndarray, fmt = 'q2.6') -> List[FixedPoint]:
    if fmt != 'q2.6':
        raise NotImplementedError
    out = []
    if len(arr.shape) == 1:
        if isinstance(arr[0], (np.float16, np.float32, np.float64)):
            out = [FixedPoint(float(el), m=2, n=6) for el in arr]
        elif isinstance(arr[0], (np.int8, np.int16, np.int32, np.int64)):
            out = [FixedPoint(int(el), m=2, n=6) for el in arr]
    else:
        raise NotImplementedError

    return out

def fxp2np(arr: List[FixedPoint]) -> np.array:
    return np.array([float(x) for x in arr], dtype=np.float32)


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

        insig = cocotb.binary.LogicArray("".join([f"{fxp:08b}" for fxp in fxparr]))
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
    def __init__(self, fifo, name, clk, mon_out = True, debug=False, **kwargs):
        self.debug = debug
        self._signals = ['wr_en', 'wr_data', 'rd_en', 'rd_data', 'wrap_rd']
        self.mon_out = mon_out
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
            en = self.dut.wr_en if self.mon_out else self.dut.rd_en
            data = self.dut.wr_data if self.mon_out else self.dut.rd_data
            if en.value:
                bbuff = data.value.buff
                fxpbuff = [FixedPoint(f"0b{brep:08b}", signed=True, m=2, n=6) for brep in bbuff]
                self.fxp_recvd_buf.append(fxpbuff)
                npbuf = fxp2np(fxpbuff)
                self.np_recvd_buf.append(npbuf)
                if self.debug:
                    print(self.name,"RECV:", bbuff, "=>", npbuf)
                self._recv(BspkVectorMsg(npbuf, f"{self.name}_{len(self.np_recvd_buf)}_msg"))

def onnx_infer(x, sess):
    x = np.expand_dims(x, axis=0)
    # print("x", x, x.shape)
    return sess.run(["Y"], {"X": x})

class BspkModuleTester(ABC):
    def __init__(self, dut, modelpath, in_dim, scoreboard_class, debug=False):
        self.dut = dut
        self.in_dim = in_dim
        self.debug = debug

        self.expectedbuf = []
        self.onnx_sess = ort.InferenceSession(modelpath)
        self.model = lambda x: self.expectedbuf.append(onnx_infer(x.data, self.onnx_sess))

        self.ind = BspkDriver(dut, 'top_in_drv', dut.clk_in, debug=debug)
        self.inm = BspkFIFOMonitor(dut.v_fifo_in_top, 'top_in_mon', dut.clk_in, debug=debug, callback=self.model)
        self.outm = BspkFIFOMonitor(dut.v_fifo_out_top, 'top_out_mon', dut.clk_in, debug=debug)

        self.scoreboard = scoreboard_class(self.dut, fail_immediately=False)
        self.scoreboard.add_interface(self.outm, self.expectedbuf)

    def _init_clks(self):
        cocotb.start_soon(generate_clock(self.dut.clk_in))

    async def startup(self):
        self._init_clks()
        await reset(self.dut.rst_in, self.dut.clk_in)
        await ClockCycles(self.dut.clk_in, 500)
        self.inm.dut.rd_en.value = 0
        self.outm.dut.rd_en.value = 0

    def stop(self):
        self.inm.stop()
        self.outm.stop()
        self.ind.stop()

    @abstractmethod
    def _gen_indata(self):
        pass

    async def run_io_test(self, n_io_pairs):
        for _ in range(n_io_pairs - 1):
            msg = self._gen_indata()
            if self.debug:
                print("RUNNING IO INSTANCE ON", msg)
            self.ind.append(msg)
            await ClockCycles(self.dut.clk_in, 1)

        msg = np.zeros(self.in_dim)
        self.ind.append(msg)
        await ClockCycles(self.dut.clk_in, 1)