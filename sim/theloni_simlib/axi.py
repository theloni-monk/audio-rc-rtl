import random
from abc import ABC, abstractmethod
from enum import Enum

import cocotb
import numpy as np
from cocotb.triggers import ( FallingEdge, First, Join, ReadOnly, ClockCycles,
                             RisingEdge, Timer, with_timeout)
from cocotb_bus.scoreboard import Scoreboard
from cocotb_bus.drivers import BusDriver
from cocotb_bus.monitors import BusMonitor

from .ctb_util import *

class AXISMonitor(BusMonitor):
    """
    monitors axi streaming bus
    """
    def __init__(self, dut, name, clk, debug=False, **kwargs):
        self.debug=debug
        self._signals = ['axis_tvalid','axis_tready','axis_tlast','axis_tdata','axis_tstrb']
        BusMonitor.__init__(self, dut, name, clk, **kwargs)
        self.clock = clk
        self.dut = dut
        self.i32_recv_buf = np.array([])

    async def _monitor_recv(self):
        """
        Monitor receiver
        """
        falling_edge = FallingEdge(self.clock)
        read_only = ReadOnly() #This is
        while True:
            await falling_edge #sometimes see in AXI shit
            await read_only  #readonly (the postline)
            valid = self.bus.axis_tvalid.value
            ready = self.bus.axis_tready.value
            last = self.bus.axis_tlast.value
            data = self.bus.axis_tdata.value
            try:
                idata  = data.integer if data.integer < 2**31 else data.integer - 2**32
            except ValueError:
                idata = -1
            # self.dut._log.info(f"monitor step %d %d %d %d", valid, ready, last, data)
            if valid and ready:
                received = AXISMessage(msg_name = self.name,
                          msg_type=ContentType.SINGLE,
                          msg_id=self.stats.received_transactions,
                          data=[idata],
                          last=last,
                          strb = 0)
                self.i32_recv_buf = np.concatenate((self.i32_recv_buf, np.array([idata], dtype=np.int32)))
                if self.debug:
                    self.dut._log.info(f"recieved data {self.i32_recv_buf[-1]}, last: {received.last}")
                self._recv(received)

class AXISDriver(BusDriver):
  def __init__(self, dut, name, clk, debug=False):
    self.debug = debug
    self._signals = ['axis_tvalid', 'axis_tready', 'axis_tlast', 'axis_tdata','axis_tstrb']
    BusDriver.__init__(self, dut, name, clk)
    self.clock = clk
    self.dut = dut
    self.bus.axis_tdata.value = 0
    self.bus.axis_tstrb.value = 0
    self.bus.axis_tlast.value = 0
    self.bus.axis_tvalid.value = 0
    self.n_sent = 0
    self.i32_sent_buf = np.array([])

  async def _driver_send(self, msg, sync=True):
    if msg.msg_id == -1:
        msg.msg_id = self.n_sent

    if msg.msg_type == ContentType.SINGLE:
        await RisingEdge(self.clock)
        self.bus.axis_tvalid.value = 1
        self.bus.axis_tdata.value = msg.data[0]
        self.bus.axis_tlast.value = msg.last
        if self.debug:
            self.dut._log.info(f"driver sent data {msg.data[0]}")
        await ClockCycles(self.clock, 1)
        while not self.bus.axis_tready:
            await FallingEdge(self.clock)
        self.bus.axis_tvalid.value = 0
        self.bus.axis_tlast.value = 0
        self.n_sent += 1
        self.i32_sent_buf = np.concatenate((self.i32_sent_buf, np.array([msg.data[0]], dtype=np.int32)))

    elif msg.msg_type == ContentType.BURST:
        for i in range(len(msg.data)):
            await RisingEdge(self.clock)
            self.bus.axis_tvalid.value = 1
            self.bus.axis_tdata.value = msg.data[i]
            self.bus.axis_tlast.value = i == len(msg.data)-1
            if self.debug:
                self.dut._log.info(f"driver sent data burst packet {msg.data[i]}")
            await ClockCycles(self.clock, 1)
            while not self.bus.axis_tready:
                await FallingEdge(self.clock)
            self.bus.axis_tvalid.value = 0
            self.bus.axis_tlast.value = 0
        self.i32_sent_buf = np.concatenate((self.i32_sent_buf, np.array(msg.data, dtype=np.int32)))

        self.n_sent += 1

class AXISModuleTester(ABC):
    def __init__(self, dut, modulemodel, scoreboard_class = Scoreboard, debug=False):
        self.dut = dut

        self.expectedbuf = []
        self.model = lambda x: self.expectedbuf.append(modulemodel(x))

        self.inm = AXISMonitor(dut, 's00', dut.s00_axis_aclk,debug=debug, callback=self.model)
        self.ind = AXISDriver(dut, 's00', dut.s00_axis_aclk, debug=debug)
        self.outm = AXISMonitor(dut, 'm00', dut.m00_axis_aclk, debug=debug)

        self.scoreboard = scoreboard_class(self.dut, fail_immediately=False)
        self.scoreboard.add_interface(self.outm, self.expectedbuf)


    def _init_clks(self):
        cocotb.start_soon(generate_clock(self.dut.s00_axis_aclk))
        cocotb.start_soon(generate_clock(self.dut.m00_axis_aclk))

    async def startup(self):
        self._init_clks()
        self.dut.m00_axis_tready.value = 1
        self.dut.m00_axis_aresetn.value = 1
        self.dut.s00_axis_aresetn.value = 1
        await reset(self.dut.s00_axis_aresetn, self.dut.s00_axis_aclk)
        await reset(self.dut.m00_axis_aresetn, self.dut.s00_axis_aclk)
        await ClockCycles(self.dut.s00_axis_aclk, 500)

    def stop(self):
        self.inm.stop()
        self.outm.stop()
        self.ind.stop()

    @abstractmethod
    def gen_indata(self):
        pass

    async def run_io_test(self, n_io_pairs, singles = False):
        if singles:
            for _ in range(n_io_pairs - 1):
                msg = AXISMessage.nullsingle()
                msg.data = [self.gen_indata()]
                self.ind.append(msg)
                await ClockCycles(self.dut.s00_axis_aclk, 1)
            msg = AXISMessage.nullsingle()
            msg.data = [self.gen_indata()]
            msg.last = 1
            self.ind.append(msg)
            await ClockCycles(self.dut.s00_axis_aclk, 1)
        else:
            msg = AXISMessage.nullburst()
            msg.data = [self.gen_indata() for _ in range(n_io_pairs)]
            msg.last = 1
            self.ind.append(msg)
            await ClockCycles(self.dut.s00_axis_aclk, 2*n_io_pairs)
