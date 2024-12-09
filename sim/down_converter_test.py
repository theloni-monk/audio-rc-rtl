import math
import os
import sys
from pathlib import Path
from struct import *

import cocotb
import matplotlib.pyplot as plt
import numpy as np
from cocotb.runner import get_runner
from cocotb.triggers import (FallingEdge, First, Join, ReadOnly, RisingEdge,
                             Timer, with_timeout)
from cocotb.utils import get_sim_time as gst

from theloni_simlib.axi import *

from theloni_simlib.boards import *

DEBUG = False

def sig2iq(sig):
    i = ((sig>>16)&0xFFFF).astype(np.int16)
    q = (sig&0xFFFF).astype(np.int16)
    return i, q

class IQTester(AXISModuleTester):
    def __init__(self, dut, testlen, debug=False):
        self.testlen = testlen
        self.i = 0
        super().__init__(dut, lambda x: None, scoreboard_class=DummyScorebaord, debug=debug) # no model or scoreboard
        fs = 100e6 #sampling frequency
        n = testlen  #number of samples
        T = n*1.0/fs #total time
        fc = 10e6 #carrier frequency
        cps = 8 #cycles per symbol
        sps = fs/fc*cps #samples per symbol
        t = np.linspace(0, T, n, endpoint=False) #time vector in seconds
        ns = np.linspace(0,fs,n,endpoint=False) #sample vector
        phase_noise = np.arange(len(t))/len(t) * 6.28 #phase ranges from 0 to 2pi over the duration
        self.samples = (500*np.cos(10e6*2*np.pi*t+phase_noise)).astype(np.int32)

    def gen_indata(self):
        self.i = (self.i+1) % self.testlen
        return int(self.samples[self.i])

@cocotb.test()
async def test_dconvert(dut):
    """cocotb test for basic axis module"""
    tlen = 1024
    test = IQTester(dut, tlen, debug=True)
    test.ind.debug = True
    await test.startup()

    await test.run_io_test(test.testlen)

    await ClockCycles(dut.s00_axis_aclk, tlen+1000)

    plt.plot(test.inm.i32_recv_buf, label="recv_in")
    plt.plot(test.ind.i32_sent_buf, label="sent_in")
    plt.plot(test.samples, label="intended")
    plt.legend()
    plt.show()

    i,q = sig2iq(test.outm.i32_recv_buf.astype(np.int32))
    plt.plot(i, label="I")
    plt.plot(q, label="Q")
    plt.legend()
    plt.show()

    dut._log.info(f"Processed {test.inm.stats.received_transactions} transactions in and {test.outm.stats.received_transactions} out")
    assert test.inm.stats.received_transactions==test.outm.stats.received_transactions, f"Transaction Count doesn't match! :/"

"""the code below should largely remain unchanged in structure, though the specific files and things
specified should get updated for different simulations.
"""
def test_runner(svname):
    """Simulate the counter using the Python runner."""
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(proj_path / "sim" / "model"))
    sources = [proj_path / fname for fname in [os.path.relpath(os.path.join(root, file), '.') for root, _, files in os.walk('./hdl') for file in files] ] #grow/modify this as needed.
    build_test_args = ["-Wall"] #,"COCOTB_RESOLVE_X=ZEROS"]
    parameters = {}
    sys.path.append(str(proj_path / "sim"))
    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel=svname,
        always=True,
        build_args=build_test_args,
        parameters=parameters,
        timescale = ('1ns','1ps'),
        waves=True,
        verbose=True
    )
    run_test_args = []
    runner.test(
        hdl_toplevel=svname,
        test_module=f"{svname}_test",
        test_args=run_test_args,
        waves=True
    )

if __name__ == "__main__":
    test_runner("down_converter")