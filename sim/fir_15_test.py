import numpy as np
import cocotb
import os
import sys
import logging
from pathlib import Path

from cocotb.utils import get_sim_time as gst
from cocotb.runner import get_runner
from cocotb.triggers import Timer,  RisingEdge, FallingEdge, ReadOnly,with_timeout, First, Join

from theloni_simlib.axi import *
from theloni_simlib.ctb_util import *

NCOEFFS = 15

@cocotb.test()
async def test_ir(dut):
    """cocotb test for basic axis module"""
    inm = AXISMonitor(dut,'s00',dut.s00_axis_aclk)
    outm = AXISMonitor(dut,'m00',dut.s00_axis_aclk, debug=True)
    ind = AXISDriver(dut,'s00',dut.s00_axis_aclk)
    cocotb.start_soon(generate_clock(dut.s00_axis_aclk))
    cocotb.start_soon(generate_clock(dut.m00_axis_aclk))

    dut.m00_axis_tready.value = 1
    await reset(dut.s00_axis_aresetn, dut.s00_axis_aclk)
    # await reset(dut.m00_axis_aresetn, dut.m00_axis_aclk)
    for i in range(NCOEFFS + 5):
        msg = AXISMessage.nullsingle()
        ind.append(msg)
        await ClockCycles(dut.s00_axis_aclk, 1)
    # put an impulse into the fir
    msg = AXISMessage.nullsingle()
    msg.data = [1]
    ind.append(msg)
    await ClockCycles(dut.s00_axis_aclk, 1)

    for i in range(NCOEFFS + 5):
        msg = AXISMessage.nullsingle()
        ind.append(msg)
        await ClockCycles(dut.s00_axis_aclk, 1)
    msg = AXISMessage.nullsingle()
    msg.last = 1
    ind.append(msg)
    await ClockCycles(dut.s00_axis_aclk, 1)

    # make sure we get the known impulse response back
    
    # -2, -3, -4, 0, 9, 21, 32, 36 ...symm
    
    await ClockCycles(dut.s00_axis_aclk, 1000)
    dut._log.info(f"Processed {inm.stats.received_transactions} transactions in and {outm.stats.received_transactions} out")
    assert inm.stats.received_transactions==outm.stats.received_transactions, f"Transaction Count doesn't match! :/"

"""the code below should largely remain unchanged in structure, though the specific files and things
specified should get updated for different simulations.
"""
def test_runner(svname, subdir):
    """Simulate the counter using the Python runner."""
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(proj_path / "sim" / "model"))
    sources = [proj_path / "hdl" / subdir / f"{svname}.sv"] #grow/modify this as needed.
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
    test_runner("fir_15", "filtering")