import numpy as np
import cocotb
import os
import sys
import logging
from pathlib import Path
from cocotb.utils import get_sim_time as gst
from cocotb.runner import get_runner
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, ReadOnly,with_timeout, First, Join

async def drive_data_in(dut, data:np.int32):
    while not dut.sel_out.value:
        await RisingEdge(dut.clk_in)
    dut.data_in.value = data
    dut.trigger_in.value = True
    await ClockCycles(dut.clk_in, 1)
    dut.trigger_in.value = False
    await FallingEdge(dut.sel_out)
    dut._log.info("drove %d into module", data,)

async def reset(dut):
    dut.rst_in.value = 1
    await ClockCycles(dut.clk_in, 5)
    dut.rst_in.value = 0

async def model_spi_device(dut, received_messages):
    dclk_edges = 0
    while True:
        await FallingEdge(dut.sel_out) # starts being busy
        temp_int = 0
        for i in range(dut.DATA_WIDTH.value): # while busy
            await RisingEdge(dut.data_clk_out) # data is readable
            temp_int = temp_int << 1
            temp_int |= dut.data_out.value
            dclk_edges += 1
            # dut._log.info("tmp int %d, sel_out %d", temp_int, dut.sel_out)
        assert dclk_edges % dut.DATA_WIDTH.value == 0, f"Number of dataclk edges {dclk_edges} is not a multiple of the datawidth {dut.DATA_WIDTH.value}"
        received_messages.append(temp_int) # append messages
        dut._log.info("pushed recieved int %d onto q", temp_int)

async def assert_spi_clk(dut):
    while True:
        void = False
        await RisingEdge(dut.data_clk_out) # when data valid
        for i in range(dut.DATA_PERIOD.value):
            await ClockCycles(dut.clk_in, 1)
            if dut.rst_in:
                void = True
                break
            assert not dut.data_clk_out or void, "data clk rose too early"
        assert dut.data_clk_out or void, "data clk rose too late"

async def generate_clock(clock_wire):
	while True: # repeat forever
		clock_wire.value = 0
		await Timer(5,units="ns")
		clock_wire.value = 1
		await Timer(5,units="ns")

@cocotb.test()
async def rint_spi_tx_test(dut):
    await cocotb.start( generate_clock( dut.clk_in ) ) #launches clock
    await reset(dut)
    message_in = 0
    messages_out = []
    cocotb.start_soon(model_spi_device(dut, messages_out))
    cocotb.start_soon(assert_spi_clk(dut))
    await ClockCycles(dut.clk_in, 100)
    for _ in range(5):
        message_in = np.random.randint(2**dut.DATA_WIDTH.value - 1)
        await drive_data_in(dut, message_in)
        await RisingEdge(dut.sel_out)
        await ClockCycles(dut.clk_in, 30)
        assert messages_out[-1] == message_in, f"Message put in {message_in} was recieved out as {messages_out[-1]}"

"""the code below should largely remain unchanged in structure, though the specific files and things
specified should get updated for different simulations.
"""
def test_runner():
    """Simulate the counter using the Python runner."""
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(proj_path / "sim" / "model"))
    sources = [proj_path / "hdl" / "spi_tx.sv"] #grow/modify this as needed.
    build_test_args = ["-Wall"] #,"COCOTB_RESOLVE_X=ZEROS"]
    parameters = {}
    sys.path.append(str(proj_path / "sim"))
    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="spi_tx",
        always=True,
        build_args=build_test_args,
        parameters=parameters,
        timescale = ('1ns','1ps'),
        waves=True
    )
    run_test_args = []
    runner.test(
        hdl_toplevel="spi_tx",
        test_module="test_spi_tx",
        test_args=run_test_args,
        waves=True
    )

if __name__ == "__main__":
    test_runner()