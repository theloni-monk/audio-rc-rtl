import pathlib
import numpy as np
import cocotb
import os
import sys
import logging
from pathlib import Path

from cocotb.utils import get_sim_time as gst
from cocotb.runner import get_runner
from cocotb.triggers import Timer,  RisingEdge, FallingEdge, ReadOnly,with_timeout, First, Join

import onnx
from onnx import helper as onnxhelp
from onnx import TensorProto

from hls.compiler import parsemodel, svimpl

from theloni_simlib.bespoke import *
from theloni_simlib.ctb_util import *

from fixedpoint import FixedPoint



@cocotb.test()
async def test_vwb_macc(dut):
    """cocotb test for vwbmacc"""
    # instantiate fifos
    # instantiate uut
    # wrap fifos in monitors and driver

    inm = BspkFIFOMonitor(dut.v_fifo_0, "main_in_mon", dut.clk_in, dut.in_data_ready)
    outm = BspkFIFOMonitor(dut.v_fifo_1, "main_out_mon", dut.clk_in,  dut.out_data_valid)
    ind = BspkFIFODriver(dut.v_fifo_0,  "main_in_drv", dut.clk_in,  dut.in_data_ready)

    cocotb.start_soon(generate_clock(dut.clk_in))
    await reset(dut.rst_in, dut.clk_in)

    ind.append(np.zeros(6))

    await ClockCycles(dut.clk_in, 1000)
    dut._log.info(f"Processed {inm.stats.received_transactions} transactions in and {outm.stats.received_transactions} out")
    assert inm.stats.received_transactions==outm.stats.received_transactions, f"Transaction Count doesn't match! :/"

"""the code below should largely remain unchanged in structure, though the specific files and things
specified should get updated for different simulations.
"""
def test_runner(svname):
    """Simulate the counter using the Python runner."""
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent

    sys.path.append(str(proj_path / "sim" / "model"))

    sources = [proj_path / "hdl" / "dummymodels" / f"{svname}_dummy_tl.sv"]
    sources.append(proj_path / "hdl" / "interface/v_fifo.sv")
    sources.append(proj_path / "hdl" / "mlops/vwb_mac.sv")
    sources.append(proj_path / "hdl" / "mlops/mac1d.sv")
    sources.append(proj_path / "hdl" / "interface/xilinx_single_port_ram_read_first.v")

    print("\n ### SOURCES ### \n ", [s.name for s in  sources], "\n")
    build_test_args = ["-Wall"]
    parameters = {}
    sys.path.append(str(proj_path / "sim"))
    runner = get_runner(sim)
    runner.build(
        verilog_sources=sources,
        hdl_toplevel=f"{svname}_dummy_tl",
        always=True,
        build_args=build_test_args,
        parameters=parameters,
        timescale = ('1ns','1ps'),
        waves=True,
        verbose=True
    )
    run_test_args = []
    runner.test(
        hdl_toplevel=f"{svname}_dummy_tl",
        test_module=f"{svname}_test",
        test_args=run_test_args,
        verbose=True,
        waves=True
    )

def make_random_bn_onnx(in_dim, out_dim):
    X = onnxhelp.make_tensor_value_info(f"X", TensorProto.FLOAT, [1, in_dim])
    Y = onnxhelp.make_tensor_value_info(f"Y", TensorProto.FLOAT, [1, out_dim])
    rmean = onnxhelp.make_tensor(   f"rmean", TensorProto.FLOAT, 
                                    [in_dim],
                                    -np.ones(in_dim))
                                    # np.random.rand(in_dim))
    rvar = onnxhelp.make_tensor(    f"rvar", TensorProto.FLOAT, 
                                    [in_dim],
                                    np.ones(in_dim))
                                    # np.random.rand(in_dim))

    scale = onnxhelp.make_tensor(   f"scale", TensorProto.FLOAT, 
                                    [in_dim],
                                    np.ones(out_dim))
    bias = onnxhelp.make_tensor(    f"B", TensorProto.FLOAT, 
                                    [in_dim],
                                    np.ones(in_dim))

    bn_node = onnxhelp.make_node("BatchNormalization",
                            ["X", "scale", "B", "rmean", "rvar"],
                            ["Y"],
                            name="onnx_bn")
    grf = onnxhelp.make_graph([bn_node], "testgraph",
                            [X],
                            [Y],
                            [scale, bias, rmean, rvar])

    opset = onnx.OperatorSetIdProto()
    opset.version = 19
    mdl = onnxhelp.make_model(grf, opset_imports = [opset])

    onnx.checker.check_model(mdl)
    onnx.shape_inference.infer_shapes(mdl, check_type=True, strict_mode=True, data_prop=True)
    return mdl

def make_testing_tl(dim, tlname):
    mdl = make_random_bn_onnx(dim,dim)
    spec = svimpl.FPGASpec(120, 600_000, 2_700_000, 100_000)
    fpga_module = parsemodel.parse_model(mdl, dim, spec)
    fpga_module.alloc_regs()
    fpga_module.alloc_bram("sim")
    fpga_module.tlname = tlname
    sv = fpga_module.make_sv()

    return mdl, sv

if __name__ == "__main__":
    tlname = "vwb_mac_dummy_tl"
    uut_mdl, sv = make_testing_tl(6, tlname)
    
    with open(f"hdl/dummymodels/{tlname}.sv", 'w') as f:
        f.write(sv)
    test_runner("vwb_mac") #TODO pass onnx model for valid
