import logging
import math
import os
import sys
from pathlib import Path

import cocotb
import numpy as np
import onnx
from cocotb.runner import get_runner
from cocotb.triggers import (FallingEdge, First, Join, ReadOnly, RisingEdge,
                             Timer, with_timeout)
from cocotb.utils import get_sim_time as gst
from hls.compiler import parsemodel, svimpl
from onnx import TensorProto
from onnx import helper as onnxhelp
from bespoketb.bespoke import *
from bespoketb.boards import EpsScoreboard
from bespoketb.ctb_util import *


class VWBMACTester(BspkModuleTester):
    def __init__(self, dut, modelpath, in_dim, debug, **kwargs):
        board = EpsScoreboard(0.1, debug, dut)  # debug
        super().__init__(dut, modelpath, in_dim, board, True, debug, **kwargs)
        self.outm.mon_in = False  # monitor output and manually request read

    def _gen_indata(self):
        # d = np.ones(self.in_dim).astype(dtype=np.float32)#np.random.rand(self.in_dim).astype(dtype=np.float32)
        d = np.random.randn(self.in_dim).astype(np.float)
        return d / np.max(np.abs(d))


CYCLES = 8 * 11 * (int(math.ceil(math.log2(48))) + 1)


@cocotb.test()
async def test_vwb_macc(dut):
    """cocotb test for vwbmacc"""
    tlen = int(os.getenv("TLEN"))
    mdl_dim = int(os.getenv("MDIM"))
    mdl_path = os.getenv("MPATH")
    debug = bool(os.getenv("DEBUG"))
    test = VWBMACTester(dut, mdl_path, mdl_dim, debug)
    await test.startup()

    await test.run_io_test(tlen)

    await ClockCycles(dut.clk_in, tlen * CYCLES)
    dut._log.info(
        f"Processed {test.inm.stats.received_transactions} transactions in and {test.outm.stats.received_transactions} out"
    )
    assert (
        test.inm.stats.received_transactions == test.outm.stats.received_transactions
    ), f"Transaction Count doesn't match! :/"

    dut._log.info(
        f"maximum float err between recieved and expected: {test.scoreboard.maxerr}"
    )


"""the code below should largely remain unchanged in structure, though the specific files and things
specified should get updated for different simulations.
"""


def test_runner(svname, mdl_dim, mdl_path, tlen, debug):
    """Simulate the counter using the Python runner."""
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent

    sys.path.append(str(proj_path / "sim" / "model"))

    sources = [proj_path / "hdl" / "dummymodels" / f"{svname}_dummy_tl.sv"]
    sources.append(proj_path / "hdl" / "interface/v_fifo.sv")
    sources.append(proj_path / "hdl" / "mlops/vwb_gemm.sv")
    sources.append(proj_path / "hdl" / "mlops/mac1d.sv")
    sources.append(proj_path / "hdl" / "interface/xilinx_single_port_ram_read_first.v")
    sources.append(proj_path / "hdl" / "mlops/addertree.sv")

    print("\n ### SOURCES ### \n ", [s.name for s in sources], "\n")
    build_test_args = ["-Wimplicit"]
    parameters = {}
    sys.path.append(str(proj_path / "sim"))
    runner = get_runner(sim)
    runner.build(
        verilog_sources=sources,
        hdl_toplevel=f"{svname}_dummy_tl",
        always=True,
        build_args=build_test_args,
        parameters=parameters,
        timescale=("1ns", "1ps"),
        waves=True,
        verbose=True,
    )
    test_env = {"MDIM": mdl_dim, "MPATH": mdl_path, "TLEN": tlen, "DEBUG": debug}
    runner.test(
        hdl_toplevel=f"{svname}_dummy_tl",
        test_module=f"{svname}_test",
        extra_env=test_env,
        verbose=True,
        waves=True,
    )


def make_random_gemm_onnx(in_dim, out_dim):
    X = onnxhelp.make_tensor_value_info(f"X", TensorProto.FLOAT, [1, in_dim])
    Y = onnxhelp.make_tensor_value_info(f"Y", TensorProto.FLOAT, [1, out_dim])

    A = onnxhelp.make_tensor(
        f"A",
        TensorProto.FLOAT,
        [in_dim, out_dim],
        # np.eye(in_dim, out_dim))
        0.1 * np.random.randn(in_dim, out_dim),
    )
    bias = onnxhelp.make_tensor(
        f"B",
        TensorProto.FLOAT,
        [out_dim],
        # np.ones(out_dim))
        0.2 * np.random.randn(out_dim),
    )

    gemm_node = onnxhelp.make_node("Gemm", ["X", "A", "B"], ["Y"], name="onnx_gemm")
    grf = onnxhelp.make_graph([gemm_node], "testgraph", [X], [Y], [A, bias])

    opset = onnx.OperatorSetIdProto()
    opset.version = 19
    mdl = onnxhelp.make_model(grf, opset_imports=[opset])

    onnx.checker.check_model(mdl)
    onnx.shape_inference.infer_shapes(
        mdl, check_type=True, strict_mode=True, data_prop=True
    )
    return mdl


def make_testing_tl(idim, odim, tlname):
    mdl = make_random_gemm_onnx(idim, odim)
    spec = svimpl.FPGASpec(120, 600_000, 2_700_000, 100_000)
    fpga_module = parsemodel.gen_chain_sv_top(mdl, idim, spec)
    fpga_module.alloc_regs()
    fpga_module.alloc_bram("sim")
    fpga_module.tlname = tlname
    sv = fpga_module.make_sv()

    return mdl, sv


if __name__ == "__main__":
    tlname = "vwb_gemm_dummy_tl"
    IDIM = 48
    ODIM = 11
    LEN = 25
    DEBUG = False
    uut_mdl, sv = make_testing_tl(IDIM, ODIM, tlname)
    onnx.save_model(uut_mdl, "sim/data/gemm_mdl.onnx")
    with open(f"hdl/dummymodels/{tlname}.sv", "w") as f:
        f.write(sv)
    # exit(0)
    test_runner(
        "vwb_gemm",
        str(IDIM),
        os.path.abspath("sim/data/gemm_mdl.onnx"),
        str(LEN),
        str(DEBUG),
    )
