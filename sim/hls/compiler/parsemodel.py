import onnx
from collections import Counter
import numpy as np
from .svimpl import SVImpl
from .modules import *
from .modules.ml_module_abc import BRAMFile

def tp2np(tprotobuf):
    arr = onnx.numpy_helper.to_array(tprotobuf)
    return arr

def to_ml_module(nodename):
    # Translates from onnx op_type to corresponding fpga module class
    return {"BatchNormalization": VWB_MAC,
            "Gemm" : VWB_Gemm,
            "MatMul": VW_Matmul}[nodename]

EPS = 1e-5
def parse_model(onnx_model, initdim,  spec):
    weights = onnx_model.graph.initializer
    wnames = map(lambda w: w.name, weights)
    wdict = dict(zip(wnames, weights))
    # print("B", tp2np(wdict["B"]))
    # print("WDICT", [(k, onnx.numpy_helper.to_array(wdict[k]).T.shape) for k in wdict.keys()])

    instancecounter = Counter()

    fpga_module = SVImpl(onnx_model, spec)
    currdim = initdim

    # shitty non-recursive construction
    fpga_module.modules = [V_FIFO([], [], 0, initdim, -1, -1, 1, fpga_module.clk, fpga_module.rst)]
    num_fifos = 1
    for idx, node in enumerate(onnx_model.graph.node):
        try:
            mod = to_ml_module(node.op_type)([fpga_module.modules[-1]], [], instancecounter[node.op_type])
        except KeyError: # the node has no systemverilog representation
            raise NotImplementedError

        instancecounter[node.op_type] += 1
        mod.in_vec_size = currdim

        if node.op_type == "BatchNormalization":
            s, b, rm, rv = map(tp2np, (wdict[node.input[i]] for i in range(1, 5)))
            vscaler = s*(np.sqrt(rv + EPS) ** -1)
            effbias = b - vscaler*rm
            mod.wfile = BRAMFile(f"{node.name}_weight", vscaler)
            mod.bfile = BRAMFile(f"{node.name}_bias", effbias)
        elif node.op_type == "MatMul" or node.op_type=="Gemm":
            mod.wfile = BRAMFile(f"{node.name}_weight",
                                 tp2np(wdict[node.input[1]]).T)
            newdim = wdict[node.input[1]].dims[-1]
            mod.out_vec_size = newdim
            mod.req_chunk_ptr_rst = fpga_module.modules[-1].wrap_rd
            currdim = newdim
            if node.op_type=="Gemm":
                mod.bfile = BRAMFile(f"{node.name}_bias",
                                        tp2np(wdict[node.input[2]]))
        else: # only matmul and gemm require a wrap_rd wire
            fpga_module.modules[-1].wrap_rd.tie_zero = True

        mod.clk_in = fpga_module.clk
        mod.rst_in = fpga_module.rst
        mod.in_data = fpga_module.modules[-1].write_out_data
        mod.req_chunk_in = fpga_module.modules[-1].req_chunk_out

        fpga_module.modules[-1].out_nodes.append(mod)
        fpga_module.modules.append(mod)
        fifo = V_FIFO([mod], [], num_fifos, currdim, -1, -1, 1, fpga_module.clk, fpga_module.rst)
        mod.out_nodes.append(fifo)
        num_fifos += 1
        mod.req_chunk_out = fifo.req_chunk_in
        mod.write_out_data = fifo.in_data

        fpga_module.modules.append(fifo)
        if idx > 0:
            fpga_module.modules[-2].in_data_ready = fpga_module.modules[-4].out_vec_valid
        else:
            fpga_module.modules[-2].in_data_ready = fpga_module.in_data_ready
    fpga_module.modules[-1].wrap_rd.tie_zero = True
    return fpga_module