import onnx
from collections import Counter
import numpy as np
from .svimpl import *
from .modules import *
from .modules.sv_module_abc import BRAMFile

def tp2np(tprotobuf):
    arr = onnx.numpy_helper.to_array(tprotobuf)
    return arr.T

def to_ml_module(nodename):
    # Translates from onnx op_type to corresponding fpga module class
    return {"BatchNormalization": VWB_MAC,
            "LeakyRelu": V_LeakyReLU,
            "Gemm" : VWB_Gemm,
            "MatMul": VW_Matmul,
            "Concat": Cat}[nodename]

"""feature description
its kind of hacky and does not maximize efficiency, but for simplicity's sake i will be.
processing all of the modules as if they were in a chain.
All chains will terminate at join, clone, or split operator and will be wrapped in a fifo on each end.
clone operator: one-to-many cloning into new fifos also creates combinational clones of the valid signal
join operator: Cat class implements syncronization via binary tree decomposition, reduces out valids to one signal
"""

def join_nodes(chains: List[SVModNode]) -> SVModNode:
    pass

def clone_node(node: SVModNode, n_clone:int):
    pass

def split_node(node: SVModNode, splits:List[int]) -> List[SVModNode]:
    pass

def clean_name(fname):
    return copy.deepcopy(fname).replace("/", "_")

EPS = 1e-5
def gen_chain_sv_top(onnx_model, initdim,  spec):
    weights = onnx_model.graph.initializer
    wnames = map(lambda w: w.name, weights)
    wdict = dict(zip(wnames, weights))

    instancecounter = Counter()

    fpga_module = SVImplTop(onnx_model, spec)
    currdim = initdim

    # shitty non-recursive construction
    fpga_module.modules = [V_FIFO([], [], 0, initdim, -1, -1, 1, fpga_module.clk, fpga_module.rst)]
    num_fifos = 1
    for idx, node in enumerate(onnx_model.graph.node):
        try:
            curr_mod = to_ml_module(node.op_type)([fpga_module.modules[-1]], [], instancecounter[node.op_type])
        except KeyError: # the node has no systemverilog representation
            raise NotImplementedError

        instancecounter[node.op_type] += 1
        curr_mod.in_vec_size = currdim

        if node.op_type == "BatchNormalization":
            s, b, rm, rv = map(tp2np, (wdict[node.input[i]] for i in range(1, 5)))
            vscaler = s*(np.sqrt(rv + EPS) ** -1)
            effbias = b - vscaler*rm
            curr_mod.wfile = BRAMFile(f"{clean_name(node.name)}_weight", vscaler)
            curr_mod.bfile = BRAMFile(f"{clean_name(node.name)}_bias", effbias)
        elif node.op_type == "MatMul" or node.op_type=="Gemm":
            curr_mod.wfile = BRAMFile(f"{clean_name(node.name)}_weight",
                                 tp2np(wdict[node.input[1]]).T)
            newdim = wdict[node.input[1]].dims[0]
            curr_mod.out_vec_size = newdim
            curr_mod.req_chunk_ptr_rst = fpga_module.modules[-1].ptr_rst
            currdim = newdim
            if node.op_type=="Gemm":
                curr_mod.bfile = BRAMFile(f"{clean_name(node.name)}_bias",
                                        tp2np(wdict[node.input[2]]))
        else: # only matmul and gemm require a ptr_rst wire
            fpga_module.modules[-1].ptr_rst.tie_zero = True

        curr_mod.clk_in = fpga_module.clk
        curr_mod.rst_in = fpga_module.rst

        prev_fifo = fpga_module.modules[-1]

        curr_mod.in_data = prev_fifo.write_out_data
        curr_mod.req_chunk_in = prev_fifo.req_chunk_out

        prev_fifo.out_nodes.append(curr_mod)
        fpga_module.modules.append(curr_mod)

        post_fifo = V_FIFO([curr_mod], [], num_fifos, currdim, -1, -1, 1, fpga_module.clk, fpga_module.rst)
        curr_mod.out_nodes.append(post_fifo)
        num_fifos += 1
        curr_mod.req_chunk_out = post_fifo.req_chunk_in
        curr_mod.write_out_data = post_fifo.in_data

        fpga_module.modules.append(post_fifo)
        if idx > 0:
            fpga_module.modules[-2].in_data_ready = fpga_module.modules[-4].out_vec_valid
        else:
            fpga_module.modules[-2].in_data_ready = fpga_module.in_data_ready
    fpga_module.modules[-1].ptr_rst.tie_zero = True
    return fpga_module
