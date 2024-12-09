from pathlib import Path
from typing import List
from dataclasses import dataclass
from functools import reduce

def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
import onnx

from .codebram import gen_bram_file
from .modules import *
from .modules.ml_module_abc import *

@dataclass
class FPGASpec():
    num_dsp: int
    num_reg: int
    total_bram: int
    max_clock: int

#TODO: fixed point bram

def next_smallest_factor(vec_size, max_factor):
    largest_under = 1
    for i in sorted(factors(vec_size)):
        if i < max_factor:
            largest_under = i
        else:
            break
    return largest_under

class SVImpl():

    model: onnx.ModelProto

    in_dim: int
    out_dim: int
    spec: FPGASpec
    avail_bram: int

    clk: Var
    rst: Var
    in_data_ready: Var
    modules: List[MLModule]

    def __init__(self, model, spec, tlname = "ml_inf"):
        self.tlname = tlname
        self.model = model
        self.clk = Var("clk_in", True, 0, 1, False)
        self.rst = Var("rst_in", True, 0, 1, False)
        self.spec = spec
        self.avail_bram = spec.total_bram
        self.in_data_ready = Var("in_data_ready", True, 0, 1, False)

    def alloc_bram(self, path):
        for mod in self.modules:
            if isinstance(mod, VWB_MAC):
                gen_bram_file(Path(path) / f"data/{mod.wfile.fname}.mem", mod.working_regs, mod.wfile.buffer)
                gen_bram_file(Path(path) / f"data/{mod.bfile.fname}.mem", mod.working_regs, mod.bfile.buffer)

                self.avail_bram -= mod.nbits * (len(mod.wfile.buffer) + len(mod.bfile.buffer))
            #WRITEME: allocate correct from class
            # elif type(mod) == Bias:
            #     mod.biasfile = f"data/{mod.name}_bfile.mem"
            #     gen_bram_file(f"data/{mod.name}_bfile.mem", mod.working_regs, mod.bram)
            #     self.avail_bram -= mod.bram.size * 8
        assert self.avail_bram > 0, "Insufficient BRAM"

    def alloc_regs(self):
        num_mult_mods = sum(1 if isinstance(mod, (VW_Matmul, VWB_MAC)) else 0 for mod in self.modules)
        mult_cycles = self.modules[2].in_vec_size * self.modules[1].in_vec_size // next_smallest_factor(self.modules[1].in_vec_size, self.spec.num_dsp // num_mult_mods + 1)
        for mod in self.modules:
            if isinstance(mod, (VW_Matmul, VWB_MAC)):
                mod.working_regs = next_smallest_factor(mod.in_vec_size, self.spec.num_dsp//num_mult_mods + 1)
                mod.write_out_data.num_elements = 1
            else:
                mod.working_regs = max(1, mod.in_vec_size // mult_cycles)
                mod.write_out_data.num_elements = mod.working_regs
            mod.in_data.num_elements = mod.working_regs

        self.modules[0].elements_per_read = self.modules[1].working_regs
        self.modules[0].elements_per_write = 1
        self.modules[0].in_data.num_elements = 1
        for idx, fifo in enumerate(self.modules):
            if idx == 0 or idx == len(self.modules)-1:
                continue
            if not isinstance(fifo, V_FIFO):
                continue
            fifo.elements_per_write = 1 if type(self.modules[idx-1]) == VW_Matmul else self.modules[idx-1].working_regs
            if type(self.modules[idx-1]) == VW_Matmul:
                fifo.in_data.num_elements = 1
            fifo.elements_per_read = self.modules[idx+1].working_regs
        self.modules[-1].elements_per_write = self.modules[-2].working_regs
        self.modules[-1].elements_per_read = self.modules[-1].in_vec_size
        self.modules[-1].write_out_data.num_elements = self.modules[-1].in_vec_size

    def make_sv(self):
        input_fifo = self.modules[0]
        input_fifo.elements_per_write = self.modules[1].in_vec_size
        input_fifo.req_chunk_in.name = "in_data_ready"
        input_fifo.req_chunk_in.defined = True

        input_fifo.in_data.name = "in_data_top"
        input_fifo.in_data.defined = True
        input_fifo.clk_in = Var("clk_in", True, 0, 1, False)
        input_fifo.rst_in = Var("rst_in", True, 0, 1, False)

        output_fifo = self.modules[-1]
        output_fifo.req_chunk_out.name = "rd_out_top"
        output_fifo.req_chunk_out.defined = True
        
        output_fifo.write_out_data.name = "out_data_top"
        output_fifo.write_out_data.defined = True

        output_fifo.clk_in = Var("clk_in", True, 0, 1, False)
        output_fifo.rst_in = Var("rst_in", True, 0, 1, False)

        first_proc_node = self.modules[1]
        first_proc_node.in_data_ready.name = "in_data_ready"
        first_proc_node.in_data_ready.defined = True

        last_proc_node = self.modules[-2]
        last_proc_node.out_vec_valid.name = "out_data_valid"
        last_proc_node.out_vec_valid.defined = True
        #chr(92) is a newline character
        return f"""`timescale 1ps/1ps
`default_nettype none
module {self.tlname} (
    input wire clk_in,
    input wire rst_in,
    input wire in_data_ready,
    input wire [{self.modules[1].in_vec_size-1}:0][{self.modules[0].nbits}-1:0] in_data_top,
    input wire rd_out_top,
    output logic out_data_valid,
    output logic [{self.modules[1].in_vec_size-1}:0][{self.modules[0].nbits}-1:0] out_data_top
);

{input_fifo.systemverilog()}

{output_fifo.systemverilog()}

{chr(92).join([mod.systemverilog() for mod in self.modules[1:-1]])}

endmodule;
"""