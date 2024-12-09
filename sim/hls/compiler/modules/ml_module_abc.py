from abc import ABC
from typing import *
from dataclasses import dataclass
import numpy as np

@dataclass
class Var:
    name: str
    defined: bool
    num_elements: int # 0 elements indicates that it is a logic
    n_bits: int
    tie_zero: bool

    def define(self):
        if self.defined:
            return ""
        self.defined = True
        elements_dcl = f"{f'[{self.num_elements-1}:0]' if self.num_elements > 1 else ''}[{self.n_bits-1}:0]"
        return f"logic {elements_dcl if self.num_elements else ''} {self.name};"

    def __repr__(self):
        return "0" if self.tie_zero else self.name

    def debug(self):
        return f"{vars(self)}"
@dataclass
class BRAMFile:
    fname: str
    buffer: np.ndarray
    @property
    def path(self):
        return f"../sim/data/{self.fname}.mem"
    def __repr__(self):
        return self.fname

class MLModule(ABC):
    name: str
    instance_num: int
    in_nodes: list
    out_nodes: list
    visited: bool # for dfs traversal

    in_vec_size: int
    working_regs: int
    nbits: int
    wfile: BRAMFile | None
    bfile: BRAMFile | None

    clk_in: Var
    rst_in: Var

    in_data_ready: Var
    write_out_data: Var
    in_data: Var
    req_chunk_in: Var
    req_chunk_out: Var
    out_vec_valid: Var

    def __init__(self, in_nodes:list, out_nodes:list, instance_num:int, nbits:int=8):
        self.name = f"{self.__class__.__name__.lower()}_{instance_num}"
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.instance_num = instance_num
        self.visited = False
        self.nbits = nbits
        self.out_vec_valid = Var(f"{self.name}_out_vec_valid_{instance_num}", False, 0, 1, False)

    def __repr__(self):
        return f"{self.name}:(1x{self.in_vec_size}) [{list(v for v in self.variables)}]"

    def systemverilog(self):
        raise NotImplementedError

    @property
    def variables(self):
        return [self.in_data_ready, self.in_data,
                self.write_out_data,
                self.req_chunk_in, self.req_chunk_out,
                self.out_vec_valid]

    @property
    def is_source(self):
        return self.in_nodes is []

    @property
    def is_sink(self):
        # handles recurrent relations by defining a node with already visited outputs as terminal
        if self.out_nodes is []:
            return True
        onodes_visited = True
        for node in self.out_nodes:
            onodes_visited &= node.visited
        return onodes_visited
