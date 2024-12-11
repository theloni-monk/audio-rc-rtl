from .sv_module_abc import *

class VW_Matmul(SVModule):

    out_vec_size: int
    weightfile: str
    bfile: BRAMFile

    req_chunk_ptr_rst: Var

    def __init__(self, inodes, onodes, instance_num):
        super().__init__(inodes, onodes, instance_num)

    @property
    def variables(self):
        return super().variables + [self.req_chunk_ptr_rst]

    def systemverilog(self):
        defs = [v.define() for v in self.variables]
        defs.append(
f"""vw_matmul #(
  .InVecLength({self.in_vec_size}),
  .OutVecLength({self.out_vec_size}),
  .WorkingRegs({self.working_regs}),
  .AccumBits({self.nbits}),
  .WeightFile("{self.wfile}"))
  {self.name}
  (
  .clk_in({self.clk_in}),
  .rst_in({self.rst_in}),
  .in_data_ready({self.in_data_ready}),
  .in_data({self.in_data}),
  .write_out_data({self.write_out_data}),
  .req_chunk_in({self.req_chunk_in}),
  .req_chunk_out({self.req_chunk_out}),
  .req_chunk_ptr_rst({self.req_chunk_ptr_rst}),
  .out_vector_valid({self.out_vec_valid})
    );
""")
        return "\n".join(defs)
