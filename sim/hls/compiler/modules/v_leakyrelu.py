from .sv_module_abc import *

class V_LeakyReLU(SVModule):

    def __init__(self, inodes, onodes, instance_num):
        super().__init__(inodes, onodes, instance_num)

    def systemverilog(self):
        defs = [v.define() for v in self.variables]
        defs.append(f"""v_leakyrelu #(
  .InVecLength({self.in_vec_size}),
  .NBits({self.nbits}),
  .WorkingRegs({self.working_regs})) {self.name} (
  .clk_in({self.clk_in}),
  .rst_in({self.rst_in}),
  .in_data_ready({self.in_data_ready}),
  .in_data({self.in_data}),
  .write_out_data({self.write_out_data}),
  .req_chunk_in({self.req_chunk_in}),
  .req_chunk_out({self.req_chunk_out}),
  .out_vector_valid({self.out_vec_valid})
            );""")
        return """
""".join(defs)

