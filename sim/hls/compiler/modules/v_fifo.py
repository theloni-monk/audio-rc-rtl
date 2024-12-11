from .sv_module_abc import *

class V_FIFO(SVModule):
    elements_per_read: int
    elements_per_write: int
    depth: int

    ptr_rst: Var

    def __init__(self, i_nodes, o_nodes, instance_num, vec_size, elements_per_read, elements_per_write, depth, clk_in, rst_in):
        super().__init__(i_nodes, o_nodes, instance_num)
        self.in_vec_size = vec_size
        self.elements_per_read = elements_per_read
        self.elements_per_write = elements_per_write
        self.depth = depth

        self.clk_in = clk_in
        self.rst_in = rst_in

        self.in_data_ready = Var("dummy_fifo_in_data_ready", True, 0, 1, True)

        self.req_chunk_in = Var(f"wr_en_{instance_num}", False, 0, 1, False)
        self.in_data = Var(f"wr_data_{instance_num}", False, elements_per_read, self.nbits, False)
        self.req_chunk_out = Var(f"rd_en_{instance_num}", False, 0, 1, False)
        self.write_out_data = Var(f"rd_data_{instance_num}", False, elements_per_write, self.nbits, False)

        self.ptr_rst = Var(f"ptr_rst_{instance_num}", False, 0, 1, False)
        self.working_regs = -1

    @property
    def variables(self):
        return super().variables + [self.ptr_rst]

    def systemverilog(self):
        # define captured vars
        defs = [v.define() for v in self.variables]
        defs.append(
f"""v_fifo #(
        .VecElements({self.in_vec_size}),
        .ElementsPerRead({self.elements_per_read}),
        .ElementsPerWrite({self.elements_per_write}),
        .NBits({self.nbits}),
        .Depth({self.depth}))
        {self.name}
        (
            .clk_in({self.clk_in}),
            .rst_in({self.rst_in}),
            .wr_en({self.req_chunk_in}),
            .wr_data({self.in_data}),
            .rd_en({self.req_chunk_out}),
            .rd_data({self.write_out_data}),
            .ptr_rst({self.ptr_rst})
        );
""")

        # write instantiation
        return "\n".join(defs)
