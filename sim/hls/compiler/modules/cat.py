from .sv_module_abc import *


class Cat2(SVModule):
    elements_per_read: int
    elements_per_write0: int
    elements_per_write1: int

    def __init__(self, i_nodes, o_nodes, instance_num, cattreedepth,
                 ivec_size0, ivec_size1, elements_per_read, elements_per_write0, elements_per_write1, clk_in, rst_in):
        self.cattreedepth = cattreedepth # used to differentiate indep vars
        super().__init__(i_nodes, o_nodes, instance_num)
        self.in_vec_size = (ivec_size0, ivec_size1)

        self.elements_per_read = elements_per_read
        self.elements_per_write0 = elements_per_write0
        self.elements_per_write1 = elements_per_write1

        self.clk_in = clk_in
        self.rst_in = rst_in

        # moot, gets both vecs in one chunk each for simplicity
        self.req_chunk_in0 = Var(f"wr_en_{instance_num}_0", False, 0, 1, False)
        self.req_chunk_in1 = Var(f"wr_en_{instance_num}_1", False, 0, 1, False)

        self.in_data0 = Var(f"wr_data_{instance_num}_0", False, elements_per_read, self.nbits, False)
        self.in_data1 = Var(f"wr_data_{instance_num}_1", False, elements_per_read, self.nbits, False)
        self.req_chunk_out = Var(f"rd_en_{instance_num}", False, 0, 1, False)

        self.collision_err = Var(f"cat2collisionerr_{instance_num}", False, 0, 1, False)
        self.working_regs = -1

    @property
    def variables(self):
        return [self.in_data_ready0, self.in_data0,
                self.in_data_ready1, self.in_data1,
                self.write_out_data,
                self.req_chunk_in0, self.req_chunk_in1, self.req_chunk_out,
                self.out_vec_valid]

    def systemverilog(self):
        defs = [v.define() for v in self.variables]
        # greedy lut usage for simplicity
        defs += f""" cat #(
            .NBits({self.nbits}),
            .VecElements0({self.in_vec_size[0]}),
            .ElementsPerWrite0({self.elements_per_write0}),
            .VecElements1({self.in_vec_size[1]}),
            .ElementsPerWrite1({self.elements_per_write1}),
            .ElementsPerRead({self.elements_per_read})
        ) {self.name} (
    .clk_in({self.clk_in}),
    .rst_in({self.rst_in}),
    .wr_en0({self.in_data_ready0}),
    .wr_data0({self.in_data0}),
    .wr_en1({self.in_data_ready1}),
    .wr_data1({self.in_data1}),
    .rd_en({self.req_chunk_out}),
    .rd_data({self.write_out_data}),
    .cat_valid({self.out_vec_valid}),
    .wr_collision({self.collision_err})
    );

"""

class Cat(SVModule):
    elements_per_read: int
    elements_per_write_per_vec: List[int]

    #WRITEME: decompose into Cat2 Instances, with no fifos between them
    
    def __init__(self, in_nodes, out_nodes, instance_num,
                 ivec_sizes: List[int], elements_per_read, elements_per_write_per_vec: List[int], clk_in, rst_in):
        super().__init__(in_nodes, out_nodes, instance_num)
        
